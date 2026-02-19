import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# Utils: batch-wise time shift (vectorized)
# =========================
def batch_time_shift(x: torch.Tensor, lag: torch.Tensor) -> torch.Tensor:
    """
    x:  (B, T, D)
    lag:(B,) int64, shift to the right by lag
    return: (B, T, D)
    """
    B, T, D = x.shape
    # indices[t] = (t - lag) mod T  -> right shift
    t = torch.arange(T, device=x.device).view(1, T)              # (1,T)
    idx = (t - lag.view(B, 1)) % T                                # (B,T)
    idx = idx.unsqueeze(-1).expand(B, T, D)                       # (B,T,D)
    return torch.gather(x, dim=1, index=idx)


# =========================
# Series Decomposition (moving average)
# =========================
class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size, stride=1)

    def forward(self, x):
        # x: (B,T,D)
        x_t = x.transpose(1, 2)                                   # (B,D,T)
        x_pad = F.pad(x_t, (self.pad, self.pad), mode="replicate")
        trend = self.avg(x_pad).transpose(1, 2)                   # (B,T,D)
        seasonal = x - trend
        return seasonal, trend


# =========================
# AutoCorrelation: FFT-based correlation + time-delay aggregation
# =========================
class AutoCorrelation(nn.Module):
    def __init__(self, d_model, top_k=8):
        super().__init__()
        self.top_k = top_k
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

    def forward(self, x):
        """
        x: (B,T,D)
        out: (B,T,D)
        """
        B, T, D = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # FFT length for correlation
        nfft = 1 << (2 * T - 1).bit_length()
        fq = torch.fft.rfft(q, n=nfft, dim=1)
        fk = torch.fft.rfft(k, n=nfft, dim=1)

        corr = torch.fft.irfft(fq * torch.conj(fk), n=nfft, dim=1)[:, :T]  # (B,T,D)
        score = corr.mean(dim=-1)                                          # (B,T)
        score[:, 0] = -1e9                                                 # exclude lag 0

        lags = torch.topk(score, k=min(self.top_k, T - 1), dim=1).indices  # (B,K)

        out = torch.zeros_like(v)
        K = lags.shape[1]
        for i in range(K):
            out = out + batch_time_shift(v, lags[:, i])
        return out / K


# =========================
# Encoder Layer (Autoformer-style)
# =========================
class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, ff_dim, kernel_size=25, top_k=8, dropout=0.0):
        super().__init__()
        self.decomp1 = SeriesDecomposition(kernel_size)
        self.decomp2 = SeriesDecomposition(kernel_size)

        self.attn = AutoCorrelation(d_model, top_k=top_k)
        self.dropout = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )

    def forward(self, x):
        # x: (B,T,D)
        seasonal, trend1 = self.decomp1(x)
        seasonal = seasonal + self.dropout(self.attn(seasonal))

        seasonal, trend2 = self.decomp2(seasonal)
        seasonal = seasonal + self.dropout(self.ff(seasonal))

        trend = trend1 + trend2
        return seasonal, trend


# =========================
# Decoder Layer (Autoformer-style)
#  - self AutoCorrelation on seasonal
#  - cross attention to encoder seasonal (practical, stable)
#  - decomposition + trend accumulation
# =========================
class AutoformerDecoderLayer(nn.Module):
    def __init__(self, d_model, ff_dim, kernel_size=25, top_k=8, nhead=8, dropout=0.0):
        super().__init__()
        self.decomp1 = SeriesDecomposition(kernel_size)
        self.decomp2 = SeriesDecomposition(kernel_size)
        self.decomp3 = SeriesDecomposition(kernel_size)

        self.self_attn = AutoCorrelation(d_model, top_k=top_k)

        # Cross: 안정적으로 encoder 정보를 주입하기 위해 MHA 사용 (실전에서 수렴 안정)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, seasonal, trend, enc_seasonal):
        """
        seasonal: (B, L, D)   decoder seasonal stream
        trend:    (B, L, D)   decoder trend stream
        enc_seasonal: (B, T_in, D)
        """
        # (1) self autocorr on seasonal
        s, t1 = self.decomp1(seasonal)
        s = s + self.dropout(self.self_attn(s))

        # (2) cross attention: decoder seasonal queries -> encoder seasonal keys/values
        s, t2 = self.decomp2(s)
        cross, _ = self.cross_attn(query=s, key=enc_seasonal, value=enc_seasonal)
        s = s + self.dropout(cross)

        # (3) feedforward
        s, t3 = self.decomp3(s)
        s = s + self.dropout(self.ff(s))

        # trend accumulation (Autoformer 핵심)
        trend = trend + (t1 + t2 + t3)
        return s, trend


# =========================
# Full Autoformer for OD (Low-rank heads)
# =========================
class AutoformerODFormal(nn.Module):
    """
    Inputs:
        x_log: (B, T_in, N, N)     log1p already recommended
        time_enc_hist: (B, T_in, C_time)  (optional)
        time_enc_fut:  (B, T_out, C_time) (optional)
        weekday: (B,) long (optional)

    Output:
        y_hat: (B, T_out, N, N)  log1p(relu(OD)) 형태
    """
    def __init__(
        self,
        num_nodes,
        d_model=128,
        ff_dim=256,
        enc_layers=2,
        dec_layers=2,
        pred_steps=6,
        rank=32,
        kernel_size=25,
        top_k=8,
        nhead=8,
        time_dim=0,          # time_enc channel 수, 없으면 0
        use_weekday=True,
        dropout=0.0,
    ):
        super().__init__()
        self.N = num_nodes
        self.rank = rank
        self.pred_steps = pred_steps
        self.time_dim = time_dim
        self.use_weekday = use_weekday

        in_dim = num_nodes * num_nodes + time_dim
        self.in_proj = nn.Linear(in_dim, d_model)

        if use_weekday:
            self.weekday_emb = nn.Embedding(7, d_model)

        self.encoder = nn.ModuleList([
            AutoformerEncoderLayer(d_model, ff_dim, kernel_size, top_k, dropout)
            for _ in range(enc_layers)
        ])

        self.decoder = nn.ModuleList([
            AutoformerDecoderLayer(d_model, ff_dim, kernel_size, top_k, nhead, dropout)
            for _ in range(dec_layers)
        ])

        # Low-rank OD heads
        self.origin_head = nn.Linear(d_model, num_nodes * rank)
        self.dest_head = nn.Linear(d_model, num_nodes * rank)

    def forward(self, x_log, time_enc_hist=None, time_enc_fut=None, weekday=None):
        B, T_in, N, _ = x_log.shape
        assert N == self.N, f"N mismatch: got {N}, expected {self.N}"

        # --------
        # Encoder input
        # --------
        x = x_log.reshape(B, T_in, -1)  # (B,T_in,N*N)

        if self.time_dim > 0:
            assert time_enc_hist is not None, "time_dim>0인데 time_enc_hist가 None입니다."
            x = torch.cat([x, time_enc_hist], dim=-1)  # (B,T_in,N*N + C_time)

        h_enc = self.in_proj(x)  # (B,T_in,D)

        if self.use_weekday:
            assert weekday is not None, "use_weekday=True인데 weekday가 None입니다."
            h_enc = h_enc + self.weekday_emb(weekday).unsqueeze(1)

        # --------
        # Encoder: accumulate seasonal/trend
        # --------
        seasonal = h_enc
        trend_enc_sum = torch.zeros_like(h_enc)
        for layer in self.encoder:
            seasonal, trend = layer(seasonal)
            trend_enc_sum = trend_enc_sum + trend

        enc_seasonal = seasonal                     # (B,T_in,D)
        enc_trend = trend_enc_sum                   # (B,T_in,D)

        # --------
        # Decoder init (정식 Autoformer 방식)
        # seasonal_init: [enc_seasonal, zeros(T_out)]
        # trend_init:    [enc_trend, repeat(last_trend, T_out)]
        # --------
        L = T_in + self.pred_steps

        seasonal_init = torch.zeros(B, self.pred_steps, enc_seasonal.size(-1), device=x_log.device)
        seasonal_dec = torch.cat([enc_seasonal, seasonal_init], dim=1)  # (B,L,D)

        last_trend = enc_trend[:, -1:, :]  # (B,1,D)
        trend_future = last_trend.repeat(1, self.pred_steps, 1)
        trend_dec = torch.cat([enc_trend, trend_future], dim=1)         # (B,L,D)

        # time encoding for decoder: (B,L,C_time) if used
        if self.time_dim > 0:
            assert time_enc_fut is not None, "time_dim>0인데 time_enc_fut가 None입니다."
            # 과거 time_enc_hist + 미래 time_enc_fut 를 decoder seasonal에 더해줌 (가벼운 주입)
            time_dec = torch.cat([time_enc_hist, time_enc_fut], dim=1)  # (B,L,C_time)
            time_proj = time_dec.new_zeros(B, L, seasonal_dec.size(-1))

        if self.use_weekday:
            seasonal_dec = seasonal_dec + self.weekday_emb(weekday).unsqueeze(1)
            trend_dec = trend_dec + self.weekday_emb(weekday).unsqueeze(1)

        # --------
        # Decoder layers
        # --------
        for layer in self.decoder:
            seasonal_dec, trend_dec = layer(seasonal_dec, trend_dec, enc_seasonal)

        # final output: seasonal + trend, take last T_out
        dec_out = (seasonal_dec + trend_dec)[:, -self.pred_steps:, :]  # (B,T_out,D)

        # --------
        # Low-rank OD reconstruction per step
        # --------
        Ho = self.origin_head(dec_out).view(B, self.pred_steps, N, self.rank)  # (B,T_out,N,r)
        Hd = self.dest_head(dec_out).view(B, self.pred_steps, N, self.rank)    # (B,T_out,N,r)

        od = torch.matmul(Ho, Hd.transpose(-1, -2))                            # (B,T_out,N,N)
        y_hat = torch.log1p(F.relu(od))
        return y_hat
