import torch
import torch.nn as nn
import torch.nn.functional as F


def centered_moving_average(x: torch.Tensor, win: int) -> torch.Tensor:
    """
    x: (B, T, D) or (B, T)
    centered moving average with padding.
    """
    if win <= 1:
        return x
    if x.dim() == 2:
        x = x.unsqueeze(-1)  # (B,T,1)
        squeeze_back = True
    else:
        squeeze_back = False

    B, T, D = x.shape
    pad = win // 2
    # pad on time dim
    x_pad = F.pad(x.transpose(1, 2), (pad, pad), mode="replicate").transpose(1, 2)  # (B,T+2p,D)
    # avg pool
    x_ma = F.avg_pool1d(x_pad.transpose(1, 2), kernel_size=win, stride=1).transpose(1, 2)  # (B,T,D)

    if squeeze_back:
        x_ma = x_ma.squeeze(-1)
    return x_ma

def autocorr_topk_periods(x: torch.Tensor, top_k: int, min_period: int = 2) -> torch.Tensor:
    """
    x: (B, T) time series (already detrended)
    Return: periods (B, top_k) in [min_period, T-1]
    Implements autocorrelation via FFT: r = ifft(fft(x)*conj(fft(x)))
    """
    B, T = x.shape
    # zero-mean
    x = x - x.mean(dim=1, keepdim=True)

    # FFT-based autocorr
    nfft = 1 << (2 * T - 1).bit_length()
    fx = torch.fft.rfft(x, n=nfft, dim=1)
    ac = torch.fft.irfft(fx * torch.conj(fx), n=nfft, dim=1)[:, :T]  # (B,T)

    # ignore lag 0 and too small periods
    ac[:, :min_period] = -1e9
    # pick top-k lags as periods
    periods = torch.topk(ac, k=top_k, dim=1).indices  # (B,top_k)
    return periods

def chebyshev_basis(L: torch.Tensor, K: int) -> torch.Tensor:
    """
    L: (N, N) normalized Laplacian
    Return: T (K, N, N) Chebyshev polynomials T_0..T_{K-1}
    """
    N = L.size(0)
    T_k = []
    T0 = torch.eye(N, device=L.device, dtype=L.dtype)
    if K == 1:
        return T0.unsqueeze(0)
    T1 = L
    T_k.append(T0)
    T_k.append(T1)
    for k in range(2, K):
        Tk = 2 * L @ T_k[-1] - T_k[-2]
        T_k.append(Tk)
    return torch.stack(T_k[:K], dim=0)  # (K,N,N)

class ODAttention(nn.Module):
    """
    Sparse OD Attention with Shannon Entropy query selection
    """
    def __init__(self, num_regions, feature_dim, hidden_dim):
        super().__init__()
        self.N = num_regions
        self.f = feature_dim
        self.hidden = hidden_dim

        self.origin_proj = nn.Linear(self.N * self.f, hidden_dim)
        self.dest_proj   = nn.Linear(self.N * self.f, hidden_dim)

        self.origin_score = nn.Linear(hidden_dim, self.N)
        self.dest_score   = nn.Linear(hidden_dim, self.N)

    def _sparse_attention(self, scores: torch.Tensor):
        """
        scores: (B*T, N, N)
        """
        # softmax
        p = F.softmax(scores, dim=-1)  # (BT,N,N)

        # Shannon entropy
        entropy = -torch.sum(p * torch.log(p + 1e-9), dim=-1)  # (BT,N)

        # top-u smallest entropy
        u = max(1, int(torch.log(torch.tensor(self.N, device=scores.device)).item()))
        top_idx = torch.topk(entropy, k=u, dim=1, largest=False).indices  # (BT,u)

        mask = torch.zeros_like(entropy, dtype=torch.bool)
        mask.scatter_(1, top_idx, True)  # True = keep

        # mask out non-dominant queries
        p = p * mask.unsqueeze(-1)
        return p

    def forward(self, M):
        """
        M: (B,T,N,N,F)
        """
        B, T, N, _, Fdim = M.shape
        BT = B * T

        X = M.reshape(BT, N, N*Fdim)
        Y = M.permute(0,1,3,2,4).reshape(BT, N, N*Fdim)

        # scores
        omega_scores = self.origin_score(self.origin_proj(X))  # (BT,N,N)
        delta_scores = self.dest_score(self.dest_proj(Y))      # (BT,N,N)

        # sparse attention
        omega = self._sparse_attention(omega_scores)
        delta = self._sparse_attention(delta_scores)

        M_flat = M.view(BT, N, N, Fdim)
        out = torch.einsum("bij,bjkl->bikl", omega, M_flat)
        out = torch.einsum("bikl,bjl->bijk", out, delta)

        return out.view(B, T, N, N, Fdim)


class TwoDGCN(nn.Module):
    """
    MGCNN-like 2D GCN using Chebyshev basis on origin/destination graphs.
    """
    def __init__(self, feature_dim: int, K: int = 3):
        super().__init__()
        self.K = K
        self.feature_dim = feature_dim
        # (K,K,F,F)
        self.theta = nn.Parameter(torch.randn(K, K, feature_dim, feature_dim) * 0.02)

    def forward(self, M: torch.Tensor, L_o: torch.Tensor, L_d: torch.Tensor) -> torch.Tensor:
        """
        M: (B,T,N,N,F)
        L_o, L_d: (N,N) normalized Laplacian
        return: (B,T,N,N,F)
        """
        B, T, N, _, Fdim = M.shape
        assert Fdim == self.feature_dim

        To = chebyshev_basis(L_o, self.K)  # (K,N,N)
        Td = chebyshev_basis(L_d, self.K)  # (K,N,N)

        out = torch.zeros_like(M)
        # Sum_{i,j} T_i(L_o) * M * T_j(L_d) * W_ij
        for i in range(self.K):
            for j in range(self.K):
                tmp = torch.einsum(
                    "ij,btjkf,kl->btikf",
                    To[i], M, Td[j]
                )  # (B,T,N,N,F)
                out = out + torch.einsum("btacf,fg->btacg", tmp, self.theta[i, j])  # linear on F
        return out
    
class SpatialDependency(nn.Module):
    def __init__(self, num_regions: int, feature_dim: int, hidden_dim: int, alpha: float, K: int = 3):
        super().__init__()
        self.alpha = alpha
        self.od_attn = ODAttention(num_regions, feature_dim, hidden_dim)
        self.gcn = TwoDGCN(feature_dim, K=K)

    def forward(self, M: torch.Tensor, L_o: torch.Tensor, L_d: torch.Tensor) -> torch.Tensor:
        MA = self.od_attn(M)
        MG = self.gcn(M, L_o, L_d)
        return (1.0 - self.alpha) * MG + self.alpha * MA


class PeriodicityExtractor(nn.Module):
    """
    Implements Eq (10)-(11):
      detrend with centered moving average, then autocorr top-k periods.
    """
    def __init__(self, top_k: int, mov_avg_win: int = 25, min_period: int = 2):
        super().__init__()
        self.top_k = top_k
        self.mov_avg_win = mov_avg_win
        self.min_period = min_period

    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        S: (B,T,N,N,F)
        Return periods: (B, top_k)
        """
        # Reduce OD matrices to a 1D proxy series (논문은 \hat{S_T}에 대해 autocorr)
        # 실전에서는 mean pooling이 안전한 기본값.
        x = S.mean(dim=(2, 3, 4))  # (B,T)

        trend = centered_moving_average(x, self.mov_avg_win)  # (B,T)
        period_comp = x - trend
        periods = autocorr_topk_periods(period_comp, self.top_k, min_period=self.min_period)
        return periods

class PeriodSparseSelfAttention(nn.Module):
    """
    Implements Eq (12)-(13) with causal sparse pattern:
      A_i^{(k)} = { j <= i : (i-j) mod P_k = 0 }
    We use one head per period (num_heads == top_k).
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, periods: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,D)
        periods: (B,k) period lengths
        """
        B, T, D = x.shape
        # Build attn_mask per batch (PyTorch MHA expects (T,T) or (B*num_heads,T,T))
        # We'll build (B,T,T) and expand to (B*num_heads,T,T).
        k = periods.size(1)
        assert self.mha.num_heads == k, "num_heads must equal top_k(period count) for period-specific heads."

        masks = []
        for b in range(B):
            # (k,T,T)
            head_masks = torch.ones(k, T, T, device=x.device, dtype=torch.bool)
            for h in range(k):
                p = int(periods[b, h].item())
                # allow j<=i and (i-j)%p==0
                i_idx = torch.arange(T, device=x.device).view(T, 1)
                j_idx = torch.arange(T, device=x.device).view(1, T)
                allow = (j_idx <= i_idx) & (((i_idx - j_idx) % max(p, 1)) == 0)
                head_masks[h] = ~allow  # True means masked
            masks.append(head_masks)
        attn_mask = torch.cat(masks, dim=0)  # (B*k, T, T)

        out, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return out


class ODformerEncoderLayer(nn.Module):
    def __init__(self, num_regions, feature_dim, hidden_dim, alpha, K_gcn, d_model, top_k, mov_avg_win, dropout=0.0):
        super().__init__()
        self.spatial = SpatialDependency(num_regions, feature_dim, hidden_dim, alpha, K=K_gcn)
        self.period = PeriodicityExtractor(top_k=top_k, mov_avg_win=mov_avg_win)
        self.temporal = PeriodSparseSelfAttention(d_model=d_model, num_heads=top_k, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # --- period cache ---
        self.register_buffer("step_counter", torch.zeros(1, dtype=torch.long))
        self.cached_periods = None
        self.Pmax = None

    def forward(self, M, L_o, L_d):
        # spatial on OD matrices
        M = self.spatial(M, L_o, L_d)  # (B,T,N,N,F)

        # flatten for temporal attention
        B, T, N, _, Fdim = M.shape
        x = M.view(B, T, -1)  # (B,T,D)

        # update period every Pmax
        if self.cached_periods is None:
            periods = self.period(M)
            self.cached_periods = periods.detach()
            self.Pmax = int(periods.max().item())
        elif self.step_counter.item() % self.Pmax == 0:
            periods = self.period(M)
            self.cached_periods = periods.detach()
        else:
            periods = self.cached_periods

        self.step_counter += 1

        # temporal
        h = self.temporal(self.norm1(x), periods)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, periods  # x is encoder token sequence (B,T,D)

class ODformerDecoderLayer(nn.Module):
    def __init__(self, d_model, top_k, dropout=0.0):
        super().__init__()
        self.self_attn = PeriodSparseSelfAttention(d_model=d_model, num_heads=top_k, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=top_k, dropout=dropout, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, y, enc, periods):
        # y: (B,O,D) decoder tokens
        y = y + self.self_attn(self.norm1(y), periods)
        y = y + self.cross_attn(self.norm2(y), enc, enc)[0]
        y = y + self.ffn(self.norm3(y))
        return y


class ODFormer(nn.Module):
    """
    Final ODformer:
      - Encoder: spatial + periodicity + period-sparse temporal
      - Decoder: period-sparse self-attn + cross-attn
      - Output: predict future OD matrices

    Args:
      pred_len: O
    """
    def __init__(
        self,
        num_regions: int,
        feature_dim: int,
        hidden_dim: int = 64,
        alpha: float = 0.7,
        K_gcn: int = 3,
        top_k_periods: int = 4,
        mov_avg_win: int = 25,
        num_enc_layers: int = 2,
        num_dec_layers: int = 2,
        dropout: float = 0.0,
        pred_len: int = 192,
        out_feature_dim: int | None = None,
    ):
        super().__init__()
        self.N = num_regions
        self.F = feature_dim
        self.O = pred_len
        self.top_k = top_k_periods

        self.d_model = num_regions * num_regions * feature_dim
        self.out_F = out_feature_dim if out_feature_dim is not None else feature_dim
        self.out_dim = num_regions * num_regions * self.out_F

        self.encoder_layers = nn.ModuleList([
            ODformerEncoderLayer(
                num_regions=num_regions,
                feature_dim=feature_dim,
                hidden_dim=hidden_dim,
                alpha=alpha,
                K_gcn=K_gcn,
                d_model=self.d_model,
                top_k=top_k_periods,
                mov_avg_win=mov_avg_win,
                dropout=dropout
            )
            for _ in range(num_enc_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            ODformerDecoderLayer(d_model=self.d_model, top_k=top_k_periods, dropout=dropout)
            for _ in range(num_dec_layers)
        ])

        # decoder input embedding: use zeros or learned start token
        self.start_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_emb_enc = nn.Parameter(torch.zeros(1, 2048, self.d_model))  # enough length
        self.pos_emb_dec = nn.Parameter(torch.zeros(1, 2048, self.d_model))

        self.proj_out = nn.Linear(self.d_model, self.out_dim)

    def forward(self, X: torch.Tensor, L_o: torch.Tensor, L_d: torch.Tensor) -> torch.Tensor:
        """
        X: (B, I, N, N, F)
        L_o, L_d: (N, N)
        Return:
          Y_hat: (B, O, N, N, out_F)
        """
        B, I, N, _, Fdim = X.shape
        assert N == self.N and Fdim == self.F

        # Encoder
        enc = X
        periods = None
        x_tokens = None

        # flatten + positional for encoder will happen after first layer outputs tokens
        for li, layer in enumerate(self.encoder_layers):
            x_tokens, periods = layer(enc, L_o, L_d)  # x_tokens: (B,I,D)
            # after first layer, work purely in token space; reshape back for next spatial layer is not defined in paper
            # -> Keep encoder as single spatial stage + multiple temporal layers is more stable.
            # 따라서 num_enc_layers>1일 때는 temporal stack으로 사용.
            if li == 0:
                x_tokens = x_tokens + self.pos_emb_enc[:, :I, :]
            enc = x_tokens  # token space

            # For next encoder temporal layer, we need a dummy M for period extractor; reuse same periods (paper updates every Pmax).
            # We'll keep periods constant across layers.
            break

        # If more temporal encoder layers desired, stack self-attn+ffn without re-running spatial/period extractor.
        for li in range(1, len(self.encoder_layers)):
            # re-use periods, run temporal+ffn only
            layer = self.encoder_layers[li]
            # emulate: periods computed once
            h = layer.temporal(layer.norm1(enc), periods)
            enc = enc + h
            enc = enc + layer.ffn(layer.norm2(enc))

        enc_tokens = enc  # (B,I,D)

        # Decoder input tokens: start token + zeros (teacher forcing 없을 때)
        O = self.O
        y = self.start_token.expand(B, 1, -1)
        if O > 1:
            y = torch.cat([y, torch.zeros(B, O-1, self.d_model, device=X.device, dtype=X.dtype)], dim=1)
        y = y + self.pos_emb_dec[:, :O, :]

        # Decoder layers
        for layer in self.decoder_layers:
            y = layer(y, enc_tokens, periods)

        # Project to OD matrix
        y_out = self.proj_out(y)  # (B,O,N*N*out_F)
        y_out = y_out.view(B, O, N, N, self.out_F)
        return y_out