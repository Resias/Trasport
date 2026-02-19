import numpy as np
import math
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
    Chebyshev polynomial approximation.
    T_0(L) = I
    T_1(L) = L
    T_k(L) = 2L·T_{k-1}(L) - T_{k-2}(L)
    """
    N = L.size(0)
    device, dtype = L.device, L.dtype
    
    cheb = []
    cheb.append(torch.eye(N, device=device, dtype=dtype))  # T_0
    
    if K == 1:
        return torch.stack(cheb, dim=0)
    
    cheb.append(L.clone())  # T_1
    
    for k in range(2, K):
        T_k = 2.0 * (L @ cheb[-1]) - cheb[-2]
        cheb.append(T_k)
    
    return torch.stack(cheb[:K], dim=0)  # (K, N, N)

def build_scaled_laplacian(adj: np.ndarray):
    """
    Build scaled Laplacian for Chebyshev approximation
    논문: L_scaled = 2/λ_max * L - I
    """
    A = adj.astype(np.float32)
    N = A.shape[0]
    
    # Normalized Laplacian
    D = np.diag(A.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-6))
    L = np.eye(N) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # ✅ Spectral scaling for Chebyshev (논문 faithful)
    eigvals = np.linalg.eigvalsh(L)  # symmetric matrix
    lambda_max = eigvals.max()
    
    L_scaled = (2.0 / lambda_max) * L - np.eye(N)
    
    return torch.tensor(L_scaled, dtype=torch.float32)


class TemporalProjector(nn.Module):
    """
    Project OD matrix to temporal latent tokens
    """
    def __init__(self, num_regions, feature_dim, d_model):
        super().__init__()
        self.in_dim = num_regions * num_regions * feature_dim
        self.proj = nn.Linear(self.in_dim, d_model)

    def forward(self, M):
        # M: (B,T,N,N,F)
        B, T, N, _, F = M.shape
        x = M.view(B, T, -1)      # (B,T,N*N*F)
        return self.proj(x)       # (B,T,d_model)

class ODAttention(nn.Module):
    def __init__(self, num_regions, feature_dim, hidden_dim=128):
        super().__init__()
        self.N = num_regions
        self.f = feature_dim
        
        # 1911차원 -> 128차원(hidden_dim)으로 압축하여 연산
        self.proj_dim = hidden_dim 
        
        # [수정 2] Input Projection Layer (차원 축소용)
        # (N * F) -> (proj_dim)
        self.input_proj_o = nn.Linear(self.N * self.f, self.proj_dim)
        self.input_proj_d = nn.Linear(self.N * self.f, self.proj_dim)

        # [수정 3] 가중치 파라미터 크기 축소
        # 기존: (N, N, N*F) -> 약 7.7억개 (GPU 터짐)
        # 변경: (N, N, proj_dim) -> 약 2,600만개 (현실적)
        self.W1_o = nn.Parameter(torch.randn(self.N, self.N, self.proj_dim))
        self.b_omega = nn.Parameter(torch.zeros(self.N, self.N))
        
        self.W1_d = nn.Parameter(torch.randn(self.N, self.N, self.proj_dim))
        self.b_delta = nn.Parameter(torch.zeros(self.N, self.N))

        # 초기화 (학습 안정성을 위해)
        nn.init.xavier_uniform_(self.W1_o)
        nn.init.xavier_uniform_(self.W1_d)

    def _sparse_mask(self, scores):
        """
        Implements entropy-based sparse selection.
        Selects top-u queries with smallest entropy (most informative).
        """
        # scores: (BT, N, N)
        # 1. Calculate Probability Distribution
        p = F.softmax(scores, dim=-1) # (BT, N, N)

        # 2. Calculate Shannon Entropy: H = - sum(p * log(p))
        entropy = -torch.sum(p * torch.log(p + 1e-9), dim=-1) # (BT, N)

        # 3. Select Top-u smallest entropy (u = ln(N))
        u = max(1, int(math.log(self.N)))
        
        # Indices of smallest entropy (most distinct/dominant queries)
        _, top_idx = torch.topk(entropy, k=u, dim=1, largest=False) # (BT, u)

        # 4. Create Mask
        mask = torch.zeros_like(scores, dtype=torch.float)
        # Scatter 1s to the selected top-u indices
        mask.scatter_(1, top_idx.unsqueeze(-1).expand(-1, -1, self.N), 1.0)
        
        return mask
    def forward(self, M):
        B, T, N, _, Fdim = M.shape
        BT = B * T
        
        # ------------------------------------------------
        # Origin Attention
        # ------------------------------------------------
        # 1. 입력 벡터 구성: (BT, N, N*F)
        X = M.reshape(BT, N, N*Fdim)
        
        # [수정 4] 차원 축소 (Projection)
        # (BT, N, N*F) -> (BT, N, proj_dim)
        X_proj = self.input_proj_o(X) 
        
        # 2. Attention Score 계산 (축소된 벡터 사용)
        # 최적화된 연산 (Loop 대신 einsum 사용 가능하지만, 메모리 안전을 위해 Loop 유지)
        # X_proj[:, i]: (BT, proj_dim)
        # W1_o[i, j]:   (proj_dim)
        omega_raw = torch.einsum('bid,ijd->bij', X_proj, self.W1_o)
        omega_raw = omega_raw + self.b_omega
        
        # 3. Masking & Softmax
        mask_o = self._sparse_mask(omega_raw)
        omega = F.softmax(torch.sigmoid(omega_raw), dim=-1) * mask_o
        
        # ------------------------------------------------
        # Destination Attention
        # ------------------------------------------------
        # 1. 입력 벡터 구성 (Transpose)
        Y = M.permute(0, 1, 3, 2, 4).reshape(BT, N, N*Fdim)
        
        # [수정 4] 차원 축소
        Y_proj = self.input_proj_d(Y)
        
        # 2. Attention Score 계산
        delta_raw = torch.einsum('bid,ijd->bij', Y_proj, self.W1_d)
        delta_raw = delta_raw + self.b_delta
        
        # 3. Masking & Softmax
        mask_d = self._sparse_mask(delta_raw)
        delta = F.softmax(torch.sigmoid(delta_raw), dim=-1) * mask_d

        # ------------------------------------------------
        # Final Aggregation: M_A = Omega * M * Delta
        # ------------------------------------------------
        M_flat = M.view(BT, N, N, Fdim)
        out = torch.einsum('bio,bojf,bjd->bidf', omega, M_flat, delta)
        
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
    def __init__(self, top_k, mov_avg_win=25):
        super().__init__()
        self.top_k = top_k
        self.mov_avg = mov_avg_win

    def forward(self, S):
        # S: (B, T, N, N, F)
        B, T, N, _, F = S.shape
        
        # Flatten OD dimensions
        S_flat = S.view(B, T, -1)  # (B, T, N*N*F)
        
        # ✅ Detrending (Eq.10)
        trend = centered_moving_average(S_flat, self.mov_avg)
        detrended = S_flat - trend
        
        # ✅ FFT-based autocorrelation (Eq.11)
        nfft = 1 << (2 * T - 1).bit_length()
        fx = torch.fft.rfft(detrended, n=nfft, dim=1)
        ac = torch.fft.irfft(fx * torch.conj(fx), n=nfft, dim=1)[:, :T, :]
        
        # ✅ Average over all OD pairs
        ac_avg = ac.mean(dim=-1)  # (B, T)
        
        # ✅ Ignore lag 0 (self-correlation)
        ac_avg[:, 0] = -1e9
        
        # ✅ Top-k periods
        periods = torch.topk(ac_avg, self.top_k, dim=1).indices  # (B, k)
        
        return periods

class PeriodSparseSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, periods: torch.Tensor):
        """
        x: (B, T, D)
        periods: (B, k) where k == num_heads
        """
        B, T, D = x.shape
        k = self.num_heads
        
        # Q, K, V projections
        qkv = self.qkv_proj(x)  # (B, T, 3D)
        qkv = qkv.reshape(B, T, 3, k, self.head_dim)  # (B,T,3,k,d_h)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3,B,k,T,d_h)
        q, k_mat, v = qkv[0], qkv[1], qkv[2]  # each (B,k,T,d_h)
        
        # ✅ Build period-sparse masks (per sample, per head)
        masks = []
        for b in range(B):
            for h in range(k):
                p = max(int(periods[b, h].item()), 1)
                i_idx = torch.arange(T, device=x.device).unsqueeze(1)
                j_idx = torch.arange(T, device=x.device).unsqueeze(0)
                
                causal = (j_idx <= i_idx)
                periodic = ((i_idx - j_idx) % p) == 0
                valid = causal & periodic
                
                mask = torch.zeros(T, T, device=x.device)
                mask[~valid] = float('-inf')
                masks.append(mask)
        
        attn_mask = torch.stack(masks).view(B, k, T, T)  # (B,k,T,T)
        
        # ✅ Scaled dot-product attention (per head)
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k_mat) / (self.head_dim ** 0.5)
        scores = scores + attn_mask  # (B,k,T,T)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Attend to values
        out = torch.einsum('bhqk,bhkd->bhqd', attn, v)  # (B,k,T,d_h)
        out = out.transpose(1, 2).reshape(B, T, D)  # (B,T,D)
        
        return self.out_proj(out)


class ODformerEncoderLayer(nn.Module):
    """
    Temporal-only encoder layer (for stacking)
    """
    def __init__(self, d_model, top_k, dropout=0.1):
        super().__init__()
        self.temporal = PeriodSparseSelfAttention(d_model, top_k, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, periods):
        # x: (B, T, d_model)
        # periods: (B, k)
        h = self.temporal(self.norm1(x), periods)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class ODformerDecoderLayer(nn.Module):
    def __init__(self, d_model, top_k, num_cross_heads, dropout=0.0):
        super().__init__()
        # ✅ PeriodSparse for decoder self-attention
        self.self_attn = PeriodSparseSelfAttention(d_model, top_k, dropout)
        
        # ✅ Separate head count for cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=num_cross_heads, dropout=dropout, batch_first=True
        )
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
    def __init__(
        self,
        num_regions: int,
        feature_dim: int,
        d_model: int = 128,
        hidden_dim: int = 64,
        alpha: float = 0.7,
        K_gcn: int = 3,
        top_k_periods: int = 4,
        num_heads: int = 8,
        mov_avg_win: int = 25,
        num_enc_layers: int = 2,
        num_dec_layers: int = 2,
        dropout: float = 0.1,
        pred_len: int = 192,
        out_feature_dim: int | None = None,
    ):
        super().__init__()
        self.N = num_regions
        self.F = feature_dim
        self.O = pred_len
        self.d_model = d_model
        self.out_F = out_feature_dim if out_feature_dim is not None else feature_dim

        # Shared Spatial Dependency Module (used by both Encoder and Decoder)
        self.spatial = SpatialDependency(
            num_regions, feature_dim, hidden_dim, alpha, K_gcn
        )
        
        self.temporal_proj = TemporalProjector(num_regions, feature_dim, d_model)
        self.period_extractor = PeriodicityExtractor(top_k_periods, mov_avg_win)
        
        self.encoder_layers = nn.ModuleList([
            ODformerEncoderLayer(d_model, top_k_periods, dropout)
            for _ in range(num_enc_layers)
        ])

        # 주의: ODformerDecoderLayer의 인자 이름(num_cross_heads)을 확인하세요.
        self.decoder_layers = nn.ModuleList([
            ODformerDecoderLayer(d_model=d_model, num_cross_heads=num_heads, top_k=top_k_periods, dropout=dropout)
            for _ in range(num_dec_layers)
        ])

        # Learnable Start Token (OD Matrix Shape)
        # Latent vector 대신 물리적인 OD Matrix 형태의 시작 토큰을 사용합니다.
        self.start_od_token = nn.Parameter(torch.zeros(1, 1, num_regions, num_regions, feature_dim))
        
        self.pos_emb_enc = nn.Parameter(torch.zeros(1, 2048, d_model))
        self.pos_emb_dec = nn.Parameter(torch.zeros(1, 2048, d_model))

        self.proj_out = nn.Linear(d_model, num_regions * num_regions * self.out_F)

    def forward(self, X: torch.Tensor, L_o: torch.Tensor, L_d: torch.Tensor):
        # X: (B, I, N, N, F)
        B, I, N, _, Fdim = X.shape
        
        # ------------------ ENCODER ------------------
        # 1. Spatial Dependency
        M_spatial_enc = self.spatial(X, L_o, L_d)
        
        # 2. Projection & Periodicity
        enc_tokens = self.temporal_proj(M_spatial_enc)
        enc_tokens = enc_tokens + self.pos_emb_enc[:, :I, :]
        periods = self.period_extractor(M_spatial_enc) 
        
        # 3. Temporal Layers
        for layer in self.encoder_layers:
            enc_tokens = layer(enc_tokens, periods)
        
        # ------------------ DECODER ------------------
        # 1. Prepare Input (Start Token + Zeros)
        # Shape: (B, O, N, N, F) - Matches physical OD shape
        # [Critical Fix] Decoder 입력도 Spatial Dependency를 통과해야 함 (Fig 3) 
        dec_input_od = self.start_od_token.expand(B, self.O, -1, -1, -1)
        
        # 2. Spatial Dependency for Decoder
        M_spatial_dec = self.spatial(dec_input_od, L_o, L_d)
        
        # 3. Projection
        y = self.temporal_proj(M_spatial_dec)
        y = y + self.pos_emb_dec[:, :self.O, :]
        
        # 4. Temporal Layers (PeriodSparse + Cross)
        for layer in self.decoder_layers:
            y = layer(y, enc_tokens, periods)
        
        # 5. Final Projection
        y_out = self.proj_out(y)
        y_out = y_out.view(B, self.O, N, N, self.out_F)
        
        return y_out