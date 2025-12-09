import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ------------------------
# Utility: Normalized Adj
# ------------------------
def build_normalized_adj_from_df(od_df, make_symmetric=True, device="cpu"):
    A = (od_df.values > 0).astype(np.float32)

    if make_symmetric:
        A = np.logical_or(A, A.T).astype(np.float32)

    I = np.eye(A.shape[0], dtype=np.float32)
    A_hat = A + I

    deg = A_hat.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    D_inv_sqrt = np.diag(deg_inv_sqrt)

    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return torch.tensor(A_norm, dtype=torch.float32, device=device)


# ------------------------
# Simple GCN Layer
# ------------------------
class SimpleGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, A_norm):
        # x: [B*N, F_in], we reshape before calling this
        # But simpler: x is [B, N, F]. Let's keep original
        x_agg = torch.einsum("ij,bjf->bif", A_norm, x)  # [B, N, F]
        out = self.lin(x_agg)
        return F.relu(out)


# ---------------------------------------------------
# Time-aware GNN + GRU Seq2Seq OD Predictor
# ---------------------------------------------------
class MetroGNNForecaster(nn.Module):
    def __init__(
        self,
        od_df,
        in_feat=None,        # 기본적으로 역 개수와 동일한 차원을 가정
        gnn_hidden=64,
        rnn_hidden=64,
        out_feat=None,  # output feature dim (default same as input)
        weekday_emb_dim=8,
        time_emb_dim=2,  # time_enc_hist: already sin/cos = 2
        window_size=60,
        pred_size=30,
        device="cpu",
    ):
        super().__init__()

        self.N = od_df.shape[0]
        self.window_size = window_size
        self.pred_size = pred_size

        if in_feat is None:
            in_feat = self.N
        self.in_feat = in_feat

        if out_feat is None:
            out_feat = in_feat
        self.out_feat = out_feat

        # Register adjacency
        A_norm = build_normalized_adj_from_df(od_df, device=device)
        self.register_buffer("A_norm", A_norm)

        # --------------------
        # Condition embeddings
        # --------------------
        self.weekday_emb = nn.Embedding(7, weekday_emb_dim)

        # 시간 인코딩은 sin/cos 그대로 들어오기 때문에 Linear projection만 수행
        self.time_proj = nn.Linear(time_emb_dim, weekday_emb_dim)
        self.time_enc_dim = time_emb_dim

        # --------------------
        # GCN Input dimension
        # x_tensor: [N, N]  -> flatten or compress to [N, in_feat]
        #           여기서는 열 합(sum over destination) 등으로 inflow/outflow를 만들 수도 있지만
        #           단순화를 위해 row vector로 처리할 경우 in_feat = N
        # Condition: weekday_emb + time_proj
        # --------------------
        total_cond_dim = weekday_emb_dim + weekday_emb_dim  # weekday + projected time
        self.gcn_input_dim = in_feat + total_cond_dim

        # spatial encoder
        self.gcn = SimpleGCNLayer(self.gcn_input_dim, gnn_hidden)

        # temporal encoder GRU
        self.encoder_gru = nn.GRU(
            input_size=gnn_hidden,
            hidden_size=rnn_hidden,
            batch_first=True
        )

        # decoder GRU (condition only)
        self.decoder_gru = nn.GRU(
            input_size=weekday_emb_dim + weekday_emb_dim,  # weekday_emb + time_proj
            hidden_size=rnn_hidden,
            batch_first=True
        )

        # Output: node-level features → [N, out_feat]
        self.output_mlp = nn.Linear(rnn_hidden, out_feat)

    def forward(self, x, weekday_tensor, time_enc_hist, time_enc_fut):
        """
        x: [B, T_in, N, in_feat]  (실제 in_feat=N 또는 다른 값)
        weekday_tensor: [B]
        time_enc_hist: [B, T_in, 2]
        time_enc_fut:  [B, T_out, 2]
        """
        if x.dim() != 4:
            raise ValueError("x must be a 4D tensor: [B, T_in, N, F]")

        B, T_in, N, in_feat = x.shape
        device = x.device
        T_out = self.pred_size
        assert T_in == self.window_size, "window_size mismatch"
        assert T_out == time_enc_fut.size(1)
        if N != self.N:
            raise ValueError(
                f"Input node dimension ({N}) must match adjacency size ({self.N})."
            )
        if in_feat != self.in_feat:
            raise ValueError(
                f"Input feature dim ({in_feat}) must match model in_feat ({self.in_feat})."
            )

        weekday_tensor = weekday_tensor.to(device=device, dtype=torch.long)
        time_enc_hist = time_enc_hist.to(device=device, dtype=torch.float32)
        time_enc_fut = time_enc_fut.to(device=device, dtype=torch.float32)
        if time_enc_hist.size(-1) != self.time_enc_dim:
            raise ValueError(
                f"Expected time encoding dim {self.time_enc_dim}, got {time_enc_hist.size(-1)}"
            )

        # -------------------------
        # 1) Condition embeddings
        # -------------------------
        weekday_e = self.weekday_emb(weekday_tensor)  # [B, W]
        # expand: [B, T, W]
        weekday_hist = weekday_e.unsqueeze(1).expand(B, T_in, -1)
        weekday_fut  = weekday_e.unsqueeze(1).expand(B, T_out, -1)

        # time sin/cos -> projected embedding
        # time_enc_*: [B, T, 2]
        time_hist = self.time_proj(time_enc_hist)  # [B, T_in, W]
        time_fut  = self.time_proj(time_enc_fut)   # [B, T_out, W]

        # condition for encoder: concat weekday + time
        cond_hist = torch.cat([weekday_hist, time_hist], dim=-1)  # [B,T_in, 2W]
        cond_fut  = torch.cat([weekday_fut , time_fut ], dim=-1)  # [B,T_out,2W]

        # -------------------------
        # 2) Spatial GCN Encoding
        # -------------------------
        # We must concatenate condition to node features
        # x: [B, T_in, N, in_feat]
        cond_hist_node = cond_hist.unsqueeze(2).expand(B, T_in, N, cond_hist.size(-1))
        gcn_input = torch.cat([x, cond_hist_node], dim=-1)  # [B,T_in,N,in_feat+cond]

        # reshape for GCN time-step wise
        gcn_input = gcn_input.view(B*T_in, N, -1)  # [B*T, N, F_gcn]
        spatial_out = self.gcn(gcn_input, self.A_norm)       # [B*T, N, gnn_hidden]
        spatial_out = spatial_out.view(B, T_in, N, -1)        # [B,T_in,N,H]

        # temporal encoder input: flatten node dim
        enc_in = spatial_out.transpose(1, 2).reshape(B*N, T_in, -1)  # [B*N,T_in,H]
        _, h_enc = self.encoder_gru(enc_in)  # h_enc: [1,B*N,rnn_hidden]

        # -------------------------
        # 3) Temporal Decoder
        # -------------------------
        # decoder input: condition only, but node-level repetition needed
        dec_in = cond_fut.unsqueeze(2).expand(B, T_out, N, cond_fut.size(-1))
        dec_in = dec_in.reshape(B*N, T_out, -1)  # [B*N, T_out, 2W]

        dec_out, _ = self.decoder_gru(dec_in, h_enc)  # [B*N,T_out,rnn_hidden]
        dec_out = dec_out.reshape(B, N, T_out, -1).permute(0,2,1,3)  # [B,T_out,N,H]

        # -------------------------
        # 4) Node-wise Output MLP
        # -------------------------
        y_pred = self.output_mlp(dec_out)  # [B,T_out,N,out_feat]
        return y_pred

if __name__ == '__main__':
    print('GNN.py')