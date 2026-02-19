import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse

from torch_geometric.data import Data, Batch
from SCIE_Benchmark.ODFormer import PeriodSparseSelfAttention,centered_moving_average

# ---------------------------------------------
# Positional Encoding (Sinusoidal + Learnable)
# ---------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, hid_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.learnable = nn.Parameter(torch.zeros(max_len, hid_dim))
        self.register_buffer('sinusoidal', self._get_sinusoidal_pe(max_len, hid_dim))

    def _get_sinusoidal_pe(self, max_len, hid_dim):
        pe = torch.zeros(max_len, hid_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hid_dim, 2) *
                             -(math.log(10000.0) / hid_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x):
        # x: [seq_len, N, hid_dim]
        seq_len = x.size(0)
        pe = self.sinusoidal[:seq_len] + self.learnable[:seq_len]
        return x + pe.unsqueeze(1)


class NodePeriodicityExtractor(nn.Module):
    """
    Extract periods from node-level temporal embeddings
    Input:  (B*N, T, D)
    Output: (B*N, K)
    """
    def __init__(self, top_k, mov_avg_win=25):
        super().__init__()
        self.top_k = top_k
        self.mov_avg = mov_avg_win

    def forward(self, X):
        # X: (BN, T, D)
        BN, T, D = X.shape

        # 평균 feature로 대표 시계열
        x = X.mean(dim=-1)  # (BN, T)

        trend = centered_moving_average(x, self.mov_avg)
        detrended = x - trend

        # FFT-based autocorr
        nfft = 1 << (2 * T - 1).bit_length()
        fx = torch.fft.rfft(detrended, n=nfft, dim=1)
        ac = torch.fft.irfft(fx * torch.conj(fx), n=nfft, dim=1)[:, :T]

        ac[:, 0] = -1e9
        periods = torch.topk(ac, self.top_k, dim=1).indices
        return periods

# -----------------------------------------------
# GAT Encoder
# -----------------------------------------------
class GATEncoder(nn.Module):
    def __init__(self, in_channels, hid_channels, heads=4, edge_attr_dim=None):
        super(GATEncoder, self).__init__()
        
        self.gat1 = GATv2Conv(in_channels, hid_channels, heads=heads, dropout=0.2, edge_dim=edge_attr_dim)
        self.gat2 = GATv2Conv(hid_channels*heads, hid_channels, heads=1, concat=False, edge_dim=edge_attr_dim)

    def forward(self, x, edge_index, edge_attr=None):
        # GAT: 공간 특성 인코딩
        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr)
        return x
    
class StaticGATEncoder(nn.Module):
    def __init__(self, in_channels, hid_channels, layers=1, heads=4):
        super(StaticGATEncoder, self).__init__()
        self.gat1 = GATEncoder(in_channels, hid_channels, heads=heads)
        self.gat_embedder_layers = nn.ModuleList()
        for _ in range(layers):
            self.gat_embedder_layers.append(GATEncoder(hid_channels, hid_channels, heads=heads))

    def forward(self, x, edge_index):
        # static embedding 입력
        h = self.gat1(x, edge_index)
        h = F.elu(h)
        for gat_layer in self.gat_embedder_layers:
            h = gat_layer(h, edge_index)
            h = F.elu(h)
        return h             # shape = [num_nodes, hid_channels]
    
class DynamicGATEncoder(nn.Module):
    def __init__(self, in_channels, hid_channels, layers=2, heads=4, edge_attr_dim=1):
        super(DynamicGATEncoder, self).__init__()
        self.gat1 = GATEncoder(in_channels, hid_channels, heads=heads, edge_attr_dim=edge_attr_dim)
        self.gat_embedder_layers = nn.ModuleList()
        for _ in range(layers):
            self.gat_embedder_layers.append(GATEncoder(hid_channels, hid_channels, heads=heads, edge_attr_dim=edge_attr_dim))

    def forward(self, x, edge_index, edge_attr):
        h = self.gat1(x, edge_index, edge_attr)
        h = F.elu(h)
        for gat_layer in self.gat_embedder_layers:
            h = gat_layer(h, edge_index, edge_attr)
            h = F.elu(h)
        return h             # shape = [num_nodes, hid_channels]

# -----------------------------------------------
# Transformer Decoder Block
# -----------------------------------------------
class TimeTransformerDecoder(nn.Module):
    def __init__(self, hid_dim, num_heads, ff_dim, num_layers):
        super(TimeTransformerDecoder, self).__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hid_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation='relu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)

    def forward(self, tgt, memory, tgt_mask=None):
        out = self.transformer_decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask
        )
        return out

# ============================================================
# Period-aware Decoder Layer: (self-attn = period-sparse) + cross-attn + FFN
# ============================================================
class PeriodAwareDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_self_heads: int,
        num_cross_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = PeriodSparseSelfAttention(d_model, num_self_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=num_cross_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, periods: torch.Tensor) -> torch.Tensor:
        """
        tgt:    (B, O, D)
        memory: (B, T, D)
        periods:(B, H_self)
        """
        tgt = tgt + self.self_attn(self.norm1(tgt), periods)
        tgt = tgt + self.cross_attn(self.norm2(tgt), memory, memory, need_weights=False)[0]
        tgt = tgt + self.ffn(self.norm3(tgt))
        return tgt


# ============================================================
# Positional Encoding (batch_first): sinusoidal + learnable
# ============================================================
class PositionalEncodingBF(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.learnable = nn.Parameter(torch.zeros(1, max_len, d_model))
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("sinusoidal", pe.unsqueeze(0))  # (1,max_len,d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.sinusoidal[:, :T, :] + self.learnable[:, :T, :]


# -----------------------------------------------
# GAT + Transformer OD Prediction Model
# -----------------------------------------------
class GATTransformerODWeek(nn.Module):
    def __init__(
        self,
        num_nodes,
        node_feat_dim,
        gat_hid_dim=32,
        heads=4,
        decode_num_layers=2,
        num_future_steps=6,
        weekday_emb_dim=8,
        time_enc_dim=2,
        node_latlon=None,          # ★ optional
        # period-aware params
        num_self_heads: int = 8,        # period-sparse self-attn heads
        num_cross_heads: int = 8,       # cross-attn heads (to memory)
        mov_avg_win: int = 25,
        dropout: float = 0.1,
        max_len: int = 4096,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.future_steps = num_future_steps

        self.use_geo = node_latlon is not None
        self.geo_dim = 2 if self.use_geo else 0

        self.node_embed = nn.Embedding(num_nodes, node_feat_dim)
        self.weekday_embed = nn.Embedding(7, weekday_emb_dim)

        if self.use_geo:
            self.register_buffer("node_latlon", node_latlon)

        in_dim = node_feat_dim + weekday_emb_dim + self.geo_dim

        self.short_spatial_encoder = StaticGATEncoder(
            in_dim, gat_hid_dim, heads=heads, layers=1
        )
        self.long_spatial_encoder = StaticGATEncoder(
            in_dim, gat_hid_dim, heads=heads, layers=3
        )
        self.spatial_fuse_proj = nn.Linear(2 * gat_hid_dim, gat_hid_dim)

        self.dynamic_encoder = DynamicGATEncoder(
            gat_hid_dim, gat_hid_dim, heads=heads, edge_attr_dim=1
        )

        self.time_enc_linear = nn.Linear(time_enc_dim, gat_hid_dim)
        self.pos_enc = PositionalEncoding(gat_hid_dim)
        self.decoder = TimeTransformerDecoder(
            hid_dim=gat_hid_dim,
            num_heads=8,
            ff_dim=gat_hid_dim * 4,
            num_layers=decode_num_layers
        )
        
        # # ✅ positional (batch_first)
        # self.pos_mem = PositionalEncodingBF(gat_hid_dim, max_len=max_len)
        # self.pos_tgt = PositionalEncodingBF(gat_hid_dim, max_len=max_len)

        # # ✅ period extractor: top_k must equal num_self_heads
        # self.period_extractor = NodePeriodicityExtractor(top_k=num_self_heads, mov_avg_win=mov_avg_win)
        self.fuse_proj = nn.Linear(3 * gat_hid_dim, gat_hid_dim)

        # # ✅ period-aware decoder layers (self=period-sparse, cross=multihead)
        # self.decoder_layers = nn.ModuleList([
        #     PeriodAwareDecoderLayer(
        #         d_model=gat_hid_dim,
        #         num_self_heads=num_self_heads,
        #         num_cross_heads=num_cross_heads,
        #         dropout=dropout,
        #     )
        #     for _ in range(decode_num_layers)
        # ])

        # ✅ learnable start token for tgt (better than all-zeros)
        self.start_token = nn.Parameter(torch.zeros(1, 1, gat_hid_dim))

        self.origin_proj = nn.Linear(gat_hid_dim, gat_hid_dim)
        self.dest_proj = nn.Linear(gat_hid_dim, gat_hid_dim)
        self.future_step_emb = nn.Embedding(self.future_steps, gat_hid_dim)

        # init
        nn.init.xavier_uniform_(self.fuse_proj.weight)
        nn.init.xavier_uniform_(self.spatial_fuse_proj.weight)
        nn.init.xavier_uniform_(self.origin_proj.weight)
        nn.init.xavier_uniform_(self.dest_proj.weight)
    
    def _repeat_edge_index(self, edge_index: torch.Tensor, B: int, N: int):
        # edge_index: (2, E), node id range [0, N-1]
        device = edge_index.device
        offsets = (torch.arange(B, device=device) * N).view(B, 1, 1)  # (B,1,1)
        ei = edge_index.view(1, 2, -1).repeat(B, 1, 1)               # (B,2,E)
        ei = ei + offsets                                            # (B,2,E)
        return ei.permute(1, 0, 2).reshape(2, -1)                    # (2, B*E)

    def forward(self, adjency_edge_index, batch_graph, B, T, time_enc_hist, time_enc_fut, weekday_tensor):
        N = self.num_nodes

        base = self.node_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        weekday = self.weekday_embed(weekday_tensor).unsqueeze(1).repeat(1, N, 1)

        feats = [base]
        if self.use_geo:
            feats.append(self.node_latlon.unsqueeze(0).repeat(B, 1, 1))
        feats.append(weekday)

        static_feat = torch.cat(feats, dim=-1)
        flat = static_feat.view(B * N, -1)

        edge_index_batched = self._repeat_edge_index(adjency_edge_index, B, N)
        short = self.short_spatial_encoder(flat, edge_index_batched)  # ✅
        long  = self.long_spatial_encoder(flat, edge_index_batched)   # ✅

        spatial = torch.cat([short, long], dim=-1)   # (B*N, 2d)
        spatial = self.spatial_fuse_proj(spatial)    # (B*N, d)
        spatial = spatial.view(B, N, -1)

        dyn_in = spatial.unsqueeze(1).repeat(1, T, 1, 1).view(B * T * N, -1)
        batch_graph.x = dyn_in

        dyn = self.dynamic_encoder(
            batch_graph.x,
            batch_graph.edge_index,
            batch_graph.edge_attr
        )
        dyn = dyn.view(B, T, N, -1)

        time_enc = self.time_enc_linear(time_enc_hist).unsqueeze(2).repeat(1, 1, N, 1)
        spatial_exp = spatial.unsqueeze(1).expand(-1, T, -1, -1)

        fused = torch.cat(
            [
                dyn,            # (B,T,N,d)
                spatial_exp,    # (B,T,N,d)
                time_enc        # (B,T,N,d)
            ],
            dim=-1             # (B,T,N,3d)
        )
        fused = self.fuse_proj(fused)   # (B,T,N,d)

        # memory = fused.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        # memory = self.pos_mem(memory)

        # --Before (non-period-aware) --
        memory = fused.permute(1, 0, 2, 3).reshape(T, B * N, -1)
        memory = self.pos_enc(memory)
        
        O = self.future_steps
        # # ✅ periods from memory (per node-sequence)
        # periods_global = self.period_extractor(memory)
        # periods = periods_global.repeat_interleave(N, dim=0)

        # # ---------- DECODE (Period-aware) ----------
        # tgt = self.start_token.expand(B * N, O, -1)  # (B*N,O,d)
        # tgt = self.pos_tgt(tgt)

        # for layer in self.decoder_layers:
        #     tgt = layer(tgt, memory, periods)  # (B*N,O,d)
        # dec_out = dec_out.view(B, N, O, -1).permute(2, 0, 1, 3).contiguous()  # (O,B,N,d)

        step_ids = torch.arange(O, device=memory.device)              # (O,)
        tgt = self.future_step_emb(step_ids).unsqueeze(1).expand(O, B * N, -1)  # (O,B*N,d)
        tgt = self.pos_enc(tgt)

        # ✅ (권장) 미래 시간 인코딩 추가
        # time_enc_fut: (B,O,2)
        if time_enc_fut is not None:
            tgt_time = self.time_enc_linear(time_enc_fut)          # (B,O,d)
            tgt_time = tgt_time.unsqueeze(2).expand(-1, -1, N, -1) # (B,O,N,d)
            tgt_time = tgt_time.permute(1, 0, 2, 3).reshape(O, B*N, -1)  # ✅ (O,B*N,d)
            tgt = tgt + tgt_time

        out = self.decoder(tgt, memory, tgt_mask=None)                # (O,B*N,d)
        out = out.view(O, B, N, -1)                                   # (O,B,N,d)

        H_O = self.origin_proj(out)                                   # (O,B,N,d)
        H_D = self.dest_proj(out)                                     # (O,B,N,d)

        od_hat = torch.einsum('obid, objd -> obij', H_O, H_D)         # (O,B,N,N)
        od_hat = F.softplus(od_hat)

        return od_hat.permute(1, 0, 2, 3).contiguous()                # (B,O,N,N)

# preds = model(
#     static_edge_index,
#     batch_graph,
#     B, T,
#     time_enc_hist,      # from collate batch
#     weekday_tensor      # from collate batch
# )

if __name__ == "__main__":

    # -----------------------------
    # 하이퍼파라미터 (임시)
    # -----------------------------
    B = 2                   # 배치 크기
    N = 100                 # 노드 수 (역 개수)
    T = 30                  # 입력 과거 시계열 길이
    K = 10                  # 예측 미래 시계열 길이
    node_feat_dim = 4       # 노드 feature 차원
    hid_dim = 8            # 히든 차원
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    adj = pd.read_csv('AD_matrix_trimmed_common.csv', header=None)
    adj = adj.apply(pd.to_numeric, errors="coerce")
    adj = torch.tensor(adj.values, dtype=torch.float32)
    print(adj.shape)

    # edge_index (PyG 형식) [2, num_edges]
    # adj = torch.randint(0, 2, (N, N)).float()
    # adjency_edge_index, _ = dense_to_sparse(adj)
    adjency_edge_index, _ = dense_to_sparse(adj)
    print("edge_index shape:", adjency_edge_index.shape)
    print("edge_index min:", adjency_edge_index.min().item())
    print("edge_index max:", adjency_edge_index.max().item())
    exit()
    
    # -----------------------------
    # 더미 입력 생성
    # -----------------------------
    X_past = torch.randn(B, T, N, N)

    data_list = []

    for b in range(B):
        for t in range(T):
            dummy_adj = X_past[b, t]
            edge_idx, edge_attr = dense_to_sparse(dummy_adj)
            edge_attr = edge_attr.unsqueeze(-1)

            x = torch.zeros(N, node_feat_dim)  # dummy node feat

            data_list.append(Data(x=x, edge_index=edge_idx, edge_attr=edge_attr))
    # make batch
    batch_graph = Batch.from_data_list(data_list)

    print("batch.x      ", batch_graph.x.shape)
    print("batch.edge_index", batch_graph.edge_index.shape)
    print("batch.edge_attr ", batch_graph.edge_attr.shape)
    print("batch.batch  ", batch_graph.batch.shape)

    # -----------------------------
    # 모델 생성
    # -----------------------------
    model = GATTransformerOD(
        num_nodes=N,
        node_feat_dim=node_feat_dim,
        gat_hid_dim=hid_dim,
        num_future_steps=K,
    )

    # -----------------------------
    # Forward pass
    # -----------------------------
    with torch.no_grad():
        preds = model(adjency_edge_index, batch_graph, B, T)

    # -----------------------------
    # 결과 확인
    # -----------------------------
    print("Output shape")
    print(" preds      :", preds.shape)
