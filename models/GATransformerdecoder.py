import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse

from torch_geometric.data import Data, Batch


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

    def forward(self, tgt, memory):
        # tgt: [T, N, hid_dim]
        # memory: [mem_len, N, hid_dim] -> GAT output 반복한 형태
        out = self.transformer_decoder(tgt, memory)
        return out          # [T, N, hid_dim]

# -----------------------------------------------
# GAT + Transformer OD Prediction Model
# -----------------------------------------------
class GATTransformerOD(nn.Module):
    def __init__(
        self,
        num_nodes,
        node_feat_dim,
        gat_hid_dim=32,
        heads=4,
        decode_num_layers=2,
        num_future_steps=6,
        node_latlon=None,          # ★ optional
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.future_steps = num_future_steps

        self.use_geo = node_latlon is not None
        self.geo_dim = 2 if self.use_geo else 0

        # base node embedding
        self.node_embed = nn.Embedding(num_nodes, node_feat_dim)

        # optional geo
        if self.use_geo:
            self.register_buffer("node_latlon", node_latlon)  # [N, 2]

        in_dim = node_feat_dim + self.geo_dim

        # static spatial encoders
        self.short_spatial_encoder = StaticGATEncoder(
            in_dim, gat_hid_dim, heads=heads, layers=1
        )
        self.long_spatial_encoder = StaticGATEncoder(
            in_dim, gat_hid_dim, heads=heads, layers=3
        )

        # dynamic encoder
        self.dynamic_encoder = DynamicGATEncoder(
            in_dim, gat_hid_dim, heads=heads, edge_attr_dim=1
        )

        self.pos_enc = PositionalEncoding(gat_hid_dim)

        self.decoder = TimeTransformerDecoder(
            hid_dim=gat_hid_dim,
            num_heads=8,
            ff_dim=gat_hid_dim * 4,
            num_layers=decode_num_layers
        )

    def _build_node_feat(self):
        base = self.node_embed.weight  # [N, D]
        if self.use_geo:
            return torch.cat([base, self.node_latlon], dim=-1)
        return base

    def forward(self, adjency_edge_index, batch_graph, B, T):
        N = self.num_nodes

        # ---------- STATIC ----------
        node_feat = self._build_node_feat()
        short = self.short_spatial_encoder(node_feat, adjency_edge_index)
        long = self.long_spatial_encoder(node_feat, adjency_edge_index)
        spatial_emb = short + long  # [N, D]

        # ---------- DYNAMIC ----------
        node_feat_bt = node_feat.unsqueeze(0).repeat(B * T, 1, 1)
        batch_graph.x = node_feat_bt.view(B * T * N, -1)

        dyn = self.dynamic_encoder(
            batch_graph.x,
            batch_graph.edge_index,
            batch_graph.edge_attr
        )
        dyn = dyn.view(B, T, N, -1)

        spatial_time = dyn + spatial_emb.unsqueeze(0).unsqueeze(0)
        fused = spatial_time.permute(1, 0, 2, 3).reshape(T, B * N, -1)
        fused = self.pos_enc(fused)

        # ---------- DECODE ----------
        tgt = torch.zeros(self.future_steps, B * N, fused.size(-1), device=fused.device)
        dec_out = self.decoder(tgt, fused)

        dec_out = dec_out.view(self.future_steps, B, N, -1)

        preds = []
        for t in range(self.future_steps):
            H = dec_out[t]
            preds.append(H @ H.transpose(-1, -2))

        return torch.stack(preds, dim=1)

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

    def forward(self, adjency_edge_index, batch_graph, B, T, time_enc_hist, weekday_tensor):
        N = self.num_nodes

        base = self.node_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        weekday = self.weekday_embed(weekday_tensor).unsqueeze(1).repeat(1, N, 1)

        feats = [base]
        if self.use_geo:
            feats.append(self.node_latlon.unsqueeze(0).repeat(B, 1, 1))
        feats.append(weekday)

        static_feat = torch.cat(feats, dim=-1)
        flat = static_feat.view(B * N, -1)

        short = self.short_spatial_encoder(flat, adjency_edge_index)
        long = self.long_spatial_encoder(flat, adjency_edge_index)
        spatial = (short + long).view(B, N, -1)

        dyn_in = spatial.unsqueeze(1).repeat(1, T, 1, 1).view(B * T * N, -1)
        batch_graph.x = dyn_in

        dyn = self.dynamic_encoder(
            batch_graph.x,
            batch_graph.edge_index,
            batch_graph.edge_attr
        )
        dyn = dyn.view(B, T, N, -1)

        time_enc = self.time_enc_linear(time_enc_hist).unsqueeze(2).repeat(1, 1, N, 1)
        fused = dyn + spatial.unsqueeze(1) + time_enc

        fused = fused.permute(1, 0, 2, 3).reshape(T, B * N, -1)
        fused = self.pos_enc(fused)

        tgt = torch.zeros(self.future_steps, B * N, fused.size(-1), device=fused.device)
        dec_out = self.decoder(tgt, fused)

        dec_out = dec_out.view(self.future_steps, B, N, -1)

        preds = []
        for t in range(self.future_steps):
            H = dec_out[t]
            preds.append(H @ H.transpose(-1, -2))

        return torch.stack(preds, dim=1)

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
