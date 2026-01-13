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
    def __init__(self,
                 num_nodes,
                 node_feat_dim,
                 gat_hid_dim=32,
                 heads=4,
                 decode_num_layers=2,
                 num_future_steps=6):
        super(GATTransformerOD, self).__init__()
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.spatial_dim = gat_hid_dim
        self.model_dim = gat_hid_dim
        self.future_steps = num_future_steps

        self.node_embed = nn.Embedding(num_nodes, node_feat_dim)
        # 1) 공간 인코더 (GAT)
        self.short_spatial_encoder = StaticGATEncoder(node_feat_dim, gat_hid_dim, heads=heads, layers=1)
        self.long_spatial_encoder = StaticGATEncoder(node_feat_dim, gat_hid_dim, heads=heads, layers=3)

        # 2) 시간적 패턴용 linear & positional embed
        self.dynamic_encoder = DynamicGATEncoder(node_feat_dim, gat_hid_dim, heads=heads, edge_attr_dim=1)
        # Positional encoding
        self.pos_enc = PositionalEncoding(self.model_dim)

        # 3) Transformer Decoder
        self.decoder = TimeTransformerDecoder(
            hid_dim=self.model_dim,
            num_heads=8,
            ff_dim=self.model_dim*4,
            num_layers=decode_num_layers
        )

        # 4) 최종 OD 예측 layer
        self.predictor = nn.Linear(self.model_dim, node_feat_dim)

    def forward(self, adjency_edge_index, batch_graph, B, T):
        # batch size and time length inference
        N = self.num_nodes

        # -------- Spatial Encoding --------
        node_feats = self.node_embed.weight
        short_spatial_emb = self.short_spatial_encoder(node_feats, adjency_edge_index)
        long_spatial_emb = self.long_spatial_encoder(node_feats, adjency_edge_index)
        spatial_emb = short_spatial_emb + long_spatial_emb

        # -------- Temporal Encoding --------
        # Transform time series into hid dim
        node_feats = self.node_embed.weight  # [N, node_feat_dim]
        node_features = node_feats.unsqueeze(0).repeat(B*T, 1, 1)  # [B*T, N, node_feat_dim]
        batch_graph.x = node_features.reshape(B*T*N, self.node_feat_dim)
        
        temporal_emb = self.dynamic_encoder(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr)
        temporal_emb = temporal_emb.view(B, T, N, self.spatial_dim)
        spatial_emb = spatial_emb.unsqueeze(0).unsqueeze(0)   # [1,1,N,D]
        spatial_emb = spatial_emb.expand(B, T, N, -1)
        spatial_time_emb = temporal_emb + F.tanh(spatial_emb)
        dyn_out = spatial_time_emb.view(B, T, N, self.spatial_dim)
        
        # -----------------------------------
        # 4) Sequence fusion
        # -----------------------------------
        # reorganize to Transformer format
        fused = dyn_out.permute(1, 0, 2, 3).reshape(T, B*N, self.spatial_dim)
        fused = self.pos_enc(fused)     # [T, B*N, spatial_dim]

        # -----------------------------------
        # 5) Transformer decoding
        # -----------------------------------
        tgt = torch.zeros(self.future_steps, B*N, self.spatial_dim, device=fused.device)
        dec_out = self.decoder(tgt, fused)
        # → shape: [future_steps, B*N, spatial_dim]

        # -----------------------------------
        # 6) OD prediction projection
        # -----------------------------------
        dec_out = dec_out.view(self.future_steps, B, N, self.spatial_dim)
        preds = []
        for t in range(self.future_steps):
            H = dec_out[t]                  # [B, N, spatial_dim]
            od_t = torch.matmul(H, H.transpose(-1, -2))  # [B, N, N]
            preds.append(od_t)
        preds = torch.stack(preds, dim=1)  # [B, future_steps, N, N]

        return preds


# -----------------------------------------------
# GAT + Transformer OD Prediction Model
# -----------------------------------------------
class GATTransformerODWeek(nn.Module):
    def __init__(self,
                 num_nodes,
                 node_feat_dim,
                 gat_hid_dim=32,
                 heads=4,
                 decode_num_layers=2,
                 num_future_steps=6,
                 weekday_emb_dim=8,
                 time_enc_dim=2):
        super(GATTransformerODWeek, self).__init__()

        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        self.spatial_dim = gat_hid_dim
        self.model_dim = gat_hid_dim
        self.future_steps = num_future_steps

        # ----- NODE FEATURE OPTIONS -----
        # base trainable embedding
        self.node_embed = nn.Embedding(num_nodes, node_feat_dim)
        # weekday embedding (0..6)
        self.weekday_embed = nn.Embedding(7, weekday_emb_dim)
        # linear to project time encoding into model_dim
        self.time_enc_linear = nn.Linear(time_enc_dim, gat_hid_dim)

        # ----- STATIC SPATIAL ENCODER -----
        self.short_spatial_encoder = StaticGATEncoder(node_feat_dim + weekday_emb_dim, gat_hid_dim,
                                                      heads=heads, layers=1)
        self.long_spatial_encoder = StaticGATEncoder(node_feat_dim + weekday_emb_dim, gat_hid_dim,
                                                     heads=heads, layers=3)

        # ----- DYNAMIC SPATIAL ENCODER -----
        # dynamic uses (node_features + weekday)
        self.dynamic_encoder = DynamicGATEncoder(gat_hid_dim, gat_hid_dim,
                                                  heads=heads, edge_attr_dim=1)

        # positional encoding for transformer
        self.pos_enc = PositionalEncoding(self.model_dim)

        # transformer decoder
        self.decoder = TimeTransformerDecoder(
            hid_dim=self.model_dim,
            num_heads=8,
            ff_dim=self.model_dim * 4,
            num_layers=decode_num_layers
        )

        # predictor (optional output projection)
        self.predictor = nn.Linear(self.model_dim, node_feat_dim)

    def forward(self, adjency_edge_index, batch_graph, B, T, time_enc_hist, weekday_tensor):
        N = self.num_nodes

        # ========== STATIC SPATIAL EMBEDDING ==========

        # base node feature
        base_node_feat = self.node_embed.weight  # [N, node_feat_dim]
        # weekday per sample
        weekday_emb = self.weekday_embed(weekday_tensor)  # [B, weekday_emb_dim]

        # expand weekday to (B, N, weekday_emb_dim)
        weekday_rep = weekday_emb.unsqueeze(1).repeat(1, N, 1)  # [B, N, W]
        # flatten static node features by weekday
        static_node_feat = base_node_feat.unsqueeze(0).repeat(B, 1, 1)  # [B, N, node_feat_dim]
        static_node_feat = torch.cat([static_node_feat, weekday_rep], dim=-1)  # [B, N, node_feat_dim+W]

        # merge B dimension for static GAT (shared graph)
        static_node_feat_flat = static_node_feat.view(B * N, -1)
        short_sp_emb = self.short_spatial_encoder(static_node_feat_flat, adjency_edge_index)
        long_sp_emb = self.long_spatial_encoder(static_node_feat_flat, adjency_edge_index)
        spatial_emb = short_sp_emb + long_sp_emb
        spatial_emb = spatial_emb.view(B, N, self.spatial_dim)

        # ========== DYNAMIC SPATIAL ENCODING ==========

        # prepare node features for dynamic
        # dynamic input uses spatial_emb (so dynamic GAT sees learned static space + weekday)
        node_dyn_feat = spatial_emb.unsqueeze(1).repeat(1, T, 1, 1)  # [B, T, N, D]
        node_dyn_feat = node_dyn_feat.view(B * T * N, self.spatial_dim)

        # assign node features to batch_graph
        batch_graph.x = node_dyn_feat

        # dynamic GAT (uses OD flow edge_attr)
        dyn_out = self.dynamic_encoder(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr)
        dyn_out = dyn_out.view(B, T, N, self.spatial_dim)

        # fuse static + dynamic
        spatial_time = dyn_out + spatial_emb.unsqueeze(1)  # [B, T, N, D]

        # ========== TIME ENCODING INTEGRATION ==========

        # time_enc_hist: [B, T, 2]
        time_enc_lin = self.time_enc_linear(time_enc_hist)  # [B, T, model_dim]
        time_enc_lin = time_enc_lin.unsqueeze(2).repeat(1, 1, N, 1)  # [B, T, N, D]

        fused = spatial_time + time_enc_lin  # integrate time positional feature

        # prepare transformer
        fused_seq = fused.permute(1, 0, 2, 3).reshape(T, B * N, self.spatial_dim)
        fused_seq = self.pos_enc(fused_seq)

        # ========== TRANSFORMER DECODING ==========

        tgt = torch.zeros(self.future_steps, B * N, self.spatial_dim, device=fused_seq.device)
        dec_out = self.decoder(tgt, fused_seq)

        # ========== OD PREDICTION ==========

        dec_out = dec_out.view(self.future_steps, B, N, self.spatial_dim)
        preds = []
        for t in range(self.future_steps):
            H = dec_out[t]  # [B, N, D]
            od_t = torch.matmul(H, H.transpose(-1, -2))
            preds.append(od_t)
        preds = torch.stack(preds, dim=1)  # [B, future_steps, N, N]

        return preds
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
