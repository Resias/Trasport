import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv
from torch_geometric.utils import dense_to_sparse

# -------------------------------------
# Spatial Encoder: TransformerConv
# -------------------------------------
class TransformerSpatialEncoder(nn.Module):
    def __init__(self, in_channels, hid_channels, heads=4):
        super(TransformerSpatialEncoder, self).__init__()
        self.conv1 = TransformerConv(in_channels, hid_channels, heads=heads, dropout=0.2)
        self.conv2 = TransformerConv(hid_channels*heads, hid_channels, heads=1, concat=False)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x  # [N, hid_channels]

# -------------------------------------
# Time Seq2Seq Transformer
# -------------------------------------
class Seq2SeqTransformer(nn.Module):
    def __init__(self, hid_dim, n_heads, num_enc_layers, num_dec_layers, ff_dim):
        super(Seq2SeqTransformer, self).__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=0.1
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_enc_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hid_dim, nhead=n_heads,
            dim_feedforward=ff_dim, dropout=0.1
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_dec_layers)

    def forward(self, src, tgt):
        # src: [T, N, hid_dim]
        # tgt: [K, N, hid_dim]
        memory = self.encoder(src)
        out = self.decoder(tgt, memory)
        return out  # [K, N, hid_dim]

# -------------------------------------
# Full OD Prediction Model
# -------------------------------------
class TransformerConvOD(nn.Module):
    def __init__(self,
                 node_feat_dim,
                 time_feat_dim,
                 hid_dim,
                 n_heads=4,
                 enc_layers=2,
                 dec_layers=2,
                 ff_dim=128,
                 future_steps=6):
        super(TransformerConvOD, self).__init__()

        self.spatial_encoder = TransformerSpatialEncoder(node_feat_dim, hid_dim, heads=n_heads)

        # embed time features
        self.time_emb = nn.Linear(time_feat_dim, hid_dim)

        # Seq2Seq Transformer
        self.seq2seq = Seq2SeqTransformer(
            hid_dim=hid_dim,
            n_heads=n_heads,
            num_enc_layers=enc_layers,
            num_dec_layers=dec_layers,
            ff_dim=ff_dim
        )

        self.future_steps = future_steps
        self.hid_dim = hid_dim

    def forward(self, node_feats, edge_index, time_series):
        """
        node_feats: [N, node_feat_dim]
        edge_index: PyG edge index
        time_series: [T, N, time_feat_dim]
        """

        # -- Spatial encoding
        spatial_emb = self.spatial_encoder(node_feats, edge_index)
        # spatial_emb: [N, hid_dim]

        # -- Temporal embedding for src (past)
        # [T, N, time_feat_dim] -> [T, N, hid_dim]
        src = self.time_emb(time_series)

        # -- Prepare decoder target initialization
        # learnable token or zeros
        tgt = torch.zeros(self.future_steps, node_feats.size(0), self.hid_dim, device=node_feats.device)

        # -- Seq2Seq Transformer decoding
        dec_out = self.seq2seq(src, tgt)
        # dec_out: [K, N, hid_dim]

        # -- OD matrix generation
        # For each future time step, compute outer product
        OD_preds = []
        for k in range(self.future_steps):
            H = dec_out[k]  # [N, hid_dim]

            # full OD: outer product
            OD_t = torch.matmul(H, H.T)  # [N, N]
            OD_preds.append(OD_t)

        OD_preds = torch.stack(OD_preds)  # [K, N, N]
        return OD_preds

# -------------------------------
# Example Run
# -------------------------------
if __name__ == "__main__":
    import pandas as pd

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # parameters
    N = 638
    T = 24
    node_feat_dim = 1
    time_feat_dim = 1
    hid_dim = 32
    K = 6

    # load adjacency
    adj = pd.read_csv('AD_matrix_trimmed_common.csv', header=None)
    adj = adj.apply(pd.to_numeric, errors="coerce")
    adj = torch.tensor(adj.values, dtype=torch.float32)

    # to edge index
    edge_index, _ = dense_to_sparse(adj)

    # dummy input
    node_feats  = torch.randn(N, node_feat_dim).to(device)
    time_series = torch.randn(T, N, time_feat_dim).to(device)

    model = TransformerConvOD(
        node_feat_dim=node_feat_dim,
        time_feat_dim=time_feat_dim,
        hid_dim=hid_dim,
        n_heads=4,
        enc_layers=3,
        dec_layers=3,
        ff_dim=hid_dim*4,
        future_steps=K
    ).to(device)

    with torch.no_grad():
        od_preds = model(node_feats, edge_index.to(device), time_series)

    print("OD_preds shape:", od_preds.shape)
    # expected: [K, N, N]