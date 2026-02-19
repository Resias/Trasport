import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, layers=2, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hid_dim))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hid_dim, hid_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, self.training)
        return x


class GCN_LSTM_OD(nn.Module):
    """
    Input:
        x_log: (B, T, N, N)  log1p OD
    Output:
        y_hat: (B, T_out, N, N) log1p OD
    """

    def __init__(
        self,
        num_nodes,
        edge_index,
        gcn_in_dim=2,
        hid_dim=64,
        lstm_layers=1,
        pred_steps=6,
    ):
        super().__init__()

        self.N = num_nodes
        self.pred_steps = pred_steps
        self.register_buffer("edge_index", edge_index)

        self.gcn = GCNEncoder(gcn_in_dim, hid_dim, layers=2)
        self.lstm = nn.LSTM(
            hid_dim, hid_dim, lstm_layers, batch_first=True
        )

        self.origin_proj = nn.Linear(hid_dim, hid_dim)
        self.dest_proj = nn.Linear(hid_dim, hid_dim)

    def _node_features(self, x_log_t):
        """
        OD → node inflow / outflow
        """
        x = torch.expm1(x_log_t)
        outflow = x.sum(dim=-1)
        inflow = x.sum(dim=-2)
        return torch.stack([outflow, inflow], dim=-1)

    def forward(self, x_log):
        B, T, N, _ = x_log.shape

        H_seq = []
        for t in range(T):
            node_feat = self._node_features(x_log[:, t])
            node_feat = node_feat.reshape(B * N, -1)

            h = self.gcn(node_feat, self.edge_index)
            h = h.reshape(B, N, -1)
            H_seq.append(h)

        H_seq = torch.stack(H_seq, dim=1)          # (B,T,N,D)
        H_seq = H_seq.permute(0, 2, 1, 3)          # (B,N,T,D)
        H_seq = H_seq.reshape(B * N, T, -1)

        _, (h_last, _) = self.lstm(H_seq)
        h_last = h_last[-1].reshape(B, N, -1)

        preds = []
        for _ in range(self.pred_steps):
            Ho = self.origin_proj(h_last)
            Hd = self.dest_proj(h_last)
            od = Ho @ Hd.transpose(-1, -2)
            preds.append(torch.log1p(F.relu(od)))

        return torch.stack(preds, dim=1)
