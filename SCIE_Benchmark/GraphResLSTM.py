import torch
import torch.nn as nn
import torch.optim as optim
import torch_geometric.nn as pyg_nn

class GraphResLSTM(nn.Module):
    def __init__(self, node_feature_dim, gcn_hidden_dim, lstm_hidden):
        super().__init__()
        # GCN: PyTorch Geometric 사용
        self.gcn1 = pyg_nn.GCNConv(node_feature_dim, gcn_hidden_dim)
        self.res1 = nn.Linear(gcn_hidden_dim, gcn_hidden_dim)
        self.lstm = nn.LSTM(gcn_hidden_dim, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, node_feature_dim)

    def forward(self, x, edge_index):
        # x: features per node at a seq time slice
        # edge_index: graph connectivity
        spatial = torch.relu(self.gcn1(x, edge_index))
        spatial = spatial + self.res1(spatial)  # ResNet skip
        # Assume sequence structure already batch dimension
        out, _ = self.lstm(spatial)            # [B, seq_len, hidden]
        return self.fc(out[:, -1])             # 마지막 시점 예측

# model_graphreslstm = GraphResLSTM(node_feature_dim=32, gcn_hidden_dim=64, lstm_hidden=64)
