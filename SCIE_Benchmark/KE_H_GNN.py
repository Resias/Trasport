import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class KE_H_GNN(nn.Module):
    def __init__(self, gnn_input_dim, external_feat_dim, gnn_hidden, mlr_hidden, od_dim):
        super().__init__()
        self.gnn1 = pyg_nn.GATConv(gnn_input_dim, gnn_hidden)
        self.relu = nn.ReLU()

        # hybrid feature 는 GNN 출력 + external features 결합
        self.mlr = nn.Sequential(
            nn.Linear(gnn_hidden + external_feat_dim, mlr_hidden),
            nn.ReLU(),
            nn.Linear(mlr_hidden, gnn_hidden)
        )

        # 최종 OD Vector 예측
        self.fc_out = nn.Linear(gnn_hidden, od_dim)

    def forward(self, x, edge_index, external):
        """
        x : [batch_size, num_nodes, node_features]
        edge_index : PyG graph 연결정보
        external : [batch_size, num_nodes, external_feat_dim]
        """

        # Spatial GNN
        # → x: [N, feature] 형태로 맞출 필요가 있음
        # (배치/노드 구조에 따라 조정 필요)
        h = self.relu(self.gnn1(x, edge_index))  # → [N, gnn_hidden]

        # external 정보 concat
        hybrid = torch.cat([h, external], dim=-1)

        # MLR block
        hybrid_feat = self.mlr(hybrid)

        # 최종 OD vector
        out = self.fc_out(hybrid_feat)

        return out
# model_ke_h_gnn = KE_H_GNN(gnn_input_dim=16, external_feat_dim=8, gnn_hidden=64, mlr_hidden=32, od_dim=400)
