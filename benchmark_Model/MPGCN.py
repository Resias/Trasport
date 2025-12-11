import torch
import torch.nn as nn
import torch.nn.functional as F
# ===========================================================
# 1. Graph & Math Utilities (Batched Support Added)
# ===========================================================

def build_normalized_laplacian(adj: torch.Tensor) -> torch.Tensor:
    """
    Computes Normalized Laplacian: L = I - D^{-1/2} A D^{-1/2}
    Supports both (N, N) and (B, N, N) inputs.
    """
    if adj.dim() == 2:
        # Static Case: (N, N)
        N = adj.size(0)
        device = adj.device
        I = torch.eye(N, device=device)
        deg = adj.sum(dim=1)
    else:
        # Dynamic Batch Case: (B, N, N)
        B, N, _ = adj.shape
        device = adj.device
        I = torch.eye(N, device=device).unsqueeze(0).expand(B, N, N)
        deg = adj.sum(dim=2) # (B, N)

    # Degree Matrix Inverse Sqrt
    deg_inv_sqrt = torch.zeros_like(deg)
    mask = deg > 0
    deg_inv_sqrt[mask] = deg[mask].pow(-0.5)
    
    if adj.dim() == 2:
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        L = I - D_inv_sqrt @ adj @ D_inv_sqrt
    else:
        # Batch diagonal matrix creation
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt) # (B, N, N)
        # L = I - D @ A @ D
        L = I - torch.bmm(torch.bmm(D_inv_sqrt, adj), D_inv_sqrt)
        
    return L

def get_chebyshev_polynomials(L: torch.Tensor, K: int):
    """
    Computes Chebyshev polynomials T_k(L).
    Supports both Static (N, N) and Dynamic (B, N, N) Laplacian.
    Returns: List of K tensors.
    """
    # Check dimension to decide between mm and bmm
    is_batch = L.dim() == 3
    
    if not is_batch:
        N = L.size(0)
        I = torch.eye(N, device=L.device)
        matmul_fn = torch.mm
    else:
        B, N, _ = L.shape
        I = torch.eye(N, device=L.device).unsqueeze(0).expand(B, N, N)
        matmul_fn = torch.bmm
    
    T0 = I
    if K == 1: return [T0]
    
    T1 = L
    if K == 2: return [T0, T1]
    
    T_k = [T0, T1]
    for k in range(2, K):
        # T_k = 2 * L * T_{k-1} - T_{k-2}
        Tk = 2 * matmul_fn(L, T_k[-1]) - T_k[-2]
        T_k.append(Tk)
        
    return T_k

# ===========================================================
# 2. 2D-GCN Layer (Sample-wise Dynamic Support)
# ===========================================================

class TwoDGCNLayer(nn.Module):
    """
    Implementation of Eq (3): 2D Graph Convolution
    H' = sum_i sum_j [ H x_1 T_i(L1) x_2 T_j(L2) ] x_3 W_ij
    """
    def __init__(self, in_dim, out_dim, K):
        super().__init__()
        self.K = K
        self.out_dim = out_dim
        
        # Learnable Weights W_{ij}: (K, K, In, Out)
        self.weights = nn.Parameter(torch.FloatTensor(K, K, in_dim, out_dim))
        self.bias = nn.Parameter(torch.FloatTensor(out_dim))
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)
        
    def forward(self, H, cheb_L1, cheb_L2):
        """
        H: (B, N, N, Fin)
        cheb_L1: List of T_i(L_origin). Each element can be (N, N) or (B, N, N).
        cheb_L2: List of T_j(L_dest). Each element can be (N, N) or (B, N, N).
        """
        B, N, _, Fin = H.shape
        out = torch.zeros(B, N, N, self.out_dim, device=H.device)
        
        # Check if graphs are dynamic (Batch-wise)
        is_dynamic_L1 = cheb_L1[0].dim() == 3
        is_dynamic_L2 = cheb_L2[0].dim() == 3
        
        for i in range(self.K):
            T_i = cheb_L1[i] 
            
            # [Operation x_1]: Convolution on Origin
            if not is_dynamic_L1:
                # Static: (N, N) @ (B, N, N, F) -> (B, N, N, F)
                # 'uv, bvwf -> buwf'
                H_x1 = torch.einsum("uv, bvwf -> buwf", T_i, H) 
            else:
                # Dynamic: (B, N, N) @ (B, N, N, F) -> (B, N, N, F)
                # 'buv, bvwf -> buwf'
                H_x1 = torch.einsum("buv, bvwf -> buwf", T_i, H)

            for j in range(self.K):
                T_j = cheb_L2[j]
                
                # [Operation x_2]: Convolution on Destination
                if not is_dynamic_L2:
                    # Static: (B, N, N, F) @ (N, N)
                    # 'buwf, wz -> buzf'
                    H_x1_x2 = torch.einsum("buwf, wz -> buzf", H_x1, T_j)
                else:
                    # Dynamic: (B, N, N, F) @ (B, N, N)
                    # 'buwf, bwz -> buzf'
                    H_x1_x2 = torch.einsum("buwf, bwz -> buzf", H_x1, T_j)
                
                # [Operation x_3]: Feature Transformation
                W_ij = self.weights[i, j]
                term_ij = torch.matmul(H_x1_x2, W_ij)
                
                out += term_ij
                
        return F.relu(out + self.bias)

# ===========================================================
# 3. Single Backbone Model (LSTM + 2D-GCN)
# ===========================================================

class MPGCNBackbone(nn.Module):
    def __init__(self, N, lstm_h, gcn_h, gcn_out, K, num_layers=2):
        super().__init__()
        
        # Temporal Feature Extraction
        # Input dim is 1 (Scalar OD flow)
        # self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_h, batch_first=True)
        self.lstm = nn.LSTM(input_size=N*N, hidden_size=lstm_h, batch_first=True)

        # Spatial Layers
        self.gcn_layers = nn.ModuleList()
        in_dim = lstm_h
        
        for _ in range(num_layers - 1):
            self.gcn_layers.append(TwoDGCNLayer(in_dim, gcn_h, K))
            in_dim = gcn_h
            
        self.gcn_layers.append(TwoDGCNLayer(in_dim, gcn_out, K))
        self.regressor = nn.Linear(gcn_out, 1)
        
    def forward(self, X, cheb_L1, cheb_L2):
        B, T, N, _ = X.shape
        
        # # [Step 1] LSTM
        # # (B, T, N, N) -> (B, N, N, T) -> (B*N*N, T, 1)
        # X_seq = X.permute(0, 2, 3, 1).contiguous().view(B * N * N, T, 1)
        # _, (h_n, _) = self.lstm(X_seq)
        # # h_n[-1]: (B*N*N, lstm_h) -> (B, N, N, lstm_h)
        # H = h_n[-1].view(B, N, N, -1)
        # (B, T, N*N)
        X_seq = X.reshape(B, T, N * N)

        # LSTM 입력 차원 변경
        _, (h_n, _) = self.lstm(X_seq)   # h_n[-1]: (B, lstm_h)

        # 동일한 크기의 OD 히트맵 형태로 확장
        H = h_n[-1].unsqueeze(1).unsqueeze(1).repeat(1, N, N, 1)

        # [Step 2] 2D-GCN
        for gcn in self.gcn_layers:
            H = gcn(H, cheb_L1, cheb_L2)
            
        # [Step 3] Output
        out = self.regressor(H).squeeze(-1)
        return out

# ===========================================================
# 4. Full MPGCN Model (Ensemble)
# ===========================================================

class MPGCN(nn.Module):
    def __init__(self, N, lstm_h=32, gcn_h=32, gcn_out=16, K=3, num_layers=2):
        super().__init__()
        self.K = K
        
        self.model_adj = MPGCNBackbone(N, lstm_h, gcn_h, gcn_out, K, num_layers)
        self.model_poi = MPGCNBackbone(N, lstm_h, gcn_h, gcn_out, K, num_layers)
        self.model_dyn = MPGCNBackbone(N, lstm_h, gcn_h, gcn_out, K, num_layers)
        
        self.cheb_adj_O = None; self.cheb_adj_D = None
        self.cheb_poi_O = None; self.cheb_poi_D = None

    # ------------ STATIC GRAPH SETUP (DEVICE-SAFE) -------------
    def set_static_graphs(self, adj_O, adj_D, poi_O, poi_D):
        device = adj_O.device

        L_adj_O = build_normalized_laplacian(adj_O).to(device)
        L_adj_D = build_normalized_laplacian(adj_D).to(device)
        L_poi_O = build_normalized_laplacian(poi_O).to(device)
        L_poi_D = build_normalized_laplacian(poi_D).to(device)

        self.cheb_adj_O = [t.to(device) for t in get_chebyshev_polynomials(L_adj_O, self.K)]
        self.cheb_adj_D = [t.to(device) for t in get_chebyshev_polynomials(L_adj_D, self.K)]
        self.cheb_poi_O = [t.to(device) for t in get_chebyshev_polynomials(L_poi_O, self.K)]
        self.cheb_poi_D = [t.to(device) for t in get_chebyshev_polynomials(L_poi_D, self.K)]

    # ------------------ FORWARD (DEVICE-SAFE) --------------------
    def forward(self, X, dynamic_adj_O=None, dynamic_adj_D=None):
        device = X.device

        # Static graph tensors → force to device
        self.cheb_adj_O = [t.to(device) for t in self.cheb_adj_O]
        self.cheb_adj_D = [t.to(device) for t in self.cheb_adj_D]
        self.cheb_poi_O = [t.to(device) for t in self.cheb_poi_O]
        self.cheb_poi_D = [t.to(device) for t in self.cheb_poi_D]

        pred_adj = self.model_adj(X, self.cheb_adj_O, self.cheb_adj_D)
        pred_poi = self.model_poi(X, self.cheb_poi_O, self.cheb_poi_D)

        # Dynamic graph 처리
        if dynamic_adj_O is not None:
            L_dyn_O = build_normalized_laplacian(dynamic_adj_O).to(device)
            L_dyn_D = build_normalized_laplacian(dynamic_adj_D).to(device)

            cheb_dyn_O = [t.to(device) for t in get_chebyshev_polynomials(L_dyn_O, self.K)]
            cheb_dyn_D = [t.to(device) for t in get_chebyshev_polynomials(L_dyn_D, self.K)]

            pred_dyn = self.model_dyn(X, cheb_dyn_O, cheb_dyn_D)
        else:
            pred_dyn = torch.zeros_like(pred_adj)

        final_pred = (pred_adj + pred_poi + pred_dyn) / 3.0
        return final_pred

# ===========================================================
# 5. Usage Example (Dimensions Verification)
# ===========================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = 20
    B = 32
    T = 5  # Paper uses 5 historical points [cite: 1351]
    
    # 1. Initialize Model
    model = MPGCN(N=N, K=3).to(device)
    
    # 2. Setup Dummy Static Graphs (Adj, POI)
    adj_static = torch.rand(N, N).to(device)
    poi_static = torch.rand(N, N).to(device)
    model.set_static_graphs(adj_static, adj_static, poi_static, poi_static)
    
    # 3. Prepare Dummy Input
    # Input: (Batch, Time, Origin, Dest)
    X = torch.rand(B, T, N, N).to(device)
    
    # 4. Prepare Dummy Dynamic Graph (Calculated from X in real scenario)
    dyn_adj = torch.rand(N, N).to(device)
    
    # 5. Forward Pass
    out = model(X, dynamic_adj_O=dyn_adj, dynamic_adj_D=dyn_adj)
    
    print(f"Input Shape: {X.shape}")   # (4, 5, 20, 20)
    print(f"Output Shape: {out.shape}") # (4, 20, 20) -> Next step OD flow