# dataset.py

import os
import datetime
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dense_to_sparse


def graph_collate_fn(batch, static_edge_index):
    data_list = []
    B = len(batch)
    T = batch[0]["x_tensor"].shape[0]
    N = batch[0]["x_tensor"].shape[1]

    for b in range(B):
        x_seq = batch[b]["x_tensor"]  # (T, N, N)
        for t in range(T):
            od_t = x_seq[t]  # (N, N)
            edge_idx, edge_attr = dense_to_sparse(od_t)
            edge_attr = edge_attr.unsqueeze(-1)
            x_node = torch.zeros(N, 1)
            data_list.append(
                Data(x=x_node, edge_index=edge_idx, edge_attr=edge_attr)
            )

    batch_graph = Batch.from_data_list(data_list)
    labels = torch.stack([b["y_tensor"] for b in batch])  # (B, K, N, N)

    return static_edge_index, batch_graph, B, T, labels

def graph_week_collate_fn(batch, static_edge_index):
    data_list = []
    B = len(batch)
    T = batch[0]["x_tensor"].shape[0]
    N = batch[0]["x_tensor"].shape[1]

    time_hist_list = []
    weekday_list = []

    for b in range(B):
        x_seq = batch[b]["x_tensor"]
        time_hist_list.append(batch[b]["time_enc_hist"])
        weekday_list.append(batch[b]["weekday_tensor"])

        for t in range(T):
            od_t = x_seq[t]
            edge_idx, edge_attr = dense_to_sparse(od_t)
            edge_attr = edge_attr.unsqueeze(-1)
            x_node = torch.zeros(N, 1)
            data_list.append(
                Data(x=x_node, edge_index=edge_idx, edge_attr=edge_attr)
            )

    batch_graph = Batch.from_data_list(data_list)
    labels = torch.stack([b["y_tensor"] for b in batch])
    time_enc_hist = torch.stack(time_hist_list)  # [B, T, 2]
    weekday = torch.stack(weekday_list)         # [B]

    return static_edge_index, batch_graph, B, T, labels, time_enc_hist, weekday




def build_time_sin_cos(minute_indices, period=1440):
    """
    minute_indices: 1D numpy array or list of int (분단위 인덱스, 0~1439)
    period: 하나의 주기 길이 (1440분 = 24시간)
    
    return: [T, 2] tensor (sin, cos)
    """
    minute_indices = np.asarray(minute_indices, dtype=np.float32)
    # 0~2π 스케일로 변환
    angles = 2.0 * np.pi * minute_indices / float(period)
    sin_vals = np.sin(angles)
    cos_vals = np.cos(angles)
    enc = np.stack([sin_vals, cos_vals], axis=-1)  # [T, 2]
    return torch.tensor(enc, dtype=torch.float32)

def weekday_onehot(weekday):
    # weekday: 0~6  (월~일)
    onehot = np.zeros(7, dtype=np.float32)
    onehot[weekday] = 1.0
    return torch.tensor(onehot, dtype=torch.float32)  # (7,)


# ==========================================
# 2. Spatial Correlation Selector (User Logic Integrated)
# ==========================================

class SpatialCorrelationSelector:
    def __init__(self, data_root, train_subdir, dist_matrix, top_k=3):
        """
        data_root: 데이터 루트 경로
        dist_matrix: (N, N) 거리 행렬 (사용자 로직 r_ij 계산용)
        """
        self.data_root = os.path.join(data_root, train_subdir)
        self.dist_matrix = dist_matrix
        self.top_k = top_k

    def compute_spatial_corr_indices(self, OD_train_tensor, inflow, outflow, s, e):
        """
        사용자 님이 제공한 로직 (p_ij, q_ij, r_ij 계산)
        OD_train_tensor: (Days, H, N, N)
        """
        Days, H, N, _ = OD_train_tensor.shape
        
        # M_se (target OD sequence flattened)
        M_se = OD_train_tensor[:, :, s, e].reshape(-1)
        # Standard deviation check to avoid division by zero or correlation with constant signal
        if M_se.std() < 1e-6:
            print(f"Warning: Target OD {s}->{e} has practically zero variance.")
            return []

        z_scores = []
        
        print(f"Calculating correlations for Target OD: {s}->{e}...")
        # (N*N loop is heavy, use tqdm if N is large)
        for i in range(N):
            for j in range(N):
                if i == s and j == e: # 자기 자신 제외 혹은 포함 여부 결정 (보통 제외)
                    continue
                if i == j: 
                    continue
                
                M_ij = OD_train_tensor[:, :, i, j].reshape(-1)
                if M_ij.std() < 1e-6:
                    continue
                
                # --- p_ij: trend correlation (Pearson)
                p_ij = np.corrcoef(M_se, M_ij)[0, 1]
                if np.isnan(p_ij): p_ij = 0.0
                
                # --- q_ij: ridership contribution
                # Pre-calculate sums to speed up
                total_ie = OD_train_tensor[:, :, i, e].sum().item()
                total_sj = OD_train_tensor[:, :, s, j].sum().item()
                
                f_e = outflow[:, :, e].sum().item() + 1e-6
                f_s = inflow[:, :, s].sum().item() + 1e-6
                
                q_ij = total_ie / f_e + total_sj / f_s
                
                # --- r_ij: location correlation
                f_i = inflow[:, :, i].sum().item() + outflow[:, :, i].sum().item()
                f_j = inflow[:, :, j].sum().item() + outflow[:, :, j].sum().item()
                
                d_ij = self.dist_matrix[i, j] + 1e-6
                i_ij = (f_i * f_j) / (d_ij * d_ij)
                
                # combine to r_ij
                # dist_matrix access needs check bounds if N matches dist_matrix shape
                d_ie = self.dist_matrix[i, e] + 1e-6
                d_sj = self.dist_matrix[s, j] + 1e-6
                
                r_ij = 0.25 * (i_ij + (f_i * f_e) / (d_ie**2) + (f_s * f_j) / (d_sj**2) + i_ij)
                
                z_scores.append(((i, j), abs(p_ij), q_ij, r_ij)) # p_ij usually abs() in some papers
        
        if not z_scores:
            return []

        # normalization
        p_vals = np.array([z[1] for z in z_scores])
        q_vals = np.array([z[2] for z in z_scores])
        r_vals = np.array([z[3] for z in z_scores])
        
        # Avoid zero division in normalization
        def normalize(vals):
            v_min, v_max = vals.min(), vals.max()
            if v_max - v_min < 1e-9: return np.zeros_like(vals)
            return (vals - v_min) / (v_max - v_min)

        p_norm = normalize(p_vals)
        q_norm = normalize(q_vals)
        r_norm = normalize(r_vals)
        
        # combine into z_ij
        z_final = 1.0 * p_norm + 1.0 * q_norm + 1.0 * r_norm
        
        ranked = sorted(
            list(zip([z[0] for z in z_scores], z_final.tolist())),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked[:self.top_k]

    def select_neighbors(self, target_s, target_e):
        # 1. Load Data for Calculation (Using sample or recent data)
        # Note: 전체 데이터를 메모리에 올리는 것은 무거울 수 있으므로 최근 7일치 등 일부만 사용 권장
        files = sorted([os.path.join(self.data_root, f) for f in os.listdir(self.data_root)])
        sample_files = files[-7:] # 최근 7일치만 사용하여 상관관계 계산 (속도 최적화)
        
        od_list = []
        for f in tqdm(sample_files, desc="Loading data for spatial corr"):
            data = np.load(f) # (1440, N, N)
            od_list.append(data)
            
        OD_train_tensor = torch.tensor(np.stack(od_list), dtype=torch.float32) # (Days, H, N, N)
        inflow = OD_train_tensor.sum(dim=3)  # (Days, H, N)
        outflow = OD_train_tensor.sum(dim=2) # (Days, H, N)
        
        # 2. Call User Logic
        ranked_neighbors = self.compute_spatial_corr_indices(
            OD_train_tensor, inflow, outflow, target_s, target_e
        )
        
        neighbors = [x[0] for x in ranked_neighbors]
        print(f"Selected Top-{self.top_k} Neighbors: {neighbors}")
        return neighbors


class MetroDataset(Dataset):
    """
    지하철 OD 데이터셋

    - data_root: 하루 단위 .npy 파일들이 있는 디렉토리
        각 파일: [1440, N, N] (분단위, 1분 1스텝, 00:00 ~ 24:00)
    - window_size: 입력 시퀀스 길이 (분 단위)
    - hop_size: 슬라이딩 윈도우 이동 간격 (분 단위)
    - pred_size: 예측 시퀀스 길이 (분 단위)
    
    운영시간만 사용: 05:30(=330분) ~ 24:00(=1440분)
    """

    def __init__(self, data_root, window_size, hop_size, pred_size):
        super().__init__()

        self.data_root = data_root
        self.window_size = window_size
        self.hop_size = hop_size
        self.pred_size = pred_size

        # 하루 중 사용할 구간 (분)
        self.day_start_minute = 5 * 60 + 30  # 05:30 -> 330
        self.day_end_minute = 24 * 60        # 24:00 -> 1440

        # 파일들을 날짜 순으로 정렬
        file_names = sorted(os.listdir(data_root))
        self.data_paths = [os.path.join(data_root, f) for f in file_names]

        # ---- 2) 파일 전체 로드 (메모리 캐싱) ----
        print("Caching OD matrices into memory...")
        self.day_data_cache = []
        for path in tqdm(self.data_paths):
            arr = np.load(path, mmap_mode='r')  # shape: [1440, N, N]
            # self.day_data_cache.append(torch.tensor(arr, dtype=torch.float32))
            self.day_data_cache.append(arr)
        print("Caching completed.")
        
        # ---- 3) sliding window 정보 생성 ----
        self.info_list = []
        for file_idx, file_name in enumerate(file_names):
            # 파일명에서 날짜 추출 (예: something_YYYYMMDD.npy)
            ymd = file_name.split('_')[-1].split('.')[0]
            date = datetime.date(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]))
            weekday = date.weekday()  # 0=월, 6=일

            for start_idx in range(self.day_start_minute, self.day_end_minute-(self.pred_size+self.window_size), hop_size):
                self.info_list.append({
                    "file_idx": file_idx,
                    "start_idx": start_idx,
                    "weekday": weekday,
                })

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, index):
        info = self.info_list[index]
        file_idx = info["file_idx"]
        start_idx = info["start_idx"]
        weekday = info["weekday"]

        # -------- Memory Cache에서 바로 가져오기 ----------
        day_data = self.day_data_cache[file_idx]  # (1440, N, N)
        
        # slicing
        x_numpy = day_data[start_idx:start_idx+self.window_size].copy()
        y_numpy = day_data[start_idx+self.window_size:
                            start_idx+self.window_size+self.pred_size].copy()
        x_tensor = torch.from_numpy(x_numpy).float()
        y_tensor = torch.from_numpy(y_numpy).float()

        # time encoding
        hist_minutes = torch.arange(start_idx, start_idx+self.window_size) % 1440
        fut_minutes = torch.arange(start_idx+self.window_size,
                                   start_idx+self.window_size+self.pred_size) % 1440

        time_enc_hist = build_time_sin_cos(hist_minutes.numpy())
        time_enc_fut = build_time_sin_cos(fut_minutes.numpy())

        return {
            "x_tensor": x_tensor,      # (T_in, N, N)
            "y_tensor": y_tensor,
            "weekday_tensor": torch.tensor(weekday),
            "time_enc_hist": time_enc_hist,
            "time_enc_fut": time_enc_fut
        }


class ODPairDatasetV2(Dataset):
    """
    MetroDataset 구조를 그대로 참고한 OD Pair Dataset
    - 특정 OD pair (i, j)에 대한 시계열만 추출
    - sliding window 적용
    - 최종 입력: (T_in, F)
    - 최종 출력: (T_out, 1)
    """

    def __init__(self, data_root, window_size, hop_size, pred_size,
                 od_i, od_j,
                 use_weekday=True,
                 use_time_encoding=True):
        super().__init__()

        self.data_root = data_root
        self.window_size = window_size
        self.hop_size = hop_size
        self.pred_size = pred_size
        self.i = od_i
        self.j = od_j
        self.use_weekday = use_weekday
        self.use_time_encoding = use_time_encoding

        self.day_start_minute = 5 * 60 + 30  # 330
        self.day_end_minute = 24 * 60        # 1440

        self.info_list = []
        self.data_list = []

        file_names = sorted(os.listdir(data_root))
        self.data_paths = [os.path.join(data_root, f) for f in file_names]

        # ---- 1) 하루치 파일 전체 메모리 캐싱 ----
        print("Caching OD matrices for ODPairDataset...")
        self.day_data_cache = []
        self.weekday_list = []
        for file_name in file_names:
            ymd = file_name.split('_')[-1].split('.')[0]
            date = datetime.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))
            self.weekday_list.append(date.weekday())

        for path in tqdm(self.data_paths):
            arr = np.load(path)
            self.day_data_cache.append(torch.tensor(arr, dtype=torch.float32))  
        print("Caching completed.")

        # ---- 2) sliding window 정의 ----
        self.info_list = []
        for file_idx, wd in enumerate(self.weekday_list):
            for start_idx in range(
                self.day_start_minute,
                self.day_end_minute - (self.window_size + self.pred_size),
                self.hop_size
            ):
                self.info_list.append({
                    "file_idx": file_idx,
                    "weekday": wd,
                    "start_idx": start_idx,
                })

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, index):
        info = self.info_list[index]
        file_idx = info["file_idx"]
        weekday = info["weekday"]
        start_idx = info["start_idx"]

        # -------- 캐싱된 메모리에서 가져오기 --------
        day_data = self.day_data_cache[file_idx]  # (1440, N, N)

        # 특정 OD pair 시계열 추출
        od_seq = day_data[:, self.i, self.j]  # (1440,)

        # window slicing
        x_vals = od_seq[start_idx:start_idx+self.window_size]
        y_vals = od_seq[start_idx+self.window_size:
                        start_idx+self.window_size+self.pred_size]

        # Build features
        # (T_in, 1)
        flow_feature = x_vals.unsqueeze(-1)

        feat_list = [flow_feature]

        # 시간 인코딩
        if self.use_time_encoding:
            hist_minutes = np.arange(start_idx, start_idx+self.window_size) % 1440
            time_enc_hist = build_time_sin_cos(hist_minutes)
            feat_list.append(time_enc_hist)

        # 요일 원핫
        if self.use_weekday:
            weekday_oh = weekday_onehot(weekday).unsqueeze(0).repeat(self.window_size, 1)
            feat_list.append(weekday_oh)

        x_feat = torch.cat(feat_list, dim=1)
        y_feat = y_vals.unsqueeze(-1)

        return {
            "x": x_feat.float(),     # (T, F)
            "y": y_feat.float(),     # (1, 1)
        }



# ==========================================
# 3. Dataset (Lag Logic Integrated)
# ==========================================
class STLSTMDataset(Dataset):
    """
    논문 ST-LSTM의 Temporal Lag 제약(W_matrix)을 논문 의도대로 구현한 Dataset.
    
    수정 사항:
    1. Lag Masking(0으로 지움) -> Lag Shifting(과거 시점으로 윈도우 이동)
    2. Inflow/Outflow는 Real-time 데이터이므로 Lag 미적용
    3. Local Normalization 제거 (절대적 크기 정보 보존)
    """

    def __init__(
        self,
        data_root,
        window_size,
        hop_size,
        pred_size,
        target_s,
        target_e,
        neighbors,
        W_matrix,
        use_weekday=True,
        use_time=True,
        normalize=False # Local normalization은 꺼두는 것이 좋음
    ):
        super().__init__()

        self.data_root = data_root
        self.window = window_size
        self.hop = hop_size
        self.pred = pred_size

        self.s = target_s
        self.e = target_e
        self.neighbors = neighbors
        self.W = W_matrix.astype(int)

        self.use_weekday = use_weekday
        self.use_time = use_time
        self.normalize = normalize

        # 운영 시간(05:30 ~ 24:00)
        self.day_start = 330
        self.day_end = 1440

        # ======================
        # Load all .npy into memory
        # ======================
        print("[ST-LSTM] Loading OD matrices...")
        file_names = sorted(os.listdir(data_root))
        self.paths = [os.path.join(data_root, f) for f in file_names]

        self.OD = []
        self.WD = []

        for fname, path in zip(file_names, self.paths):
            # 요일 계산
            ymd = fname.split("_")[-1].split(".")[0]
            dt = datetime.date(
                int(ymd[:4]), int(ymd[4:6]), int(ymd[6:])
            )
            self.WD.append(dt.weekday())

            arr = np.load(path)  # (1440, N, N)
            self.OD.append(torch.tensor(arr, dtype=torch.float32))

        N = self.OD[0].shape[1]
        print(f"[ST-LSTM] Loaded {len(self.OD)} days, stations = {N}")

        # Precompute inflow/outflow
        self.inflow = [day.sum(dim=2) for day in self.OD]
        self.outflow = [day.sum(dim=1) for day in self.OD]

        # ======================
        # Build sliding windows
        # ======================
        self.index = []
        
        # Lag 때문에 윈도우가 과거로 밀릴 수 있음을 고려하여 시작 시점 조정
        # 가장 큰 Lag 값만큼 안전 마진 확보 (여기서는 임의로 60분 추가)
        safe_start = self.day_start + self.window + 60 
        
        for d, wd in enumerate(self.WD):
            for t in range(safe_start,
                           self.day_end - self.pred,
                           self.hop):
                self.index.append((d, t, wd))

        print(f"[ST-LSTM] Total samples = {len(self.index)}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        d, t, wd = self.index[idx]

        od = self.OD[d]        # (1440, N, N)
        infl = self.inflow[d]  # (1440, N)
        out = self.outflow[d]  # (1440, N)

        features = []

        # ===========================================
        # Feature 1: Target OD (with Lag Shifting)
        # 논문: "select the latest h time slot data" considering W_ij
        # 입력 범위: [t - window - lag : t - lag]
        # ===========================================
        lag_se = int(self.W[self.s, self.e])
        t_end_se = t - lag_se
        t_start_se = t_end_se - self.window
        
        # Boundary Check (혹시 음수가 되면 0으로 패딩)
        if t_start_se < 0:
             tgt_feat = torch.zeros(self.window, 1)
        else:
             tgt_feat = od[t_start_se:t_end_se, self.s, self.e].unsqueeze(-1)
        
        features.append(tgt_feat)

        # ===========================================
        # Feature 2: Neighbors (with Lag Shifting)
        # ===========================================
        for (i, j) in self.neighbors:
            lag_ij = int(self.W[i, j])
            t_end_ij = t - lag_ij
            t_start_ij = t_end_ij - self.window
            
            if t_start_ij < 0:
                feat_ij = torch.zeros(self.window, 1)
            else:
                feat_ij = od[t_start_ij:t_end_ij, i, j].unsqueeze(-1)
            features.append(feat_ij)

        # ===========================================
        # Feature 3: Inflow/Outflow (Real-time, No Lag)
        # 논문: "Real-time inflow/outflow are available."
        # 입력 범위: [t - window : t]
        # ===========================================
        # Inflow at Origin Station (s)
        inflow_feat = infl[t - self.window : t, self.s].unsqueeze(-1)
        # Outflow at Destination Station (e)
        outflow_feat = out[t - self.window : t, self.e].unsqueeze(-1)

        features.append(inflow_feat)
        features.append(outflow_feat)

        # ===========================================
        # Feature 4: Time encoding (Current Context)
        # 입력 범위: [t - window : t]
        # ===========================================
        if self.use_time:
            mins = np.arange(t - self.window, t) % 1440
            time_enc = build_time_sin_cos(mins)  # (Window, 2)
            features.append(time_enc)

        # ===========================================
        # Feature 5: Weekday one-hot
        # ===========================================
        if self.use_weekday:
            wd_feat = weekday_onehot(wd).unsqueeze(0).repeat(self.window, 1)
            features.append(wd_feat)

        # ===========================================
        # Combine features
        # ===========================================
        x = torch.cat(features, dim=-1)  # (Window, F)

        # ===========================================
        # Prediction target (Future, No Lag)
        # ===========================================
        y = od[t : t + self.pred, self.s, self.e].unsqueeze(-1)  # (Pred, 1)

        # Local Normalization 삭제됨 (필요시 Global Scaler 사용 권장)
        
        return {"x": x.float(), "y": y.float()}

# ==========================================
# 4. MPGCN Dataset
# ==========================================
class MPGCNDataset(Dataset):
    """
    MPGCN용 데이터셋
    Input: (Window, N, N) - 전체 네트워크의 OD 흐름
    Output: (Pred, N, N) - 다음 시점의 전체 네트워크 OD 흐름
    
    [수정 사항]
    - Log1p Normalization 제거 (Raw Flow 유지)
    """
    def __init__(self, data_root, window_size, hop_size, pred_size):
        super().__init__()
        self.data_root = data_root
        self.window = window_size
        self.hop = hop_size
        self.pred = pred_size
        
        # 운영 시간 (05:30 ~ 24:00 -> 330 ~ 1440)
        self.day_start = 330
        self.day_end = 1440
        
        # 1. 데이터 로드 (Lazy Loading)
        print("[MPGCN] Loading OD matrices...")
        file_names = sorted(os.listdir(data_root))
        self.paths = [os.path.join(data_root, f) for f in file_names]
        
        self.OD = []
        for path in self.paths:
            # 전체 네트워크 OD (1440, N, N)
            arr = np.load(path, mmap_mode='r') 
            self.OD.append(arr)
            
        N = self.OD[0].shape[1]
        print(f"[MPGCN] Loaded {len(self.OD)} days, N={N}")
        
        # 2. 인덱싱
        self.index = []
        safe_start = self.day_start + self.window 
        
        for d in range(len(self.OD)):
            for t in range(safe_start, self.day_end - self.pred, self.hop):
                self.index.append((d, t))
                
        print(f"[MPGCN] Total samples = {len(self.index)}")
        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        d, t = self.index[idx]
        
        # OD Matrix Load
        # (Window, N, N)
        # mmap 객체를 슬라이싱하여 copy()하면 메모리로 로드됨
        x_seq = torch.tensor(self.OD[d][t-self.window:t].copy(), dtype=torch.float32)
        
        # Target
        # (Pred, N, N)
        y_seq = torch.tensor(self.OD[d][t:t+self.pred].copy(), dtype=torch.float32)
        
        # [수정] Log normalization 제거 -> Raw Value 사용
        
        return {"x": x_seq, "y": y_seq}

def get_mpgcn_dataset(data_root, train_subdir, val_subdir, window_size, hop_size, pred_size):
    train_path = os.path.join(data_root, train_subdir)
    val_path = os.path.join(data_root, val_subdir)
    
    trainset = MPGCNDataset(train_path, window_size, hop_size, pred_size)
    valset = MPGCNDataset(val_path, window_size, hop_size, pred_size)
    
    return trainset, valset

class CacheDataset(Dataset):
    def __init__(self, pt_path):
        self.data = torch.load(pt_path, weights_only=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

def get_dataset(data_root, train_subdir, val_subdir, window_size, hop_size, pred_size):
    
    # train_pt = os.path.join(data_root, 'train.pt')
    # val_pt = os.path.join(data_root, 'val.pt')
    
    # if os.path.exists(train_pt) and os.path.exists(val_pt):
    #     print(f'data load from {train_pt} and {val_pt}')
    #     trainset = CacheDataset(train_pt)
    #     valset = CacheDataset(val_pt)
        
    #     return trainset, valset
    
    train_path = os.path.join(data_root, train_subdir)
    val_path = os.path.join(data_root, val_subdir)
    
    trainset = MetroDataset(
        data_root=train_path,
        window_size=window_size,
        hop_size=hop_size,
        pred_size=pred_size
    )
    valset = MetroDataset(
        data_root=val_path,
        window_size=window_size,
        hop_size=hop_size,
        pred_size=pred_size
    )
    
    # print('data caching...')
    # train_list = []
    # for item in tqdm(trainset):
    #     train_list.append(item)
    # train_tensor = torch.tensor(train_list)
    # val_list = []
    # for item in tqdm(valset):
    #     val_list.append(item)
    # val_tensor = torch.tensor(val_list)
    
    # torch.save(train_tensor, train_pt)
    # torch.save(val_tensor, val_pt)
    
    return trainset, valset


def get_odpair_dataset(data_root, train_subdir, val_subdir,
                       window_size, hop_size, pred_size, od_i, od_j):
    trainset = ODPairDatasetV2(
        os.path.join(data_root, train_subdir),
        window_size, hop_size, pred_size,
        od_i, od_j
    )
    valset = ODPairDatasetV2(
        os.path.join(data_root, val_subdir),
        window_size, hop_size, pred_size,
        od_i, od_j
    )
    return trainset, valset

def get_st_lstm_dataset(data_root, train_subdir, val_subdir,
                        window_size, hop_size, pred_size,
                        target_s, target_e, top_k=3,
                        dist_matrix=None, W_matrix=None):
    """
    ST-LSTM 모델 학습을 위한 데이터셋 생성 헬퍼 함수.
    
    1. 거리 행렬(dist)과 이동 시간 행렬(W)을 로드 (없으면 더미 생성).
    2. SpatialCorrelationSelector를 통해 이웃 OD 자동 선정.
    3. Train/Val Dataset 반환.
    """
    
    train_path = os.path.join(data_root, train_subdir)
    val_path = os.path.join(data_root, val_subdir)

    # 1. 행렬(Matrix) 준비
    # N(역 개수) 확인을 위해 샘플 파일 하나 로드
    sample_files = sorted(os.listdir(train_path))
    if not sample_files:
        raise FileNotFoundError(f"No .npy files found in {train_path}")
        
    temp_data = np.load(os.path.join(train_path, sample_files[0]))
    N = temp_data.shape[1]

    # 거리 행렬 (Distance Matrix) - r_ij 계산용
    if dist_matrix is None:
        dist_path = os.path.join(data_root, "dist_matrix.npy")
        if os.path.exists(dist_path):
            print(f"Loading dist_matrix from {dist_path}")
            dist_matrix = np.load(dist_path)
        else:
            print("Warning: 'dist_matrix.npy' not found. Using Random Dummy Matrix.")
            dist_matrix = np.random.rand(N, N) * 10  # Dummy

    # 이동 시간 제한 행렬 (W Matrix) - Lag 계산용
    if W_matrix is None:
        w_path = os.path.join(data_root, "W_matrix.npy")
        if os.path.exists(w_path):
            print(f"Loading W_matrix from {w_path}")
            W_matrix = np.load(w_path)
        else:
            print("Warning: 'W_matrix.npy' not found. Using Random Dummy Matrix (5~30 mins).")
            W_matrix = np.random.randint(5, 30, size=(N, N))

    # 2. 이웃 선정 (Neighbor Selection)
    print(f"Selecting Top-{top_k} Neighbors for OD ({target_s}->{target_e})...")
    selector = SpatialCorrelationSelector(data_root, train_subdir, dist_matrix, top_k=top_k)
    neighbors = selector.select_neighbors(target_s, target_e)
    
    # 3. 데이터셋 생성
    trainset = STLSTMDataset(
        data_root=train_path,
        window_size=window_size,
        hop_size=hop_size,
        pred_size=pred_size,
        target_s=target_s,
        target_e=target_e,
        neighbors=neighbors,
        W_matrix=W_matrix
    )
    
    valset = STLSTMDataset(
        data_root=val_path,
        window_size=window_size,
        hop_size=hop_size,
        pred_size=pred_size,
        target_s=target_s,
        target_e=target_e,
        neighbors=neighbors,
        W_matrix=W_matrix
    )
    
    return trainset, valset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/workspace/od_minute")
    parser.add_argument("--train_dir", type=str, default="train")
    parser.add_argument("--val_dir", type=str, default="test")
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--hop_size", type=int, default=10)
    parser.add_argument("--pred_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    
    # ST-LSTM 테스트용 타겟 설정
    parser.add_argument("--target_s", type=int, default=10, help="Origin Station Index")
    parser.add_argument("--target_e", type=int, default=25, help="Destination Station Index")
    parser.add_argument("--top_k", type=int, default=3, help="Number of neighbors")
    args = parser.parse_args()

    print("=== Dataset Test Start ===")
    print(f"data_root      : {args.data_root}")
    print(f"train_subdir   : {args.train_dir}")
    print(f"val_subdir     : {args.val_dir}")
    print(f"window_size    : {args.window_size}")
    print(f"hop_size       : {args.hop_size}")
    print(f"pred_size      : {args.pred_size}")

    # # ---- Load datasets ----
    # trainset, valset = get_dataset(
    #     data_root=args.data_root,
    #     train_subdir=args.train_dir,
    #     val_subdir=args.val_dir,
    #     window_size=args.window_size,
    #     hop_size=args.hop_size,
    #     pred_size=args.pred_size,
    # )

    # print(f"\nTrain samples : {len(trainset)}")
    # print(f"Val samples   : {len(valset)}")

    # # ---- Loaders ----
    # train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    # # ---- Fetch one batch ----
    # batch = next(iter(train_loader))

    # print("\n=== Sample Batch ===")
    # for k, v in batch.items():
    #     if torch.is_tensor(v):
    #         print(f"{k:15s}: {tuple(v.shape)}")
    #     else:
    #         print(f"{k:15s}: {v}")

    # print("\n=== Time Encoding Check ===")
    # print("time_enc_hist[0]:", batch["time_enc_hist"][0][:5])
    # print("\nDataset test complete.")
    
    
    print("=== ST-LSTM Dataset Test Start ===")
    
    # get_st_lstm_dataset 호출
    # (dist_matrix, W_matrix는 없으면 내부에서 더미 생성)
    trainset, valset = get_st_lstm_dataset(
        data_root=args.data_root,
        train_subdir=args.train_dir,
        val_subdir=args.val_dir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
        target_s=args.target_s,
        target_e=args.target_e,
        top_k=args.top_k
    )

    print(f"\nTrain samples : {len(trainset)}")
    print(f"Val samples   : {len(valset)}")

    # Loader 생성
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    
    # 배치 확인
    batch = next(iter(train_loader))
    print("\n=== Sample Batch (ST-LSTM) ===")
    print(f"Input x shape: {batch['x'].shape}  (Batch, Window, Features)")
    print(f"Target y shape: {batch['y'].shape} (Batch, Pred_Size, 1)")
    
    # Feature Dimension 계산 검증
    # Expected: 1(Self) + K(Neighbors) + 1(S_In) + 1(E_Out) + 2(Time) + 7(Weekday)
    expected_dim = 1 + args.top_k + 1 + 1 + 2 + 7
    print(f"Feature Dim Check: {batch['x'].shape[-1]} (Expected: {expected_dim})")
    
    print("\nDataset test complete.")