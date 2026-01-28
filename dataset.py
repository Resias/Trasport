# dataset.py
import os
import re
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


def build_laplacian(adj: np.ndarray):
    """
    adj: (N, N)
    return: normalized Laplacian (N, N)
    """
    A = adj.astype(np.float32)
    D = np.diag(A.sum(axis=1))
    D_inv_sqrt = np.linalg.inv(np.sqrt(D + 1e-6))
    L = np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    return torch.tensor(L, dtype=torch.float32)

def compute_clip_value(
    data_root,
    q=0.98,
    sample_ratio=0.001,
    max_samples_per_file=500_000,
    seed=42
):
    """
    Memory-safe approximate quantile estimation.
    논문 faithful: 98% clipping 유지
    """
    rng = np.random.default_rng(seed)
    samples = []

    files = sorted(os.listdir(data_root))

    for f in tqdm(files, desc="Estimating clip value (sampling)"):
        x = np.load(os.path.join(data_root, f), mmap_mode="r")
        flat = x.reshape(-1)

        n = flat.shape[0]
        k = min(int(n * sample_ratio), max_samples_per_file)

        if k <= 0:
            continue

        idx = rng.choice(n, size=k, replace=False)
        samples.append(torch.from_numpy(flat[idx]).float())

    if len(samples) == 0:
        raise RuntimeError("No samples collected for clip value estimation")

    samples = torch.cat(samples)
    clip_val = torch.quantile(samples, q)

    return clip_val

# ==========================================
# Spatial Correlation Selector (User Logic Integrated)
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


# class MetroDataset(Dataset):
#     """
#     지하철 OD 데이터셋

#     - data_root: 하루 단위 .npy 파일들이 있는 디렉토리
#         각 파일: [1440, N, N] (분단위, 1분 1스텝, 00:00 ~ 24:00)
#     - window_size: 입력 시퀀스 길이 (분 단위)
#     - hop_size: 슬라이딩 윈도우 이동 간격 (분 단위)
#     - pred_size: 예측 시퀀스 길이 (분 단위)
    
#     운영시간만 사용: 05:30(=330분) ~ 24:00(=1440분)
#     """

#     def __init__(self, data_root, window_size, hop_size, pred_size, cache_in_mem=True):
#         super().__init__()

#         self.data_root = data_root
#         self.window_size = window_size
#         self.hop_size = hop_size
#         self.pred_size = pred_size
#         self.cache_mem = cache_in_mem

#         # 하루 중 사용할 구간 (분)
#         self.day_start_minute = 5 * 60 + 30  # 05:30 -> 330
#         self.day_end_minute = 24 * 60        # 24:00 -> 1440

#         # 파일들을 날짜 순으로 정렬
#         file_names = sorted(os.listdir(data_root))
#         self.data_paths = [os.path.join(data_root, f) for f in file_names]

#         # ---- 2) 파일 전체 로드 (메모리 캐싱) ----
#         print("Caching OD matrices into memory...")
#         self.day_data_cache = []
#         for path in tqdm(self.data_paths):
#             arr = np.load(path, mmap_mode='r')  # shape: [1440, N, N]
#             if not cache_in_mem:
#                 self.day_data_cache.append(arr)
#             else:
#                 self.day_data_cache.append(torch.tensor(arr, dtype=torch.float32))
            
#         print("Caching completed.")
        
#         # ---- 3) sliding window 정보 생성 ----
#         self.info_list = []
#         for file_idx, file_name in enumerate(file_names):
#             # 파일명에서 날짜 추출 (예: something_YYYYMMDD.npy)
#             ymd = file_name.split('_')[-1].split('.')[0]
#             date = datetime.date(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]))
#             weekday = date.weekday()  # 0=월, 6=일

#             for start_idx in range(self.day_start_minute, self.day_end_minute-(self.pred_size+self.window_size), hop_size):
#                 self.info_list.append({
#                     "file_idx": file_idx,
#                     "start_idx": start_idx,
#                     "weekday": weekday,
#                 })

#     def __len__(self):
#         return len(self.info_list)

#     def __getitem__(self, index):
#         info = self.info_list[index]
#         file_idx = info["file_idx"]
#         start_idx = info["start_idx"]
#         weekday = info["weekday"]

#         # -------- Memory Cache에서 바로 가져오기 ----------
#         day_data = self.day_data_cache[file_idx]  # (1440, N, N)
        
#         # slicing
#         if self.cache_mem:
#             x_tensor = day_data[start_idx:start_idx+self.window_size]
#             y_tensor = day_data[start_idx+self.window_size:
#                                 start_idx+self.window_size+self.pred_size]
#         else:
#             x_numpy = day_data[start_idx:start_idx+self.window_size].copy()
#             y_numpy = day_data[start_idx+self.window_size:
#                                 start_idx+self.window_size+self.pred_size].copy()
#             x_tensor = torch.from_numpy(x_numpy).float()
#             y_tensor = torch.from_numpy(y_numpy).float()

#         # time encoding
#         hist_minutes = torch.arange(start_idx, start_idx+self.window_size) % 1440
#         fut_minutes = torch.arange(start_idx+self.window_size,
#                                    start_idx+self.window_size+self.pred_size) % 1440

#         time_enc_hist = build_time_sin_cos(hist_minutes.numpy())
#         time_enc_fut = build_time_sin_cos(fut_minutes.numpy())

#         return {
#             "x_tensor": x_tensor,      # (T_in, N, N)
#             "y_tensor": y_tensor,
#             "weekday_tensor": torch.tensor(weekday),
#             "time_enc_hist": time_enc_hist,
#             "time_enc_fut": time_enc_fut
#         }

class MetroDataset(Dataset):
    """
    Metro OD Dataset (resolution-aware)

    - data_root: 하루 단위 .npy 파일 디렉토리
        each file: (T_day, N, N)
    - window_size, hop_size, pred_size: step 단위
    - time_resolution: minutes per step
        * MetroFlow: 10
        * MTA: 60
    """

    def __init__(
        self,
        data_root,
        window_size,
        hop_size,
        pred_size,
        time_resolution,
        cache_in_mem=True,
    ):
        super().__init__()

        self.data_root = data_root
        self.window_size = window_size
        self.hop_size = hop_size
        self.pred_size = pred_size
        self.time_resolution = time_resolution
        self.cache_mem = cache_in_mem

        # -------------------------
        # 운영시간 설정
        # -------------------------
        # MetroFlow만 운영시간 컷 적용
        self.use_operating_hours = (time_resolution < 60)

        self.day_start_minute = 5 * 60 + 30  # 05:30
        self.day_end_minute = 24 * 60        # 24:00

        if self.use_operating_hours:
            self.day_start_step = self.day_start_minute // time_resolution
            self.day_end_step = self.day_end_minute // time_resolution
        else:
            self.day_start_step = 0
            self.day_end_step = None  # full day

        # -------------------------
        # 파일 목록 (순수 OD만)
        # -------------------------
        file_names = sorted(
            f for f in os.listdir(data_root)
            if f.endswith(".npy")
            and not f.endswith(".time.npy")
            and not f.endswith(".mask.npy")
        )
        self.file_names = file_names
        self.data_paths = [os.path.join(data_root, f) for f in file_names]

        # -------------------------
        # Load daily OD matrices
        # -------------------------
        self.day_data_cache = []
        self.valid_masks = []

        print("Caching OD matrices...")
        for path in tqdm(self.data_paths):
            arr = np.load(path, mmap_mode='r')  # (T_day, N, N)

            mask_path = path.replace(".npy", ".mask.npy")
            if os.path.exists(mask_path):
                mask = np.load(mask_path)
            else:
                mask = np.ones(arr.shape[0], dtype=bool)

            if cache_in_mem:
                self.day_data_cache.append(
                    torch.tensor(arr, dtype=torch.float32)
                )
            else:
                self.day_data_cache.append(arr)

            self.valid_masks.append(mask)

        print("Caching completed.")

        # -------------------------
        # Build sliding windows
        # -------------------------
        self.info_list = []

        for file_idx, file_name in enumerate(self.file_names):
            stem = file_name.split(".")[0]
            
            m = re.search(r"(\d{8})", stem)
            assert m is not None, f"Cannot parse date from filename: {file_name}"

            ymd = m.group(1)

            date = datetime.date(
                int(ymd[0:4]),
                int(ymd[4:6]),
                int(ymd[6:8])
            )
            weekday = date.weekday()

            T_day = self.day_data_cache[file_idx].shape[0]
            valid_mask = self.valid_masks[file_idx]

            start_s = self.day_start_step
            end_limit = T_day if self.day_end_step is None else self.day_end_step
            end_s = min(
                end_limit,
                T_day - (self.window_size + self.pred_size)
            )

            for start_step in range(start_s, end_s, self.hop_size):
                end_step = start_step + self.window_size + self.pred_size

                if not valid_mask[start_step:end_step].all():
                    continue

                self.info_list.append({
                    "file_idx": file_idx,
                    "start_step": start_step,
                    "weekday": weekday,
                })

        print(f"Total samples: {len(self.info_list)}")

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, index):
        info = self.info_list[index]
        file_idx = info["file_idx"]
        s = info["start_step"]
        weekday = info["weekday"]

        day_data = self.day_data_cache[file_idx]

        x = torch.as_tensor(
            day_data[s : s + self.window_size],
            dtype=torch.float32
        )

        y = torch.as_tensor(
            day_data[
                s + self.window_size :
                s + self.window_size + self.pred_size
            ],
            dtype=torch.float32
        )

        # -------------------------
        # Time encoding (real minutes)
        # -------------------------
        hist_minutes = (
            torch.arange(self.window_size)
            * self.time_resolution
            + s * self.time_resolution
        ) % 1440

        fut_minutes = (
            torch.arange(self.pred_size)
            * self.time_resolution
            + (s + self.window_size) * self.time_resolution
        ) % 1440

        time_enc_hist = build_time_sin_cos(hist_minutes.numpy())
        time_enc_fut = build_time_sin_cos(fut_minutes.numpy())

        return {
            "x_tensor": x,                      # (T_in, N, N)
            "y_tensor": y,                      # (T_out, N, N)
            "weekday_tensor": torch.tensor(weekday, dtype=torch.long),
            "time_enc_hist": time_enc_hist,
            "time_enc_fut": time_enc_fut,
        }


# dataset = MetroDataset(
#     data_root="od_daily_1min",
#     window_size=60,
#     hop_size=5,
#     pred_size=30,
#     time_resolution=1,
# )

# dataset = MetroDataset(
#     data_root="od_daily_10min",
#     window_size=6,    # 60분
#     hop_size=1,       # 10분
#     pred_size=3,      # 30분
#     time_resolution=10,
# )




# ==========================================
# Dataset (Lag Logic Integrated)
# ==========================================
class STLSTMDataset(Dataset):
    """
    ST-LSTM Dataset (Paper-faithful implementation)

    Input:
        X ∈ R^{(aH + h, x + 3)}
    Target:
        y ∈ R^{(pred, 1)}

    where:
        a  = number of historical days
        H  = historical window length
        h  = realtime window length
        x  = number of spatially correlated OD pairs
    """

    def __init__(
        self,
        data_root: str,
        H: int,                    # historical window
        h: int,                    # realtime window
        pred_size: int,
        target_s: int,
        target_e: int,
        day_cluster_path: str,     # .npy (dict)
        top_x_od_path: str,        # .npy (dict)
        W_path: str,               # .npy (dict)
        a: int = 3                 # number of historical days
    ):
        super().__init__()

        # -------------------------
        # Hyperparameters
        # -------------------------
        self.H = H
        self.h = h
        self.pred = pred_size
        self.a = a

        self.s = target_s
        self.e = target_e

        # Operating time (minutes)
        self.day_start = 330   # 05:30
        self.day_end = 1440    # 24:00

        # -------------------------
        # Load offline artifacts
        # -------------------------
        self.day_cluster = np.load(day_cluster_path, allow_pickle=True).item()
        self.top_x_od = np.load(top_x_od_path, allow_pickle=True).item()
        self.W = np.load(W_path, allow_pickle=True).item()
        self.max_w = max(self.W.values())

        if (self.s, self.e) not in self.top_x_od:
            raise KeyError(f"(s,e)=({self.s},{self.e}) not found in top_x_od")

        self.neighbors = [
            (i, j) for (i, j) in self.top_x_od[(self.s, self.e)]
            if not (i == self.s and j == self.e)
        ]

        # -------------------------
        # Load OD matrices
        # -------------------------
        files = sorted(os.listdir(data_root))
        self.OD = [
            torch.tensor(
                np.load(os.path.join(data_root, f)),
                dtype=torch.float32
            )
            for f in tqdm(files, desc="Loading OD matrices for STLSTMDataset")
        ]

        # Precompute inflow / outflow
        self.inflow = [day.sum(dim=2) for day in self.OD]   # (1440, N)
        self.outflow = [day.sum(dim=1) for day in self.OD]  # (1440, N)

        # -------------------------
        # Build sample index
        # -------------------------
        self.index = []

        max_lag = max(self.W.values())

        for d in tqdm(range(len(self.OD)), desc="Building STLSTMDataset index"):
            # Temporal Feature Extraction (cluster-based days)
            cid = self.day_cluster[d]
            L = [dd for dd in range(d) if self.day_cluster[dd] == cid]

            if len(L) < self.a:
                continue  # not enough historical days

            for t in range(
                self.day_start + self.H + max_lag,
                self.day_end - self.pred
            ):
                self.index.append((d, t))

        if len(self.index) == 0:
            raise RuntimeError("STLSTMDataset has no valid samples.")

    def __len__(self):
        return len(self.index)

    # ==========================================================
    # Eq.(26): Historical OD sequence
    # ==========================================================
    def _get_w(self, i, j):
        """
        Safe W accessor.
        Fallback to max W if missing.
        """
        if (i, j) in self.W:
            return self.W[(i, j)]
        else:
            # 논문상 W는 OD travel time upper bound
            # fallback = max observed W (conservative)
            return self.max_w
    
    def _historical_seq(self, day_list, i, j, t):
        w = self._get_w(i, j)
        seqs = []
        for dd in day_list:
            t_end = t - w
            t_start = t_end - self.H
            seqs.append(self.OD[dd][t_start:t_end, i, j])
        return torch.cat(seqs, dim=0)


    # ==========================================================
    # Eq.(27): Realtime OD sequence
    # ==========================================================
    def _realtime_seq(self, d, i, j, t):
        w = self._get_w(i, j)
        t_end = t - w
        t_start = t_end - self.h
        return self.OD[d][t_start:t_end, i, j]

    # ==========================================================
    # Dataset fetch
    # ==========================================================
    def __getitem__(self, idx):
        d, t = self.index[idx]

        # Temporal cluster days
        cid = self.day_cluster[d]
        L = [dd for dd in range(d) if self.day_cluster[dd] == cid]
        L = L[-self.a:]  # most recent a days

        rows = []

        # --------------------------------------------------
        # Target OD + Spatially correlated neighbors
        # --------------------------------------------------
        for (i, j) in [(self.s, self.e)] + self.neighbors:
            hist = self._historical_seq(L, i, j, t)
            real = self._realtime_seq(d, i, j, t)
            rows.append(torch.cat([hist, real], dim=0))

        # --------------------------------------------------
        # Outflow at origin station s
        # --------------------------------------------------
        hist_out = torch.cat([
            self.outflow[dd][t - self.H:t, self.s] for dd in L
        ])
        real_out = self.outflow[d][t - self.h:t, self.s]
        rows.append(torch.cat([hist_out, real_out], dim=0))

        # --------------------------------------------------
        # Inflow at destination station e
        # --------------------------------------------------
        hist_in = torch.cat([
            self.inflow[dd][t - self.H:t, self.e] for dd in L
        ])
        real_in = self.inflow[d][t - self.h:t, self.e]
        rows.append(torch.cat([hist_in, real_in], dim=0))

        # --------------------------------------------------
        # Final input tensor
        # --------------------------------------------------
        X = torch.stack(rows, dim=0).T   # (aH + h, x + 3)

        # Prediction target
        y = self.OD[d][t:t + self.pred, self.s, self.e].unsqueeze(-1)

        return {
            "x": X.float(),
            "y": y.float()
        }


# ==========================================
# MPGCN Dataset
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

class MetroODHyperDataset(Dataset):
    """
    ST-DAMHGN용 Metro OD Dataset

    Input:
      - tendency: (m, |V|)
      - periodicity: (n, |V|)
    Target:
      - y: (|V|)

    NOTE:
      - Hypergraph는 여기서 생성하지 않음
      - Dataset은 OD pair 시계열만 책임
    """

    def __init__(
        self,
        data_root,
        od2vid,
        valid_od_pairs,
        m,              # tendency length
        n,              # periodicity length
        hop_size,
        cache_in_mem=True
    ):
        super().__init__()

        self.data_root = data_root
        self.od2vid = od2vid
        self.valid_od_pairs = valid_od_pairs
        self.V = len(valid_od_pairs)

        self.m = m
        self.n = n
        self.hop_size = hop_size
        self.cache_mem = cache_in_mem

        # self.day_start = 5 * 60 + 30   # 330
        # self.day_end = 24 * 60         # 1440
        self.day_start = 6 * 60        # 360
        self.day_end = 23 * 60         # 1380

        file_names = sorted(os.listdir(data_root))
        self.data_paths = [os.path.join(data_root, f) for f in file_names]

        # ---------- load data ----------
        self.day_data_cache = []
        for path in self.data_paths:
            arr = np.load(path, mmap_mode='r')  # (1440, N, N)
            if cache_in_mem:
                arr = torch.tensor(arr, dtype=torch.float32)
            self.day_data_cache.append(arr)

        # ---------- build index ----------
        self.index = []
        for day_idx, file_name in tqdm(enumerate(file_names), desc="Building MetroODHyperDataset index"):
            ymd = file_name.split('_')[-1].split('.')[0]
            date = datetime.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))

            for t in range(
                self.day_start + self.m,
                self.day_end,
                self.hop_size
            ):
                # periodicity requires previous days
                if day_idx < self.n:
                    continue

                self.index.append({
                    "day_idx": day_idx,
                    "t": t,
                    "weekday": date.weekday()
                })

    def __len__(self):
        return len(self.index)

    def _extract_od_vector(self, day_data, t):
        """
        day_data: (1440, N, N)
        return: (|V|)
        """
        vec = torch.zeros(self.V)
        mat = day_data[t]

        for (i, j), vid in self.od2vid.items():
            vec[vid] = mat[i, j]

        return vec

    def __getitem__(self, idx):
        info = self.index[idx]
        day_idx = info["day_idx"]
        t = info["t"]

        # ---------- tendency ----------
        tendency = []
        for k in range(self.m):
            vec = self._extract_od_vector(
                self.day_data_cache[day_idx],
                t - k - 1
            )
            tendency.append(vec)

        tendency = torch.stack(tendency)  # (m, |V|)

        # ---------- periodicity ----------
        periodicity = []
        for k in range(self.n):
            vec = self._extract_od_vector(
                self.day_data_cache[day_idx - k - 1],
                t
            )
            periodicity.append(vec)

        periodicity = torch.stack(periodicity)  # (n, |V|)

        # ---------- target ----------
        y = self._extract_od_vector(
            self.day_data_cache[day_idx],
            t
        )

        return {
            "tendency": tendency,       # (m, |V|)
            "periodicity": periodicity, # (n, |V|)
            "y": y,                     # (|V|)
            "weekday": torch.tensor(info["weekday"])
        }

class ODFormerMetroDataset(Dataset):
    """
    ODformer 전용 지하철 OD Dataset

    Output:
        X: (T_in, N, N, F)
        Y: (T_out, N, N, F)
    """

    def __init__(
        self,
        data_root,
        window_size,
        pred_size,
        hop_size,
        use_time_feature=True,
        cache_in_mem=True,
        clip_value=None,
        is_train=True
    ):
        super().__init__()

        self.data_root = data_root
        self.window_size = window_size
        self.pred_size = pred_size
        self.hop_size = hop_size
        self.use_time_feature = use_time_feature
        self.cache_mem = cache_in_mem

        # 시간 범위
        self.day_start = 5 * 60 + 30
        self.day_end = 24 * 60

        # 파일 로드
        file_names = sorted(os.listdir(data_root))
        self.data_paths = [os.path.join(data_root, f) for f in file_names]

        self.day_cache = []
        for p in tqdm(self.data_paths, desc="Caching OD matrices"):
            if cache_in_mem:
                arr = np.load(p)  # (1440, N, N)
                self.day_cache.append(torch.tensor(arr, dtype=torch.float32))
            else:
                arr = np.load(p, mmap_mode='r')  # (1440, N, N)
                self.day_cache.append(arr)

        # =========================
        # 98% clipping 기준 설정
        # =========================
        assert clip_value is not None
        self.clip_value = clip_value

        # sliding window index
        self.indices = []
        for f_idx, fname in enumerate(file_names):
            ymd = fname.split('_')[-1].split('.')[0]
            date = datetime.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))
            weekday = date.weekday()

            for s in range(
                self.day_start,
                self.day_end - (window_size + pred_size),
                hop_size
            ):
                self.indices.append((f_idx, s, weekday))

    def __len__(self):
        return len(self.indices)

    def _build_feature(self, od, minute_idx, is_target=False):
        """
        od: (T, N, N)
        return: (T, N, N, F)
        """
        T, N, _ = od.shape

        # ===== 논문 전처리 =====
        od = torch.clamp(od, max=self.clip_value)
        od = torch.log1p(od)
        # =====================

        feats = [od.unsqueeze(-1)]

        if self.use_time_feature and not is_target:
            minute = torch.tensor(minute_idx) % 1440
            time_enc = build_time_sin_cos(minute.numpy())
            time_enc = torch.tensor(time_enc, dtype=torch.float32)
            time_enc = time_enc.view(T, 1, 1, -1).expand(T, N, N, -1)
            feats.append(time_enc)

        return torch.cat(feats, dim=-1)

    def __getitem__(self, idx):
        f_idx, start, weekday = self.indices[idx]
        day_data = self.day_cache[f_idx]

        if self.cache_mem:
            x_raw = day_data[start:start+self.window_size]
            y_raw = day_data[start+self.window_size:
                             start+self.window_size+self.pred_size]
        else:
            x_raw = torch.from_numpy(
                day_data[start:start+self.window_size].copy()
            ).float()
            y_raw = torch.from_numpy(
                day_data[start+self.window_size:
                         start+self.window_size+self.pred_size].copy()
            ).float()

        hist_minutes = torch.arange(start, start+self.window_size)
        fut_minutes = torch.arange(
            start+self.window_size,
            start+self.window_size+self.pred_size
        )

        X = self._build_feature(x_raw, hist_minutes, is_target=False)
        Y = self._build_feature(y_raw, fut_minutes, is_target=True)
        return {
            "X": X,
            "Y": Y,
            "weekday": torch.tensor(weekday),
        }


def get_mpgcn_dataset(data_root, train_subdir, val_subdir, window_size, hop_size, pred_size):
    train_path = os.path.join(data_root, train_subdir)
    val_path = os.path.join(data_root, val_subdir)
    
    trainset = MPGCNDataset(train_path, window_size, hop_size, pred_size)
    valset = MPGCNDataset(val_path, window_size, hop_size, pred_size)
    
    return trainset, valset

def get_dataset(data_root, train_subdir, val_subdir, window_size, hop_size, pred_size, time_resolution,cache_in_mem=True):
    
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
        pred_size=pred_size,
        time_resolution=time_resolution,
        cache_in_mem=cache_in_mem
    )
    valset = MetroDataset(
        data_root=val_path,
        window_size=window_size,
        hop_size=hop_size,
        pred_size=pred_size,
        time_resolution=time_resolution,
        cache_in_mem=cache_in_mem
    )
    
    return trainset, valset

def get_st_lstm_dataset(
    data_root,
    train_subdir,
    val_subdir,
    H,
    h,
    pred_size,
    target_s,
    target_e,
    a,
    day_cluster_path,
    top_x_od_path,
    W_path
):
    """
    Paper-faithful ST-LSTM dataset loader.

    All spatial / temporal artifacts are assumed
    to be precomputed offline.
    """

    train_path = os.path.join(data_root, train_subdir)
    val_path = os.path.join(data_root, val_subdir)

    trainset = STLSTMDataset(
        data_root=train_path,
        H=H,
        h=h,
        pred_size=pred_size,
        target_s=target_s,
        target_e=target_e,
        day_cluster_path=day_cluster_path,
        top_x_od_path=top_x_od_path,
        W_path=W_path,
        a=a
    )

    valset = STLSTMDataset(
        data_root=val_path,
        H=H,
        h=h,
        pred_size=pred_size,
        target_s=target_s,
        target_e=target_e,
        day_cluster_path=day_cluster_path,
        top_x_od_path=top_x_od_path,
        W_path=W_path,
        a=a
    )

    return trainset, valset

def get_stdamhgn_dataset(
    data_root,
    train_subdir,
    val_subdir,
    hypergraph_path,
    m,
    n,
    hop_size,
    cache_in_mem=True
):
    """
    Paper-faithful ST-DAMHGN dataset loader.
    """

    # -------------------------
    # load hypergraph artifacts
    # -------------------------
    hg = torch.load(hypergraph_path)

    valid_od_pairs = hg["valid_od_pairs"]
    od2vid = hg["od2vid"]

    # -------------------------
    # dataset paths
    # -------------------------
    train_path = os.path.join(data_root, train_subdir)
    val_path = os.path.join(data_root, val_subdir)

    trainset = MetroODHyperDataset(
        data_root=train_path,
        od2vid=od2vid,
        valid_od_pairs=valid_od_pairs,
        m=m,
        n=n,
        hop_size=hop_size,
        cache_in_mem=cache_in_mem
    )

    valset = MetroODHyperDataset(
        data_root=val_path,
        od2vid=od2vid,
        valid_od_pairs=valid_od_pairs,
        m=m,
        n=n,
        hop_size=hop_size,
        cache_in_mem=cache_in_mem
    )

    return trainset, valset

def get_odformer_dataset(
    data_root,
    train_subdir,
    val_subdir,
    window_size,
    hop_size,
    pred_size,
    use_time_feature=True,
    cache_in_mem=True
):
    """
    ODformer 학습용 dataset 생성 함수

    Returns:
        trainset, valset
    """

    train_path = os.path.join(data_root, train_subdir)
    val_path   = os.path.join(data_root, val_subdir)

    clip_value = compute_clip_value(train_path)

    # ---- trainset ----
    trainset = ODFormerMetroDataset(
        data_root=train_path,
        window_size=window_size,
        pred_size=pred_size,
        hop_size=hop_size,
        use_time_feature=use_time_feature,
        cache_in_mem=cache_in_mem,
        clip_value=clip_value,
        is_train=True
    )

    # ---- valset (train clip_value 재사용) ----
    valset = ODFormerMetroDataset(
        data_root=val_path,
        window_size=window_size,
        pred_size=pred_size,
        hop_size=hop_size,
        use_time_feature=use_time_feature,
        cache_in_mem=cache_in_mem,
        clip_value=trainset.clip_value,
        is_train=False
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