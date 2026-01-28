import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from collections import defaultdict, deque


def build_daily_od_matrices(dataset):
    """
    MetroDataset 객체를 받아
    하루 단위 OD matrix (N,N) 리스트 반환
    """
    daily_od_list = []

    for day_idx, day_data in enumerate(tqdm(dataset.day_data_cache)):
        # day_data: (1440, N, N)
        day_data = torch.as_tensor(day_data)

        start = dataset.day_start_minute
        end   = dataset.day_end_minute

        # 운영시간 aggregate
        daily_od = day_data[start:end].sum(dim=0)  # (N, N)

        daily_od_list.append(daily_od)

    return daily_od_list

def build_daily_od_and_flows(OD):
    """
    return:
        daily_od: list of (N,N)
        daily_inflow: list of (N,)
        daily_outflow: list of (N,)
    """
    daily_od = []
    daily_inflow = []
    daily_outflow = []

    daily_od = [torch.tensor(day).sum(dim=0) for day in OD]
    daily_in = [d.sum(dim=1) for d in daily_od]
    daily_out = [d.sum(dim=0) for d in daily_od]

    return daily_od, daily_in, daily_out

def flatten_od_matrix(od_mat):
    """
    od_mat: (N, N)
    return: (N*(N-1),)
    """
    N = od_mat.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool)
    return od_mat[mask]

def build_day_feature_matrix(daily_od_list):
    """
    return: (num_days, N*(N-1))
    """
    features = []
    for od in daily_od_list:
        vec = flatten_od_matrix(od)
        features.append(vec.numpy())
    return np.stack(features)

def kmeans_with_elbow(X, k_min=2, k_max=10):
    """
    X: (num_days, feature_dim)
    """
    distortions = []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        distortions.append(km.inertia_)

    # elbow point: 가장 큰 감소 이후 완만해지는 지점
    deltas = np.diff(distortions)
    elbow_k = k_min + np.argmin(deltas) + 1

    final_kmeans = KMeans(
        n_clusters=elbow_k,
        random_state=42,
        n_init=20
    ).fit(X)

    return final_kmeans.labels_, elbow_k

def temporal_feature_extraction_raw(OD):
    print("Step 1: Build daily OD matrices")

    daily_od_list = [torch.tensor(day).sum(dim=0) for day in OD]  # (N,N)

    print("Step 2: Vectorize OD matrices")
    X = build_day_feature_matrix(daily_od_list)

    print("Step 3: Standardize features")
    X = StandardScaler().fit_transform(X)

    print("Step 4: KMeans clustering")
    labels, k = kmeans_with_elbow(X)

    day_cluster = {i: int(l) for i, l in enumerate(labels)}

    print(f"Temporal clustering completed. k={k}")
    return day_cluster

def aggregate_training_od(daily_od, daily_inflow, daily_outflow, train_days):
    """
    train_days: list of day indices used for training
    """
    od_sum = torch.stack([daily_od[d] for d in train_days]).sum(dim=0)
    inflow_sum = torch.stack([daily_inflow[d] for d in train_days]).sum(dim=0)
    outflow_sum = torch.stack([daily_outflow[d] for d in train_days]).sum(dim=0)
    return od_sum, inflow_sum, outflow_sum

def compute_spatial_correlation(
    od_sum,
    inflow_sum,
    outflow_sum,
    dist_matrix,
    top_x=10,
    omega=(0.33, 0.33, 0.34)
):
    """
    return:
        top_x_od: dict {(s,e): [(i1,j1), (i2,j2), ...]}
    """
    N = od_sum.shape[0]
    learner = SpatialCorrelationLearner(omega)
    top_x_od = {}

    # 전체 OD pair 목록
    od_pairs = [(i, j) for i in range(N) for j in range(N) if i != j]

    for s in tqdm(range(N), desc="Spatial corr (origin)"):
        for e in range(N):
            if s == e:
                continue

            z_scores = []
            for (i, j) in od_pairs:
                # p: OD 규모
                p = od_sum[i, j]

                # q: inflow / outflow 기반
                mie = od_sum[i, e]
                msj = od_sum[s, j]
                q = learner.compute_q(
                    mie, msj,
                    outflow_sum[e] + 1e-6,
                    inflow_sum[s] + 1e-6
                )

                # r: 거리 기반
                irs = learner.compute_i(
                    inflow_sum[i] + outflow_sum[i],
                    inflow_sum[s] + outflow_sum[s],
                    dist_matrix[i, s]
                )
                ije = learner.compute_i(
                    inflow_sum[j] + outflow_sum[j],
                    inflow_sum[e] + outflow_sum[e],
                    dist_matrix[j, e]
                )
                iie = learner.compute_i(
                    inflow_sum[i] + outflow_sum[i],
                    inflow_sum[e] + outflow_sum[e],
                    dist_matrix[i, e]
                )
                ijs = learner.compute_i(
                    inflow_sum[j] + outflow_sum[j],
                    inflow_sum[s] + outflow_sum[s],
                    dist_matrix[j, s]
                )

                r = learner.compute_r(irs, ije, iie, ijs)
                z = learner.compute_z(p, q, r)
                z_scores.append(z)

            z_scores = torch.tensor(z_scores)
            idx = torch.argsort(z_scores, descending=True)[:top_x]
            top_x_od[(s, e)] = [od_pairs[i] for i in idx]

    return top_x_od

@torch.no_grad()
def compute_spatial_correlation_pruned_fast(
    OD_ts,          # (Days, T, N, N) torch
    od_sum,
    inflow_sum,
    outflow_sum,
    dist_matrix,    # np or torch
    top_x=10,
    max_hop=4,
    omega=(0.33, 0.33, 0.34),
    device="cuda",
    chunk=4096,
    eps=1e-6,
):
    """
    핵심 가속:
      - (s,e)마다 M_se 한 번만 생성
      - 후보 M_ij들을 chunk로 모아서 corr 벡터화
      - torch 연산으로 cov/std 계산
    """
    # ---- device 이동 ----
    od_sum = od_sum.to(device, non_blocking=True)
    inflow_sum = inflow_sum.to(device, non_blocking=True)
    outflow_sum = outflow_sum.to(device, non_blocking=True)

    if not torch.is_tensor(dist_matrix):
        dist_matrix_t = torch.tensor(dist_matrix, device=device)
    else:
        dist_matrix_t = dist_matrix.to(device)

    w1, w2, w3 = omega
    N = od_sum.shape[0]
    top_x_od = {}

    for s in tqdm(range(N), desc="Spatial corr (origin)"):
        # (s,e)별로 반복
        for e in range(N):
            if s == e:
                continue

            # 후보 생성 (CPU numpy로 해도 됨)
            origin_candidates = torch.where(dist_matrix_t[s] <= max_hop)[0]
            dest_candidates   = torch.where(dist_matrix_t[:, e] <= max_hop)[0]
            if origin_candidates.numel() == 0 or dest_candidates.numel() == 0:
                top_x_od[(s, e)] = []
                continue

            # 후보 (i,j) 리스트 (tensor로 구성)
            # candidates = cartesian product
            ii = origin_candidates.repeat_interleave(dest_candidates.numel())
            jj = dest_candidates.repeat(origin_candidates.numel())
            valid = ii != jj
            ii = ii[valid]
            jj = jj[valid]
            C = ii.numel()
            if C == 0:
                top_x_od[(s, e)] = []
                continue

            # -------- p_ij: corr(M_se, M_ij) --------
            y = OD_ts[:, :, s, e].reshape(-1).to(device)
            y = y - y.mean()
            y_std = y.std(unbiased=False).clamp_min(eps)

            p_scores = torch.empty(C, device=device)

            for st in range(0, C, chunk):
                ed = min(st + chunk, C)
                i_chunk = ii[st:ed]
                j_chunk = jj[st:ed]

                # build X chunk without OD_flat
                # X shape: (L, Cc)
                X = torch.stack(
                    [OD_ts[:, :, int(i), int(j)].reshape(-1) for i, j in zip(i_chunk.tolist(), j_chunk.tolist())],
                    dim=1
                ).to(device)

                X = X - X.mean(dim=0, keepdim=True)
                X_std = X.std(dim=0, unbiased=False).clamp_min(eps)

                cov = (X * y[:, None]).mean(dim=0)
                p_scores[st:ed] = (cov / (X_std * y_std)).abs()

            # -------- q_ij, r_ij: 후보마다 스칼라 계산 (벡터화 가능) --------
            # q: mie/fe_out + msj/fs_in
            mie = od_sum[ii, e]                         # (C,)
            msj = od_sum[s, jj]                         # (C,)
            fe_out = (outflow_sum[e] + eps)             # scalar
            fs_in  = (inflow_sum[s] + eps)              # scalar
            q_scores = mie / fe_out + msj / fs_in       # (C,)

            # r: 0.25*(i_rs + i_je + i_ie + i_js)
            fi = inflow_sum[ii] + outflow_sum[ii]       # (C,)
            fj = inflow_sum[jj] + outflow_sum[jj]       # (C,)
            fs = inflow_sum[s] + outflow_sum[s]         # scalar
            fe = inflow_sum[e] + outflow_sum[e]         # scalar

            d_is = dist_matrix_t[ii, s].clamp_min(1.0)  # (C,)
            d_je = dist_matrix_t[jj, e].clamp_min(1.0)
            d_ie = dist_matrix_t[ii, e].clamp_min(1.0)
            d_js = dist_matrix_t[jj, s].clamp_min(1.0)

            i_rs = (fi * fs) / (d_is * d_is + eps)
            i_je = (fj * fe) / (d_je * d_je + eps)
            i_ie = (fi * fe) / (d_ie * d_ie + eps)
            i_js = (fj * fs) / (d_js * d_js + eps)

            r_scores = 0.25 * (i_rs + i_je + i_ie + i_js)

            # -------- z --------
            z = w1 * p_scores + w2 * q_scores + w3 * r_scores

            # top-k
            topk = torch.topk(z, k=min(top_x, C), largest=True).indices
            sel_i = ii[topk].tolist()
            sel_j = jj[topk].tolist()
            top_x_od[(s, e)] = list(zip(sel_i, sel_j))

    return top_x_od

def get_candidate_od_pairs(s, e, dist_matrix, max_hop=4):
    """
    Spatial locality-based candidate selection

    Only consider OD pairs (i,j) such that:
      dist(s, i) <= max_hop
      dist(j, e) <= max_hop
    """
    origin_candidates = np.where(dist_matrix[s] <= max_hop)[0]
    dest_candidates   = np.where(dist_matrix[:, e] <= max_hop)[0]

    candidates = [
        (i, j)
        for i in origin_candidates
        for j in dest_candidates
        if i != j
    ]
    return candidates

def compute_spatial_correlation_pruned(
    OD_ts,          # (Days, 1440, N, N)
    od_sum,
    inflow_sum,
    outflow_sum,
    dist_matrix,
    top_x=10,
    max_hop=4,
    omega=(0.33, 0.33, 0.34)
):
    """
    Pruned spatial correlation learning (Paper-faithful + scalable)

    return:
        top_x_od: dict {(s,e): [(i1,j1), ...]}
    """
    N = od_sum.shape[0]
    learner = SpatialCorrelationLearner(omega)
    top_x_od = {}

    for s in tqdm(range(N), desc="Spatial corr (origin)"):
        for e in range(N):
            if s == e:
                continue

            # ----------------------------------------
            # 🔥 PRUNING: spatially valid candidates
            # ----------------------------------------
            candidate_pairs = get_candidate_od_pairs(
                s, e, dist_matrix, max_hop=max_hop
            )

            if len(candidate_pairs) == 0:
                top_x_od[(s, e)] = []
                continue

            z_scores = []

            for (i, j) in candidate_pairs:
                # -----------------------------
                # Eq.(18): p_ij
                # -----------------------------
                # target OD series: (Days*1440,)
                M_se = OD_ts[:, :, s, e].reshape(-1)
                M_ij = OD_ts[:, :, i, j].reshape(-1)

                # Pearson corr
                M_se_c = M_se - M_se.mean()
                M_ij_c = M_ij - M_ij.mean()
                den = (M_se_c.std() * M_ij_c.std() + 1e-6)
                p = torch.abs((M_se_c * M_ij_c).mean() / den)
                
                # -----------------------------
                # Eq.(19): q_ij
                # -----------------------------
                mie = od_sum[i, e]
                msj = od_sum[s, j]
                q = learner.compute_q(
                    mie, msj,
                    outflow_sum[e] + 1e-6,
                    inflow_sum[s] + 1e-6
                )

                # -----------------------------
                # Eq.(20~22): r_ij
                # -----------------------------
                irs = learner.compute_i(
                    inflow_sum[i] + outflow_sum[i],
                    inflow_sum[s] + outflow_sum[s],
                    dist_matrix[i, s]
                )
                ije = learner.compute_i(
                    inflow_sum[j] + outflow_sum[j],
                    inflow_sum[e] + outflow_sum[e],
                    dist_matrix[j, e]
                )
                iie = learner.compute_i(
                    inflow_sum[i] + outflow_sum[i],
                    inflow_sum[e] + outflow_sum[e],
                    dist_matrix[i, e]
                )
                ijs = learner.compute_i(
                    inflow_sum[j] + outflow_sum[j],
                    inflow_sum[s] + outflow_sum[s],
                    dist_matrix[j, s]
                )

                r = learner.compute_r(irs, ije, iie, ijs)

                # -----------------------------
                # Eq.(23): z_ij
                # -----------------------------
                z = learner.compute_z(p, q, r)
                z_scores.append(z)

            z_scores = torch.tensor(z_scores)
            top_idx = torch.argsort(z_scores, descending=True)[:top_x]

            top_x_od[(s, e)] = [
                candidate_pairs[i] for i in top_idx
            ]

    return top_x_od

def collect_od_travel_times(afc_records):
    """
    afc_records: iterable of dict
      { 'origin', 'dest', 'entry_time', 'exit_time' }
    """
    travel_times = defaultdict(list)

    for rec in tqdm(afc_records):
        i, j = rec['origin'], rec['dest']
        t = rec['exit_time'] - rec['entry_time']
        if t > 0:
            travel_times[(i, j)].append(t)

    return travel_times

def compute_W(travel_times, time_span_minutes=15, default_w=4):
    """
    논문의 95th Percentile 방식을 따르되, 데이터가 없는 경우를 대비
    """
    W = {}
    # travel_times: {(i,j): [duration1, duration2, ...]}
    
    for (i, j), times in travel_times.items():
        if not times:
            W[(i, j)] = default_w
            continue
            
        # 데이터가 너무 적으면(예: 1~2개) 통계가 불안정하므로 보수적으로 처리하거나
        # 그냥 계산하되 최소 샘플 수 조건을 걸 수 있음
        if len(times) < 5:
            # 샘플이 적으면 평균이나 max를 쓰거나, default 사용
            p95 = np.max(times) 
        else:
            p95 = np.percentile(times, 95)

        # 논문 수식 적용: 올림 처리 후 int 변환
        slot = int(np.ceil(p95 / time_span_minutes))
        W[(i, j)] = max(1, slot) # 최소 1 slot은 걸린다고 가정

    return W

def build_hop_distance_matrix(adj_matrix):
    """
    adj_matrix: (N, N) binary adjacency matrix
    return: dist_matrix (N, N), hop distance
    """
    N = adj_matrix.shape[0]
    dist_matrix = np.full((N, N), np.inf)

    for s in range(N):
        dist_matrix[s, s] = 0.0  # self-distance (division by zero 방지)
        q = deque([s])
        visited = {s}
        d = 1

        while q:
            for _ in range(len(q)):
                u = q.popleft()
                for v in np.where(adj_matrix[u] > 0)[0]:
                    if v not in visited:
                        visited.add(v)
                        dist_matrix[s, v] = d
                        q.append(v)
            d += 1

    return dist_matrix

class TemporalFeatureExtractor:
    """
    - 논문: 운영일(day) 패턴 기반 clustering
    - 실제 clustering(KMeans 등)은 외부 수행 가정
    """

    def __init__(self, day_labels: dict):
        """
        day_labels: {day_index: class_id}
        """
        self.day_labels = day_labels

    def get_class_days(self, target_day):
        cls = self.day_labels[target_day]
        return [d for d, c in self.day_labels.items() if c == cls and d < target_day]

class SpatialCorrelationLearner:
    """
    논문 Eq.(18)~(23) 구현
    """

    def __init__(self, omega=(0.33, 0.33, 0.34)):
        self.w1, self.w2, self.w3 = omega

    def compute_q(self, mie, msj, fe_out, fs_in):
        return mie / fe_out + msj / fs_in

    def compute_i(self, fi, fj, dij):
        return (fi * fj) / (dij ** 2 + 1e-6)

    def compute_r(self, irs, ije, iie, ijs):
        return 0.25 * (irs + ije + iie + ijs)

    def normalize(self, x, xmin, xmax):
        return (xmax - x) / (xmax - xmin + 1e-6)

    def compute_z(self, p, v, g):
        return self.w1 * p + self.w2 * v + self.w3 * g

    def select_top_x(self, z_values, od_pairs, x):
        idx = torch.argsort(z_values, descending=True)[:x]
        return [od_pairs[i] for i in idx]

class STLSTM(nn.Module):
    """
    Paper-faithful ST-LSTM
    Input : (B, aH+h, x+3)
    Output: (B, 1)
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, pred_size=30):
        super().__init__()
        self.feature_dim = input_dim
        self.gate = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid()
        )

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, pred_size)

    def forward(self, x):
        """
        x: (B, aH+h, x+3)
        """
        # -------- Gate --------
        G = self.gate(x)
        x = G * x
        
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])  # last timestep

