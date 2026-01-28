from ST_LSTM import (
    temporal_feature_extraction_raw,
    build_daily_od_and_flows,
    aggregate_training_od,
    compute_W,               # [변경] Fast 버전 임포트
    compute_spatial_correlation_pruned,
    build_hop_distance_matrix
)
import pandas as pd
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import get_dataset

train_root = "/workspace/od_minute/train"

files = sorted([
    os.path.join(train_root, f)
    for f in os.listdir(train_root)
    if f.endswith(".npy")
])

print(f"Loading {len(files)} raw daily OD files...")

OD = [np.load(f) for f in files]   # each: (1440,N,N)
OD_ts = torch.tensor(np.stack(OD), dtype=torch.float32)

day_cluster = temporal_feature_extraction_raw(OD)
np.save("day_cluster.npy", day_cluster)

daily_od, daily_in, daily_out = build_daily_od_and_flows(OD)

# train_days = list(range(len(daily_od) - 7))
od_sum, in_sum, out_sum = aggregate_training_od(
    daily_od, daily_in, daily_out, range(len(daily_od))
)

od_df = pd.read_csv("AD_matrix_trimmed_common.csv", index_col=0)
adj_matrix = od_df.values

station_dist_matrix = build_hop_distance_matrix(adj_matrix)

def build_W_from_hop_distance(dist_matrix, minutes_per_hop=5.0, time_span_minutes=15, max_hop=None):
    """
    거리(Hop) 기반으로 Travel Time Limit (W)를 추정하는 함수.
    Raw 데이터가 없을 때 사용하는 Fallback 로직입니다.
    
    Args:
        dist_matrix (np.ndarray): (N, N) 크기의 Hop Distance 행렬 (inf는 도달 불가)
        minutes_per_hop (float): 1 Hop당 예상 소요 시간 (분). 
                                 환승/대기/도보 시간을 고려하여 넉넉하게(4~5분) 잡는 것이 안전함.
        time_span_minutes (int): 데이터 집계 단위 (보통 15분)
        max_hop (int, optional): 계산할 최대 Hop 수. 이보다 먼 거리는 계산에서 제외(Dataset의 max_w fallback 사용됨).
        
    Returns:
        dict: {(origin_idx, dest_idx): time_slots (int)}
    """
    W = {}
    rows, cols = dist_matrix.shape
    
    # 루프를 돌며 계산 (N=637일 때 약 40만번 반복, 순식간에 끝남)
    for i in range(rows):
        for j in range(cols):
            # 1. 자기 자신 제외
            if i == j:
                continue
            
            d = dist_matrix[i, j]
            
            # 2. 도달 불가능(inf) 제외
            if not np.isfinite(d):
                continue
                
            # 3. max_hop 필터링 (설정된 경우만)
            if max_hop is not None and d > max_hop:
                continue
            
            # 4. W 계산 로직 (핵심)
            # 예상 소요 시간 = Hop 수 * (주행+정차+환승 고려 시간)
            est_minutes = d * minutes_per_hop
            
            # Time Slot 단위로 변환 (올림 처리)
            # 예: 18분 소요, 15분 단위 -> 2 slot (t-2 시점 데이터 사용)
            w_slot = int(np.ceil(est_minutes / time_span_minutes))
            
            # 최소 1 slot 이상 지연된다고 가정 (데이터 집계 및 전송 시간 고려)
            W[(i, j)] = max(1, w_slot)
            
    return W

W = build_W_from_hop_distance(
    station_dist_matrix,
    minutes_per_hop=3.0,  # [중요] 보수적으로 5분 설정 (Data Leakage 방지)
    time_span_minutes=1, # 데이터셋의 window_size와는 다름, 집계 간격임
    max_hop=None          # 전체 네트워크에 대해 계산 (Dataset의 fallback 최소화)
)
np.save("W.npy", W)

top_x_od = compute_spatial_correlation_pruned(
    OD_ts,
    od_sum,
    in_sum,
    out_sum,
    dist_matrix=station_dist_matrix,
    top_x=10,
    max_hop=4
)
np.save("top_x_od.npy", top_x_od)
print("Preprocessing Done.")

