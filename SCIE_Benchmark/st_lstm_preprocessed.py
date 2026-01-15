from ST_LSTM import temporal_feature_extraction,\
      build_daily_od_and_flows, aggregate_training_od,\
          compute_spatial_correlation, collect_od_travel_times,\
              compute_W, build_hop_distance_matrix, \
              get_candidate_od_pairs, compute_spatial_correlation_pruned
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import get_dataset

trainset, valset = get_dataset(
    data_root='/home/data/od_minute',
    train_subdir='train',
    val_subdir='test',
    window_size=60,
    hop_size=5,
    pred_size=30,
    cache_in_mem=True
)


day_cluster = temporal_feature_extraction(trainset)

np.save("day_cluster.npy", day_cluster)

daily_od, daily_in, daily_out = build_daily_od_and_flows(trainset)

# train_days = list(range(len(daily_od) - 7))
od_sum, in_sum, out_sum = aggregate_training_od(
    daily_od, daily_in, daily_out, range(len(daily_od))
)

od_df = pd.read_csv("AD_matrix_trimmed_common.csv", index_col=0)
adj_matrix = od_df.values

station_dist_matrix = build_hop_distance_matrix(adj_matrix)


def build_W_from_hop_distance(
    dist_matrix,
    minutes_per_hop=2.5,
    max_hop=4
):
    """
    Build W(i,j) only for spatially reachable OD pairs

    - dist_matrix[i,j] == inf  → skip
    - dist_matrix[i,j] > max_hop → skip
    """
    W = {}
    N = dist_matrix.shape[0]

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            d = dist_matrix[i, j]

            # -----------------------------
            # 🔥 핵심 가드
            # -----------------------------
            if not np.isfinite(d):
                continue
            if d > max_hop:
                continue

            est_time = d * minutes_per_hop
            W[(i, j)] = int(np.ceil(est_time / 15))

    return W

W = build_W_from_hop_distance(
    station_dist_matrix,
    minutes_per_hop=2.5,
    max_hop=4
)

np.save("W.npy", W)

# top_x_od = compute_spatial_correlation(
#     od_sum, in_sum, out_sum,
#     dist_matrix=station_dist_matrix,
#     top_x=10
# )
top_x_od = compute_spatial_correlation_pruned(
    od_sum,
    in_sum,
    out_sum,
    dist_matrix=station_dist_matrix,
    top_x=10,
    max_hop=4
)
np.save("top_x_od.npy", top_x_od)

