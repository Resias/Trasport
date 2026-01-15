import os
import argparse
import numpy as np
import torch
import random
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# ======================================================
# ARGPARSE
# ======================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="OFFLINE Hypergraph Generator for ST-DAMHGN (100% paper reproduction)"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Directory containing OD_minute_YYYYMMDD.npy files, shape=(1440, N, N)"
    )
    parser.add_argument(
        "--poi_path",
        type=str,
        default=None,
        help="Optional POI numpy file, shape=(N_station, 6)"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./hypergraphs.pt",
        help="Output path for generated hypergraphs"
    )
    parser.add_argument(
        "--k_closest",
        type=int,
        default=4,
        help="k for CLOSEST strategy"
    )
    parser.add_argument(
        "--q_sample",
        type=int,
        default=64,
        help="Q for CLUSTER & SAMPLE strategy"
    )
    parser.add_argument("--n_cluster_tendency", type=int, default=3)
    parser.add_argument("--n_cluster_poi", type=int, default=4)
    parser.add_argument(
        "--flow_threshold",
        type=int,
        default=10,
        help="OD pair filtering threshold"
    )
    return parser.parse_args()


# ======================================================
# MAIN
# ======================================================

def main():
    args = parse_args()

    random.seed(42)
    np.random.seed(42)

    DAY_START = 6 * 60      # 360
    DAY_END   = 23 * 60     # 1380

    # --------------------------------------------------
    # 1. LOAD DAILY OD DATA
    # --------------------------------------------------
    print("Loading daily OD matrices...")

    file_names = sorted([
        f for f in os.listdir(args.data_root)
        if f.endswith(".npy")
    ])

    if len(file_names) == 0:
        raise RuntimeError("No .npy files found in data_root")

    daily_data = []
    for fname in tqdm(file_names):
        arr = np.load(os.path.join(args.data_root, fname))
        if not (arr.ndim == 3 and arr.shape[0] == 1440):
            raise ValueError(
                f"Invalid shape {arr.shape} in {fname}, expected (1440, N, N)"
            )
        daily_data.append(arr)

    daily_data = np.stack(daily_data)  # (D, 1440, N, N)

    D, _, N, _ = daily_data.shape
    print(f"Loaded {D} days, {N} stations")

    # --------------------------------------------------
    # 2. FILTER VALID OD PAIRS
    # --------------------------------------------------
    print("Filtering OD pairs (max flow > threshold)...")

    max_flow = daily_data.max(axis=(0, 1))  # (N, N)
    valid_od_pairs = [
        (i, j)
        for i in range(N)
        for j in range(N)
        if max_flow[i, j] > args.flow_threshold
    ]

    V = len(valid_od_pairs)
    print(f"Valid OD vertices: {V}")

    if V == 0:
        raise RuntimeError("No valid OD pairs found. Check flow_threshold.")

    od2vid = {od: idx for idx, od in enumerate(valid_od_pairs)}

    # --------------------------------------------------
    # 3. BUILD TENDENCY TIME SERIES (MEAN DAILY PATTERN)
    # --------------------------------------------------
    print("Building tendency time series...")

    T = DAY_END - DAY_START
    TS = np.zeros((V, T), dtype=np.float32)

    for v, (i, j) in enumerate(valid_od_pairs):
        ts = daily_data[:, DAY_START:DAY_END, i, j]  # (D, T)
        TS[v] = ts.mean(axis=0)

    # --------------------------------------------------
    # 4. CORRELATION MATRIX (TENDENCY)
    # --------------------------------------------------
    print("Computing correlation matrix (COR)...")

    def moving_average(ts, window=3):
        return np.convolve(ts, np.ones(window)/window, mode="same")

    TS_trend = np.stack(
        [moving_average(TS[v]) for v in range(V)],
        axis=0
    )

    COR = np.corrcoef(TS_trend)
    COR = np.nan_to_num(COR)

    # --------------------------------------------------
    # 5. LOAD POI FEATURES (WITH FALLBACK)
    # --------------------------------------------------
    use_poi = False

    if args.poi_path is not None and os.path.exists(args.poi_path):
        print("Loading POI features...")
        POI_station = np.load(args.poi_path)

        if POI_station.shape != (N, 6):
            raise ValueError(
                f"POI shape mismatch: expected ({N}, 6), got {POI_station.shape}"
            )

        POI = np.zeros((V, 12), dtype=np.float32)
        for v, (i, j) in enumerate(valid_od_pairs):
            POI[v] = np.concatenate([POI_station[i], POI_station[j]])

        POI_DIST = np.linalg.norm(
            POI[:, None, :] - POI[None, :, :],
            axis=-1
        )
        use_poi = True
    else:
        print(
            "WARNING: POI data not found. "
            "Using identity hypergraphs for POI-based hypergraphs."
        )

    # --------------------------------------------------
    # 6. HYPEREDGE CONSTRUCTION FUNCTIONS
    # --------------------------------------------------
    def build_closest(sim_matrix, k):
        hyperedges = []
        for v in range(sim_matrix.shape[0]):
            idx = np.argsort(-sim_matrix[v])[:k + 1]
            hyperedges.append(idx.tolist())
        return hyperedges

    def cluster_and_sample(cluster_id, Q):
        clusters = {}
        for v, c in enumerate(cluster_id):
            clusters.setdefault(c, []).append(v)

        hyperedges = []
        for v, c in enumerate(cluster_id):
            pool = clusters[c]
            hyperedges.append(
                random.sample(pool, min(Q, len(pool)))
            )
        return hyperedges

    # --------------------------------------------------
    # 7. BUILD FOUR HYPERGRAPHS
    # --------------------------------------------------
    print("Constructing hypergraphs...")

    # (i) tendency + CLOSEST
    HG1 = build_closest(COR, args.k_closest)

    # (ii) tendency + CLUSTER & SAMPLE (hierarchical)
    dist_mat = 1 - COR
    condensed_dist = squareform(dist_mat, checks=False)

    Z = linkage(condensed_dist, method="average")
    cluster_t = fcluster(Z, t=args.n_cluster_tendency, criterion="maxclust")
    HG2 = cluster_and_sample(cluster_t, args.q_sample)

    # (iii) POI + CLOSEST
    if use_poi:
        HG3 = build_closest(-POI_DIST, args.k_closest)
    else:
        HG3 = [[v] for v in range(V)]

    # (iv) POI + CLUSTER & SAMPLE
    if use_poi:
        cluster_p = KMeans(
            n_clusters=args.n_cluster_poi,
            random_state=42
        ).fit(POI).labels_
        HG4 = cluster_and_sample(cluster_p, args.q_sample)
    else:
        HG4 = [[v] for v in range(V)]

    # --------------------------------------------------
    # 8. SAVE
    # --------------------------------------------------
    print("Saving hypergraphs...")

    hypergraphs = {
        "tendency_closest": HG1,
        "tendency_cluster": HG2,
        "poi_closest": HG3,
        "poi_cluster": HG4,
        "valid_od_pairs": valid_od_pairs,
        "od2vid": od2vid,
        "use_poi": use_poi
    }

    torch.save(hypergraphs, args.save_path)
    print(f"Saved hypergraphs to: {args.save_path}")
    print("DONE.")
    
    hg = torch.load(args.save_path)
    print(hg.keys())
    print(len(hg["tendency_closest"]))
    print(hg["use_poi"])

if __name__ == "__main__":
    main()
