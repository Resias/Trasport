# file: st_lstm_preprocessed_mta_hourly.py
"""
Preprocess MTA hourly "daily-like" OD files for ST-LSTM.

Assumes you already created:
  data_root/
    train/
      YYYYMMDD.npy
      YYYYMMDD.mask.npy   (optional)
      YYYYMMDD.time.npy   (optional)
    test/
      YYYYMMDD.npy
      YYYYMMDD.mask.npy   (optional)
      YYYYMMDD.time.npy   (optional)

Each YYYYMMDD.npy is (T_day, N, N) where T_day is usually 24 (hourly).

Outputs (by default next to this script, configurable):
  out_dir/
    day_cluster.train.npy    : dict {day_idx: cluster_id}  (weekday by default)
    day_cluster.test.npy     : dict {day_idx: cluster_id}  (weekday by default)
    W.npy                    : dict {(i,j): w_step}        (default constant lag = 1)
    top_x_od.npy             : dict {(s,e): [(i1,j1),...]} neighbors for target OD(s)
    dist_hop.npy             : hop-distance matrix (N,N) from kNN graph
    adj_knn.npy              : adjacency matrix (N,N) of kNN graph (binary)
    meta.json                : run metadata

IMPORTANT:
- For ST-LSTM you typically compute artifacts from TRAIN only to avoid leakage.
- This script creates day_cluster for both train/test separately (indexing differs per folder).
- Neighbor selection uses a kNN graph built from stations lat/lon (not true rail topology).
"""

import os
import re
import json
import math
import datetime
from pathlib import Path
from collections import deque

import numpy as np
import torch
from tqdm import tqdm


# -----------------------------
# Utilities
# -----------------------------
def list_daily_od_files(folder: Path):
    files = sorted(
        f for f in folder.iterdir()
        if f.name.endswith(".npy")
        and not f.name.endswith(".time.npy")
        and not f.name.endswith(".mask.npy")
    )
    return files


def parse_ymd_from_filename(path: Path) -> str:
    m = re.search(r"(\d{8})", path.stem)
    if m is None:
        raise ValueError(f"Cannot parse YYYYMMDD from filename: {path.name}")
    return m.group(1)


def weekday_from_ymd(ymd: str) -> int:
    d = datetime.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))
    return d.weekday()  # 0=Mon


def load_days_as_tensor(files, dtype=torch.float32, device="cpu"):
    """
    Loads list of daily files into torch tensor (Days, T, N, N).
    """
    od_list = []
    for f in tqdm(files, desc=f"Loading OD days ({device})"):
        arr = np.load(f)
        od_list.append(arr.astype(np.float32, copy=False))
    od = torch.tensor(np.stack(od_list, axis=0), dtype=dtype, device=device)
    return od


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Vectorized haversine (km). Inputs in degrees.
    """
    R = 6371.0
    lat1 = np.deg2rad(lat1); lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2); lon2 = np.deg2rad(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def build_knn_adjacency_from_stations(
    stations_csv: Path,
    k: int = 8,
    self_loops: bool = False
) -> np.ndarray:
    """
    Build symmetric kNN adjacency using station lat/lon.
    Returns binary adjacency (N,N).
    """
    import pandas as pd  # local import to keep top clean
    stations = pd.read_csv(stations_csv)
    if not {"lat", "lon"}.issubset(set(stations.columns)):
        raise ValueError("stations.csv must contain lat and lon columns.")

    lat = stations["lat"].to_numpy(np.float64)
    lon = stations["lon"].to_numpy(np.float64)
    N = lat.shape[0]

    # pairwise distance: compute in blocks if N large
    # For N ~ few hundreds, full NxN is fine. For safety, we still do blocks.
    dist = np.full((N, N), np.inf, dtype=np.float64)

    block = 512
    for i0 in tqdm(range(0, N, block), desc="Haversine distance blocks"):
        i1 = min(i0 + block, N)
        lat1 = lat[i0:i1][:, None]
        lon1 = lon[i0:i1][:, None]
        # broadcast against all points
        dist[i0:i1] = haversine_km(lat1, lon1, lat[None, :], lon[None, :])

    # Build kNN adjacency
    adj = np.zeros((N, N), dtype=np.int8)
    for i in range(N):
        # exclude self
        order = np.argsort(dist[i])
        neigh = [j for j in order if j != i][:k]
        adj[i, neigh] = 1

    # symmetrize
    adj = np.maximum(adj, adj.T)

    if self_loops:
        np.fill_diagonal(adj, 1)

    return adj


def build_hop_distance_matrix(adj: np.ndarray) -> np.ndarray:
    """
    BFS hop distance over an unweighted graph.
    adj: binary (N,N)
    return dist: (N,N) with np.inf for unreachable, dist[i,i]=0
    """
    N = adj.shape[0]
    dist = np.full((N, N), np.inf, dtype=np.float32)
    neigh = [np.where(adj[i] > 0)[0].tolist() for i in range(N)]

    for s in tqdm(range(N), desc="Hop-distance BFS"):
        dist[s, s] = 0.0
        q = deque([s])
        while q:
            u = q.popleft()
            du = dist[s, u]
            for v in neigh[u]:
                if dist[s, v] == np.inf:
                    dist[s, v] = du + 1.0
                    q.append(v)
    return dist


def build_daily_sums(OD_ts: torch.Tensor):
    """
    OD_ts: (Days, T, N, N)
    Returns:
      od_sum: (N,N)
      inflow_sum: (N,)  (sum over all origins into node)  -> column sums
      outflow_sum: (N,) (sum over all destinations out of node) -> row sums
    """
    # daily aggregate -> sum over Days and T
    od_sum = OD_ts.sum(dim=(0, 1))             # (N,N)
    outflow_sum = od_sum.sum(dim=1)            # (N,)  rows
    inflow_sum = od_sum.sum(dim=0)             # (N,)  cols
    return od_sum, inflow_sum, outflow_sum


@torch.no_grad()
def compute_top_x_for_targets_pruned_fast(
    OD_ts: torch.Tensor,        # (Days, T, N, N) torch
    od_sum: torch.Tensor,       # (N,N)
    inflow_sum: torch.Tensor,   # (N,)
    outflow_sum: torch.Tensor,  # (N,)
    dist_hop: np.ndarray,       # (N,N) hop distance
    targets,                    # list[(s,e)]
    top_x: int = 10,
    max_hop: int = 4,
    omega=(0.33, 0.33, 0.34),
    device="cuda",
    chunk: int = 512,
    eps: float = 1e-6,
):
    """
    Fast-ish neighbor selection for only specified (s,e).
    - Candidate (i,j): dist(s,i)<=max_hop and dist(j,e)<=max_hop and i!=j
    - p: Pearson corr between M_se and M_ij (flattened over Days*T)
    - q: mie/fe_out + msj/fs_in   using od_sum/inflow/outflow
    - r: 0.25*(i_rs+i_je+i_ie+i_js) using hop distances
    """
    w1, w2, w3 = omega

    OD_ts = OD_ts.to(device)
    od_sum = od_sum.to(device)
    inflow_sum = inflow_sum.to(device)
    outflow_sum = outflow_sum.to(device)

    dist_t = torch.tensor(dist_hop, device=device, dtype=torch.float32)

    Days, T, N, _ = OD_ts.shape
    top_x_od = {}

    for (s, e) in targets:
        s = int(s); e = int(e)
        if s == e:
            top_x_od[(s, e)] = []
            continue

        # candidates
        origin_candidates = torch.where(dist_t[s] <= max_hop)[0]
        dest_candidates   = torch.where(dist_t[:, e] <= max_hop)[0]
        if origin_candidates.numel() == 0 or dest_candidates.numel() == 0:
            top_x_od[(s, e)] = []
            continue

        ii = origin_candidates.repeat_interleave(dest_candidates.numel())
        jj = dest_candidates.repeat(origin_candidates.numel())
        valid = ii != jj
        ii = ii[valid]
        jj = jj[valid]
        C = ii.numel()
        if C == 0:
            top_x_od[(s, e)] = []
            continue

        # target series
        y = OD_ts[:, :, s, e].reshape(-1)
        y = y - y.mean()
        y_std = y.std(unbiased=False).clamp_min(eps)

        p_scores = torch.empty(C, device=device)

        # chunked corr
        for st in range(0, C, chunk):
            ed = min(st + chunk, C)
            i_chunk = ii[st:ed]
            j_chunk = jj[st:ed]

            # stack candidate series: (L, Cc)
            # This uses Python list in chunk; OK for modest C/topology.
            X = torch.stack(
                [OD_ts[:, :, int(i), int(j)].reshape(-1) for i, j in zip(i_chunk.tolist(), j_chunk.tolist())],
                dim=1
            )

            X = X - X.mean(dim=0, keepdim=True)
            X_std = X.std(dim=0, unbiased=False).clamp_min(eps)
            cov = (X * y[:, None]).mean(dim=0)
            p_scores[st:ed] = (cov / (X_std * y_std)).abs()

        # q
        mie = od_sum[ii, e]
        msj = od_sum[s, jj]
        fe_out = (outflow_sum[e] + eps)
        fs_in  = (inflow_sum[s] + eps)
        q_scores = mie / fe_out + msj / fs_in

        # r
        fi = inflow_sum[ii] + outflow_sum[ii]
        fj = inflow_sum[jj] + outflow_sum[jj]
        fs = inflow_sum[s] + outflow_sum[s]
        fe = inflow_sum[e] + outflow_sum[e]

        d_is = dist_t[ii, s].clamp_min(1.0)
        d_je = dist_t[jj, e].clamp_min(1.0)
        d_ie = dist_t[ii, e].clamp_min(1.0)
        d_js = dist_t[jj, s].clamp_min(1.0)

        i_rs = (fi * fs) / (d_is * d_is + eps)
        i_je = (fj * fe) / (d_je * d_je + eps)
        i_ie = (fi * fe) / (d_ie * d_ie + eps)
        i_js = (fj * fs) / (d_js * d_js + eps)

        r_scores = 0.25 * (i_rs + i_je + i_ie + i_js)

        z = w1 * p_scores + w2 * q_scores + w3 * r_scores

        k = min(top_x, C)
        topk = torch.topk(z, k=k, largest=True).indices
        sel_i = ii[topk].tolist()
        sel_j = jj[topk].tolist()
        top_x_od[(s, e)] = list(zip(sel_i, sel_j))

    return top_x_od


def save_dict_npy(path: Path, d: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), d, allow_pickle=True)


# -----------------------------
# Main
# -----------------------------
def main():
    import argparse

    p = argparse.ArgumentParser("MTA hourly ST-LSTM preprocessing")

    # Paths
    p.add_argument("--data_root", type=str, default='/home/data/MTA_dataset_NY/od_daily_from_sequence', help="root containing train/ and test/")
    p.add_argument("--stations_csv", type=str, default='/home/data/MTA_dataset_NY/od_daily_from_sequence/stations.csv',
                   help="Path to stations.csv (lat/lon). If None, will try data_root/../od_sequence_out/stations.csv")
    p.add_argument("--out_dir", type=str, default="st_lstm_artifacts_mta_hourly")

    # Topology / neighbors
    p.add_argument("--knn_k", type=int, default=8, help="k for kNN graph on lat/lon")
    p.add_argument("--max_hop", type=int, default=4, help="max hop for candidate pruning")
    p.add_argument("--top_x", type=int, default=10, help="number of neighbors")
    p.add_argument("--omega", type=float, nargs=3, default=(0.33, 0.33, 0.34))

    # Targets
    p.add_argument("--targets", type=str, default="10,25",
                   help='targets as "s,e; s,e; ...", example: "10,25; 3,7"')

    # W (lag)
    p.add_argument("--default_w", type=int, default=1, help="constant W lag (in steps)")

    # Device/compute
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--chunk", type=int, default=512)

    args = p.parse_args()

    data_root = Path(args.data_root)
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve stations.csv
    if args.stations_csv is not None:
        stations_csv = Path(args.stations_csv)
    else:
        # common fallback: od_sequence_out/stations.csv next to data_root
        stations_csv = (data_root.parent / "od_sequence_out" / "stations.csv")
    if not stations_csv.exists():
        raise FileNotFoundError(f"stations.csv not found: {stations_csv}")

    # Parse targets
    targets = []
    for part in args.targets.split(";"):
        part = part.strip()
        if not part:
            continue
        s, e = part.split(",")
        targets.append((int(s.strip()), int(e.strip())))
    if len(targets) == 0:
        raise ValueError("No targets parsed. Use --targets '10,25' or '10,25;3,7'")

    # List files
    train_files = list_daily_od_files(train_dir)
    test_files = list_daily_od_files(test_dir)
    if len(train_files) == 0:
        raise RuntimeError(f"No daily OD files in {train_dir}")
    if len(test_files) == 0:
        print(f"[WARN] No daily OD files in {test_dir}. You can still generate train artifacts.")

    # Sanity: shapes
    sample = np.load(train_files[0])
    if sample.ndim != 3:
        raise ValueError(f"Expected (T,N,N) in {train_files[0].name}, got shape {sample.shape}")
    T_day, N, N2 = sample.shape
    if N != N2:
        raise ValueError(f"OD matrix must be square (N,N), got {sample.shape}")
    print(f"[OK] Found train days={len(train_files)}, T_day={T_day}, N={N}")

    # -----------------------------
    # 1) day_cluster (weekday-based) for train/test separately
    # -----------------------------
    day_cluster_train = {i: weekday_from_ymd(parse_ymd_from_filename(f)) for i, f in enumerate(train_files)}
    save_dict_npy(out_dir / "day_cluster.train.npy", day_cluster_train)

    if len(test_files) > 0:
        day_cluster_test = {i: weekday_from_ymd(parse_ymd_from_filename(f)) for i, f in enumerate(test_files)}
        save_dict_npy(out_dir / "day_cluster.test.npy", day_cluster_test)

    # -----------------------------
    # 2) W.npy (constant lag)
    #   Minimal dict is enough; STLSTMDataset falls back to max_w.
    # -----------------------------
    W = {(0, 1): int(args.default_w)}
    save_dict_npy(out_dir / "W.npy", W)

    # -----------------------------
    # 3) Build kNN adjacency + hop distance from stations lat/lon
    # -----------------------------
    print("[3] Building kNN adjacency from stations lat/lon...")
    adj = build_knn_adjacency_from_stations(stations_csv, k=args.knn_k, self_loops=False)
    np.save(out_dir / "adj_knn.npy", adj)

    print("[3] Building hop-distance matrix...")
    dist_hop = build_hop_distance_matrix(adj)
    np.save(out_dir / "dist_hop.npy", dist_hop)

    # -----------------------------
    # 4) Build top_x_od for target(s) using TRAIN ONLY (avoid leakage)
    # -----------------------------
    print("[4] Loading TRAIN OD_ts for neighbor selection...")
    OD_ts = load_days_as_tensor(train_files, dtype=torch.float32, device=args.device)

    # Optionally drop invalid hours using mask: simple version (set invalid to 0)
    # If you want strict sample filtering, do it in STLSTMDataset index via mask.
    # Here we only keep it simple.

    print("[4] Computing train aggregates...")
    od_sum, inflow_sum, outflow_sum = build_daily_sums(OD_ts)

    print("[4] Selecting spatial neighbors for targets (train only)...")
    top_x_od = compute_top_x_for_targets_pruned_fast(
        OD_ts=OD_ts,
        od_sum=od_sum,
        inflow_sum=inflow_sum,
        outflow_sum=outflow_sum,
        dist_hop=dist_hop,
        targets=targets,
        top_x=args.top_x,
        max_hop=args.max_hop,
        omega=tuple(args.omega),
        device=args.device,
        chunk=args.chunk,
    )
    save_dict_npy(out_dir / "top_x_od.npy", top_x_od)

    meta = {
        "data_root": str(data_root),
        "train_days": len(train_files),
        "test_days": len(test_files),
        "T_day": int(T_day),
        "N": int(N),
        "stations_csv": str(stations_csv),
        "knn_k": int(args.knn_k),
        "max_hop": int(args.max_hop),
        "top_x": int(args.top_x),
        "omega": list(args.omega),
        "targets": targets,
        "default_w": int(args.default_w),
        "device": args.device,
        "chunk": int(args.chunk),
        "note": "day_cluster is weekday-based; top_x_od and W computed from train only.",
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("✅ Preprocessing complete.")
    print(f"Artifacts saved to: {out_dir.resolve()}")
    print("Outputs:")
    for name in ["day_cluster.train.npy", "day_cluster.test.npy", "W.npy", "top_x_od.npy", "adj_knn.npy", "dist_hop.npy", "meta.json"]:
        pth = out_dir / name
        if pth.exists():
            print(" -", pth.name)


if __name__ == "__main__":
    main()
