import os
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm


# =====================================================
# Utils
# =====================================================
def load_csv(path):
    return pd.read_csv(path, index_col=0)


def list_minutes(dir_path):
    return sorted([
        f.replace("od_minute_", "").replace(".csv", "")
        for f in os.listdir(dir_path)
        if f.startswith("od_minute_")
    ])


# =====================================================
# Core computation
# =====================================================
def compute_day_station_metrics(day_dir):
    """
    Returns:
        station_mae: dict
        station_avg_flow: dict
    """
    pred_dir = os.path.join(day_dir, "minute")
    real_dir = os.path.join(day_dir, "real_minute")

    minutes = list_minutes(pred_dir)

    station_abs_error = defaultdict(list)
    station_flow = defaultdict(list)

    for m in minutes:
        pred = load_csv(os.path.join(pred_dir, f"od_minute_{m}.csv"))
        real = load_csv(os.path.join(real_dir, f"od_minute_{m}.csv"))

        # 역 총 교통량 = 출발 + 도착
        pred_flow = pred.sum(axis=1) + pred.sum(axis=0)
        real_flow = real.sum(axis=1) + real.sum(axis=0)

        abs_err = (pred_flow - real_flow).abs()

        for station in pred.index:
            station_abs_error[station].append(abs_err.loc[station])
            station_flow[station].append(real_flow.loc[station])

    station_mae = {
        s: np.mean(v) for s, v in station_abs_error.items()
    }

    station_avg_flow = {
        s: np.mean(v) for s, v in station_flow.items()
    }

    return station_mae, station_avg_flow


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser("Station-level Performance Analysis")
    parser.add_argument(
        "--result_root",
        default="./inference_results_no_mix",
        help="Inference result directory"
    )
    parser.add_argument(
        "--min_avg_flow",
        type=float,
        default=50.0,
        help="Minimum average daily flow to consider a station"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10
    )
    parser.add_argument(
        "--save_csv",
        action="store_true",
        help="Save full station metrics as CSV"
    )

    args = parser.parse_args()

    all_station_mae = defaultdict(list)
    all_station_flow = defaultdict(list)

    day_dirs = sorted([
        os.path.join(args.result_root, d)
        for d in os.listdir(args.result_root)
        if d.startswith("day_")
    ])

    print(f"[INFO] Found {len(day_dirs)} days")

    for day_dir in tqdm(day_dirs, desc="Processing days"):
        day_mae, day_flow = compute_day_station_metrics(day_dir)

        for station in day_mae:
            all_station_mae[station].append(day_mae[station])
            all_station_flow[station].append(day_flow[station])

    # =====================================================
    # Aggregate over days
    # =====================================================
    rows = []
    for station in all_station_mae:
        mae = np.mean(all_station_mae[station])
        avg_flow = np.mean(all_station_flow[station])

        rows.append({
            "station": station,
            "MAE": mae,
            "avg_daily_flow": avg_flow
        })

    df = pd.DataFrame(rows)
    print("[DEBUG] Total stations:", len(df))
    print("[DEBUG] avg_daily_flow stats:")
    print(df["avg_daily_flow"].describe())
    # =====================================================
    # Filter low-demand stations
    # =====================================================
    df_valid = df[df["avg_daily_flow"] >= args.min_avg_flow].copy()
    print("[DEBUG] Stations after filtering:", len(df_valid))

    df_valid = df_valid.sort_values("MAE")

    # =====================================================
    # Top / Bottom K
    # =====================================================
    top_k = df_valid.head(args.top_k)
    bottom_k = df_valid.tail(args.top_k).sort_values("MAE", ascending=False)

    print("\n================ TOP Stations (Best Performance) ================")
    for i, row in enumerate(top_k.itertuples(), 1):
        print(
            f"{i:2d}. {row.station:<20} "
            f"| MAE={row.MAE:8.2f} "
            f"| AvgFlow={row.avg_daily_flow:10.1f}"
        )

    print("\n================ BOTTOM Stations (Worst Performance) ================")
    for i, row in enumerate(bottom_k.itertuples(), 1):
        print(
            f"{i:2d}. {row.station:<20} "
            f"| MAE={row.MAE:8.2f} "
            f"| AvgFlow={row.avg_daily_flow:10.1f}"
        )

    # =====================================================
    # Save CSV
    # =====================================================
    if args.save_csv:
        save_path = os.path.join(args.result_root, "station_level_metrics.csv")
        df_valid.to_csv(save_path, index=False)
        print(f"\n[INFO] Saved station metrics → {save_path}")


if __name__ == "__main__":
    main()
