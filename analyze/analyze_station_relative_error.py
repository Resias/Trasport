import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


def smape(y_true, y_pred, eps=1e-6):
    return np.mean(
        2.0 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + eps)
    )


# =====================================================
# Compute station-level sMAPE
# =====================================================
def compute_station_smape(day_dir):
    pred_dir = os.path.join(day_dir, "minute")
    real_dir = os.path.join(day_dir, "real_minute")

    minutes = list_minutes(pred_dir)

    station_smape_values = defaultdict(list)

    for m in minutes:
        pred = load_csv(os.path.join(pred_dir, f"od_minute_{m}.csv"))
        real = load_csv(os.path.join(real_dir, f"od_minute_{m}.csv"))

        pred_flow = pred.sum(axis=1) + pred.sum(axis=0)
        real_flow = real.sum(axis=1) + real.sum(axis=0)

        for station in pred.index:
            s = smape(
                real_flow.loc[station],
                pred_flow.loc[station]
            )
            station_smape_values[station].append(s)

    return {
        s: np.mean(v)
        for s, v in station_smape_values.items()
    }


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser("Relative Error & Scatter Analysis")
    parser.add_argument(
        "--result_root",
        default="./inference_results_no_mix"
    )
    parser.add_argument(
        "--station_metric_csv",
        default="./inference_results_no_mix/station_level_metrics.csv"
    )
    parser.add_argument(
        "--min_avg_flow",
        type=float,
        default=50.0
    )
    parser.add_argument(
        "--save_dir",
        default="./analysis_results"
    )

    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # =====================================================
    # Load MAE / AvgFlow
    # =====================================================
    df = pd.read_csv(args.station_metric_csv)
    df = df[df["avg_daily_flow"] >= args.min_avg_flow].copy()

    # =====================================================
    # Compute sMAPE across days
    # =====================================================
    station_smape_all = defaultdict(list)

    day_dirs = sorted([
        os.path.join(args.result_root, d)
        for d in os.listdir(args.result_root)
        if d.startswith("day_")
    ])

    for day_dir in tqdm(day_dirs, desc="Computing sMAPE"):
        day_smape = compute_station_smape(day_dir)
        for station, v in day_smape.items():
            station_smape_all[station].append(v)

    df["sMAPE"] = df["station"].map(
        lambda s: np.mean(station_smape_all[s])
        if s in station_smape_all else np.nan
    )

    # =====================================================
    # Save updated metrics
    # =====================================================
    metric_save_path = os.path.join(
        args.save_dir,
        "station_metrics_with_smape.csv"
    )
    df.to_csv(metric_save_path, index=False)
    print(f"[INFO] Saved metrics → {metric_save_path}")

    # =====================================================
    # Scatter plot: AvgFlow vs MAE
    # =====================================================
    plt.figure()
    plt.scatter(
        np.log10(df["avg_daily_flow"]),
        df["MAE"]
    )
    plt.xlabel("log10(Average Daily Flow)")
    plt.ylabel("MAE")
    plt.title("Station-level MAE vs Demand")

    plt.tight_layout()
    plt.savefig(
        os.path.join(args.save_dir, "avgflow_vs_mae.png"),
        dpi=300
    )
    plt.close()

    # =====================================================
    # Scatter plot: AvgFlow vs sMAPE
    # =====================================================
    plt.figure()
    plt.scatter(
        np.log10(df["avg_daily_flow"]),
        df["sMAPE"]
    )
    plt.xlabel("log10(Average Daily Flow)")
    plt.ylabel("sMAPE")
    plt.title("Station-level sMAPE vs Demand")

    plt.tight_layout()
    plt.savefig(
        os.path.join(args.save_dir, "avgflow_vs_smape.png"),
        dpi=300
    )
    plt.close()

    print("[INFO] Scatter plots saved")


if __name__ == "__main__":
    main()
