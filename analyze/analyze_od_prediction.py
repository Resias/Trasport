import os
import numpy as np
import pandas as pd
from tqdm import tqdm


PRED_ROOT = "./inference_results_no_mix"
REAL_DATA_DIR = "/home/data/od_minute/test"


# =====================================================
# Utils
# =====================================================
def basic_stats(name, data):
    flat = data.reshape(-1)
    return {
        "name": name,
        "nonzero_ratio": np.count_nonzero(flat) / flat.size,
        "min": flat.min(),
        "max": flat.max(),
        "mean": flat.mean(),
        "median": np.median(flat),
        "p95": np.percentile(flat, 95),
        "p99": np.percentile(flat, 99),
    }


def compare_single_time(pred_od, real_od, tag=""):
    diff = pred_od - real_od
    print(f"\n=== SINGLE TIME ({tag}) ===")
    print(f"Pred nonzero : {np.count_nonzero(pred_od)}")
    print(f"Real nonzero : {np.count_nonzero(real_od)}")
    print(f"MAE          : {np.abs(diff).mean():.2f}")
    print(f"ME           : {diff.mean():.2f}")
    print(f"MaxOver      : {diff.max():.1f}")
    print(f"MaxUnder     : {diff.min():.1f}")


# =====================================================
# Loaders
# =====================================================
def load_pred_day(day_dir):
    minute_dir = os.path.join(day_dir, "minute")
    files = sorted(f for f in os.listdir(minute_dir) if f.endswith(".csv"))
    minute_to_od = {}

    for f in files:
        minute = int(f.split("_")[-1].split(".")[0])
        df = pd.read_csv(os.path.join(minute_dir, f), index_col=0)
        minute_to_od[minute] = df.values

    return minute_to_od  # dict: minute -> (N, N)


def load_real_day(real_path):
    return np.load(real_path)  # (1110 or 1440, N, N)


# =====================================================
# Main
# =====================================================
def main():
    pred_days = sorted(d for d in os.listdir(PRED_ROOT) if d.startswith("day_"))
    real_files = sorted(os.listdir(REAL_DATA_DIR))

    num_days = min(len(pred_days), len(real_files))
    print(f"Total comparable days: {num_days}")

    out_root = os.path.join(PRED_ROOT, "daily_compare")
    os.makedirs(out_root, exist_ok=True)

    for d in range(num_days):
        print(f"\n================ DAY {d:02d} =================")

        pred_day_dir = os.path.join(PRED_ROOT, pred_days[d])
        real_day_path = os.path.join(REAL_DATA_DIR, real_files[d])

        pred_map = load_pred_day(pred_day_dir)
        real_day = load_real_day(real_day_path)

        # ---------- minute alignment ----------
        common_minutes = sorted(
            m for m in pred_map.keys()
            if m < real_day.shape[0]
        )

        if len(common_minutes) == 0:
            print("No overlapping minutes. Skipping.")
            continue

        pred_ods = np.stack([pred_map[m] for m in common_minutes])
        real_ods = np.stack([real_day[m] for m in common_minutes])

        # ---------- minute-level stats ----------
        pred_stats = basic_stats("Pred", pred_ods)
        real_stats = basic_stats("Real", real_ods)

        print("\n--- Minute-level statistics ---")
        for k in pred_stats:
            if k != "name":
                print(
                    f"{k:10s} | "
                    f"Pred={pred_stats[k]:.4f} | "
                    f"Real={real_stats[k]:.4f}"
                )

        # ---------- noon comparison ----------
        NOON_MINUTE = 720  # absolute minute (12:00)

        if NOON_MINUTE in common_minutes:
            idx = common_minutes.index(NOON_MINUTE)
            compare_single_time(
                pred_ods[idx],
                real_ods[idx],
                tag="12:00"
            )

        # ---------- day-level aggregation ----------
        pred_day_sum = pred_ods.sum(axis=0)
        real_day_sum = real_ods.sum(axis=0)
        diff = pred_day_sum - real_day_sum

        print("\n--- Day-level aggregated ---")
        print(f"Day MAE      : {np.abs(diff).mean():.2f}")
        print(f"Day ME       : {diff.mean():.2f}")
        print(f"Day MaxOver  : {diff.max():.1f}")
        print(f"Day MaxUnder : {diff.min():.1f}")

        # ---------- save ----------
        day_out = os.path.join(out_root, f"day_{d:02d}")
        os.makedirs(day_out, exist_ok=True)

        pd.DataFrame(pred_day_sum).to_csv(
            os.path.join(day_out, "pred_day_sum.csv")
        )
        pd.DataFrame(real_day_sum).to_csv(
            os.path.join(day_out, "real_day_sum.csv")
        )
        pd.DataFrame(diff).to_csv(
            os.path.join(day_out, "diff_day_sum.csv")
        )

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()
