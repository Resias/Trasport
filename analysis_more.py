import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# =====================================================
# Config
# =====================================================
ROOT = "./inference_results_no_mix"
OUT_CSV = "od_inflow_outflow_concentration_by_day.csv"

TIME_BINS = {
    "Early Morning": (390, 420),    # 06:30–06:59
    "AM Peak":       (420, 540),    # 07:00–08:59
    "Midday":        (540, 1020),   # 09:00–16:59
    "PM Peak":       (1020, 1140),  # 17:00–18:59
    "Evening":       (1140, 1410),  # 19:00–23:29
}

TOP_K_RATIO = 0.10   # 상위 10%

# =====================================================
# Utils
# =====================================================
def load_real_minutes(day_dir):
    real_minute_dir = os.path.join(day_dir, "real_minute")
    files = sorted(f for f in os.listdir(real_minute_dir) if f.endswith(".csv"))

    minute_to_od = {}
    for f in files:
        minute = int(f.split("_")[-1].split(".")[0])
        arr = pd.read_csv(
            os.path.join(real_minute_dir, f),
            index_col=0
        ).values
        minute_to_od[minute] = arr

    return minute_to_od


def analyze_time_block(data):
    """
    data: (T, N, N)
    """
    T, N, _ = data.shape
    eps = 1e-6

    # ---------- Inflow / Outflow ----------
    inflow = data.sum(axis=2)    # (T, N)
    outflow = data.sum(axis=1)   # (T, N)

    inflow_mean = inflow.mean(axis=0)
    outflow_mean = outflow.mean(axis=0)

    io_ratio = inflow_mean / (outflow_mean + eps)

    # ---------- OD concentration ----------
    flat = data.reshape(-1)
    nonzero_ratio = np.count_nonzero(flat) / flat.size

    sorted_vals = np.sort(flat[flat > 0])[::-1]
    k = max(1, int(len(sorted_vals) * TOP_K_RATIO))
    topk_ratio = sorted_vals[:k].sum() / sorted_vals.sum()

    return {
        "Mean_Inflow": inflow_mean.mean(),
        "Mean_Outflow": outflow_mean.mean(),
        "Mean_IO_Ratio": io_ratio.mean(),
        "OD_Nonzero_Ratio": nonzero_ratio,
        "Top10pct_OD_Ratio": topk_ratio,
    }

# =====================================================
# Main
# =====================================================
def main():
    day_dirs = sorted(d for d in os.listdir(ROOT) if d.startswith("day_"))
    rows = []

    for day_idx, day in enumerate(tqdm(day_dirs, desc="Processing days")):
        day_dir = os.path.join(ROOT, day)
        minute_map = load_real_minutes(day_dir)

        for period, (start, end) in TIME_BINS.items():
            minutes = sorted(
                m for m in minute_map.keys()
                if start <= m < end
            )

            if len(minutes) == 0:
                continue

            data = np.stack([minute_map[m] for m in minutes])

            stats = analyze_time_block(data)
            stats.update({
                "Day": day_idx,
                "Time_Period": period,
                "Start_Time": f"{start//60:02d}:{start%60:02d}",
                "End_Time": f"{(end-1)//60:02d}:{(end-1)%60:02d}",
                "Num_Minutes": len(minutes),
            })

            rows.append(stats)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved per-day time-period OD analysis to: {OUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()
