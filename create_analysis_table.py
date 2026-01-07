import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# =====================================================
# Config
# =====================================================
ROOT = "./inference_results_no_mix"
OUT_CSV = "od_hourly_sum_analysis_by_day.csv"

TIME_BINS = {
    "Early Morning": (390, 420),    # 06:30–06:59
    "AM Peak":       (420, 540),    # 07:00–08:59
    "Midday":        (540, 1020),   # 09:00–16:59
    "PM Peak":       (1020, 1140),  # 17:00–18:59
    "Evening":       (1140, 1410),  # 19:00–23:29
}

TOP_K_RATIO = 0.10
EPS = 1e-6

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


def aggregate_to_hourly(minute_map, start, end):
    """
    분 단위 OD → 시간 단위 OD 합산
    return: dict {hour_index: (N, N)}
    """
    hourly = {}
    for m, od in minute_map.items():
        if start <= m < end:
            hour = m // 60
            if hour not in hourly:
                hourly[hour] = od.copy()
            else:
                hourly[hour] += od
    return hourly


def gini_coefficient(x):
    """불균형/집중도 측정"""
    if np.all(x == 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    cumx = np.cumsum(x)
    return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


def analyze_hourly_od(od):
    """
    od: (N, N) 시간당 합산 OD
    """
    N = od.shape[0]

    inflow = od.sum(axis=0)
    outflow = od.sum(axis=1)
    total_flow = od.sum()

    nonzero_ratio = np.count_nonzero(od) / od.size

    flat = od.flatten()
    positive = flat[flat > 0]
    if len(positive) == 0:
        topk_ratio = 0.0
    else:
        k = max(1, int(len(positive) * TOP_K_RATIO))
        topk_ratio = np.sort(positive)[-k:].sum() / positive.sum()

    inflow_gini = gini_coefficient(inflow)
    outflow_gini = gini_coefficient(outflow)

    hub_inflow_ratio = inflow.max() / (inflow.sum() + EPS)
    hub_outflow_ratio = outflow.max() / (outflow.sum() + EPS)

    self_loop_ratio = np.trace(od) / (total_flow + EPS)

    io_balance = np.mean(inflow / (outflow + EPS))

    return {
        "Total_Flow": total_flow,
        "Mean_Inflow": inflow.mean(),
        "Mean_Outflow": outflow.mean(),
        "IO_Balance": io_balance,
        "OD_Nonzero_Ratio": nonzero_ratio,
        "Top10pct_OD_Ratio": topk_ratio,
        "Inflow_Gini": inflow_gini,
        "Outflow_Gini": outflow_gini,
        "Hub_Inflow_Ratio": hub_inflow_ratio,
        "Hub_Outflow_Ratio": hub_outflow_ratio,
        "Self_Loop_Ratio": self_loop_ratio,
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
            hourly_map = aggregate_to_hourly(minute_map, start, end)

            for hour, od in hourly_map.items():
                stats = analyze_hourly_od(od)

                # 🔥 메타정보 먼저
                row = {
                    "Day": day_idx,
                    "Time_Period": period,
                    "Hour": hour,
                    "Hour_Label": f"{hour:02d}:00–{hour:02d}:59",
                }

                # 🔥 분석 결과를 뒤에 붙임
                row.update(stats)

                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print(f"\nSaved hourly-sum OD analysis to: {OUT_CSV}")
    print(df.head())


if __name__ == "__main__":
    main()
