import argparse
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPS = 1e-6

TIME_BINS = [
    ("Early Morning", 390, 420),
    ("AM Peak", 420, 540),
    ("Midday", 540, 1020),
    ("PM Peak", 1020, 1140),
    ("Evening", 1140, 1410),
]

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class MetricAccumulator:
    n: int = 0
    abs_sum: float = 0.0
    sq_sum: float = 0.0
    smape_sum: float = 0.0
    true_sum: float = 0.0
    pred_sum: float = 0.0
    bias_sum: float = 0.0
    true_nonzero: int = 0
    pred_nonzero: int = 0
    tp_nonzero: int = 0
    fp_nonzero: int = 0
    fn_nonzero: int = 0

    def update(self, pred: np.ndarray, true: np.ndarray) -> None:
        diff = pred - true
        abs_diff = np.abs(diff)
        denom = np.abs(pred) + np.abs(true) + EPS

        true_nz = true > 0
        pred_nz = pred > 0

        self.n += diff.size
        self.abs_sum += float(abs_diff.sum())
        self.sq_sum += float(np.square(diff).sum())
        self.smape_sum += float((2.0 * abs_diff / denom).sum())
        self.true_sum += float(true.sum())
        self.pred_sum += float(pred.sum())
        self.bias_sum += float(diff.sum())
        self.true_nonzero += int(true_nz.sum())
        self.pred_nonzero += int(pred_nz.sum())
        self.tp_nonzero += int((true_nz & pred_nz).sum())
        self.fp_nonzero += int((~true_nz & pred_nz).sum())
        self.fn_nonzero += int((true_nz & ~pred_nz).sum())

    def to_dict(self) -> dict:
        precision = self.tp_nonzero / (self.tp_nonzero + self.fp_nonzero + EPS)
        recall = self.tp_nonzero / (self.tp_nonzero + self.fn_nonzero + EPS)
        return {
            "count": self.n,
            "MAE": self.abs_sum / max(self.n, 1),
            "RMSE": math.sqrt(self.sq_sum / max(self.n, 1)),
            "sMAPE": 100.0 * self.smape_sum / max(self.n, 1),
            "WAPE": 100.0 * self.abs_sum / (self.true_sum + EPS),
            "Bias": self.bias_sum / max(self.n, 1),
            "True_Total": self.true_sum,
            "Pred_Total": self.pred_sum,
            "True_Nonzero_Ratio": self.true_nonzero / max(self.n, 1),
            "Pred_Nonzero_Ratio": self.pred_nonzero / max(self.n, 1),
            "Nonzero_Precision": precision,
            "Nonzero_Recall": recall,
        }


def minute_from_file(path: Path) -> int:
    match = re.search(r"od_minute_(\d+)\.csv$", path.name)
    if not match:
        raise ValueError(f"Cannot parse minute from {path}")
    return int(match.group(1))


def day_index_from_dir(path: Path) -> int:
    match = re.search(r"day_(\d+)$", path.name)
    if not match:
        raise ValueError(f"Cannot parse day index from {path}")
    return int(match.group(1))


def time_period(minute: int) -> str:
    for label, start, end in TIME_BINS:
        if start <= minute < end:
            return label
    return "Other"


def read_matrix(path: Path) -> np.ndarray:
    return pd.read_csv(path, index_col=0).to_numpy(dtype=np.float64)


def load_station_names(adj_csv: Path) -> list[str]:
    return list(pd.read_csv(adj_csv, index_col=0).index)


def safe_log10(values: pd.Series) -> np.ndarray:
    return np.log10(np.maximum(values.to_numpy(dtype=float), EPS))


def save_scatter(df: pd.DataFrame, x: str, y: str, path: Path, ylabel: str) -> None:
    plt.figure(figsize=(7.2, 4.8))
    plt.scatter(safe_log10(df[x]), df[y], s=16, alpha=0.68, edgecolors="none")
    plt.xlabel(f"log10({x})")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25, linewidth=0.6)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_hourly_plot(hourly_df: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    by_hour = hourly_df.sort_values("Hour")
    ax1.plot(by_hour["Hour"], by_hour["MAE"], marker="o", label="MAE")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("MAE")
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax2 = ax1.twinx()
    ax2.plot(by_hour["Hour"], by_hour["sMAPE"], marker="s", color="#c75c3c", label="sMAPE")
    ax2.set_ylabel("sMAPE (%)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def station_group_summary(station_df: pd.DataFrame) -> pd.DataFrame:
    q20 = station_df["Demand"].quantile(0.20)
    q80 = station_df["Demand"].quantile(0.80)

    def group(demand: float) -> str:
        if demand >= q80:
            return "High demand (top 20%)"
        if demand <= q20:
            return "Low demand (bottom 20%)"
        return "Middle demand (middle 60%)"

    grouped = station_df.assign(Demand_Group=station_df["Demand"].map(group))
    order = ["High demand (top 20%)", "Middle demand (middle 60%)", "Low demand (bottom 20%)"]
    summary = (
        grouped.groupby("Demand_Group", observed=False)
        .agg(
            Stations=("station_name", "count"),
            Demand=("Demand", "mean"),
            MAE=("MAE", "mean"),
            RMSE=("RMSE", "mean"),
            sMAPE=("sMAPE", "mean"),
            nMAE=("nMAE", "mean"),
            Bias=("Bias", "mean"),
        )
        .reindex(order)
        .reset_index()
    )
    return summary


def top_pair_table(pair_df: pd.DataFrame, station_names: list[str]) -> pd.DataFrame:
    top = pair_df.sort_values("Demand", ascending=False).head(30).copy()
    top["origin"] = top["origin_idx"].map(lambda i: station_names[int(i)])
    top["destination"] = top["dest_idx"].map(lambda i: station_names[int(i)])
    cols = ["origin", "destination", "Demand", "MAE", "RMSE", "sMAPE", "nMAE", "Bias"]
    return top[cols]


def write_report(
    out_dir: Path,
    overall: dict,
    period_df: pd.DataFrame,
    day_df: pd.DataFrame,
    hour_df: pd.DataFrame,
    station_df: pd.DataFrame,
    group_df: pd.DataFrame,
    high_mae: pd.DataFrame,
    high_demand_stable: pd.DataFrame,
    relative_vulnerable: pd.DataFrame,
    top_pairs: pd.DataFrame,
    source_info: dict,
) -> None:
    def fmt(value, digits=4):
        if isinstance(value, (int, np.integer)):
            return str(value)
        if pd.isna(value):
            return ""
        return f"{float(value):,.{digits}f}"

    def df_to_markdown(df: pd.DataFrame, digits=4) -> str:
        headers = list(df.columns)
        rows = []
        for record in df.to_dict(orient="records"):
            row = []
            for col in headers:
                value = record[col]
                if isinstance(value, (float, np.floating)):
                    row.append(fmt(value, digits))
                elif isinstance(value, (int, np.integer)):
                    row.append(str(value))
                elif pd.isna(value):
                    row.append("")
                else:
                    row.append(str(value))
            rows.append(row)

        lines = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)

    report_path = out_dir / "analysis_report.md"
    lines = []
    lines.append("# Final Station/OD Analysis Report")
    lines.append("")
    lines.append("## Source")
    for key, value in source_info.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("## Overall OD Matrix Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for key in [
        "MAE",
        "RMSE",
        "sMAPE",
        "WAPE",
        "Bias",
        "True_Total",
        "Pred_Total",
        "True_Nonzero_Ratio",
        "Pred_Nonzero_Ratio",
        "Nonzero_Precision",
        "Nonzero_Recall",
    ]:
        digits = 6 if "Ratio" in key or "Precision" in key or "Recall" in key else 4
        lines.append(f"| {key} | {fmt(overall[key], digits)} |")

    def table(title: str, df: pd.DataFrame, digits=4):
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        lines.append(df_to_markdown(df, digits))

    table("Time Period Metrics", period_df, 4)
    table("Daily Metrics", day_df, 4)
    table("Hourly Metrics", hour_df, 4)
    table("Demand Group Summary", group_df, 4)
    table("High MAE Stations", high_mae, 4)
    table("High-demand Stable Stations", high_demand_stable, 4)
    table("High-demand Relative Vulnerable Stations", relative_vulnerable, 4)
    table("Top-demand OD Pairs", top_pairs, 4)
    lines.append("")
    lines.append("## Notes")
    lines.append("- Station demand is computed as origin sum + destination sum - self-loop diagonal.")
    lines.append("- Current inference files do not include checkpoint metadata; interpret station-level results as metrics for the saved `inference_results_no_mix` predictions.")
    lines.append("- Model comparison metrics are kept separately in the existing review run outputs.")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default=str(REPO_ROOT / "inference_results_no_mix"))
    parser.add_argument("--adj-csv", default=str(REPO_ROOT / "AD_matrix_trimmed_common.csv"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "final_station_analysis"))
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--save-pair-metrics", action="store_true")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    adj_csv = Path(args.adj_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    station_names = load_station_names(adj_csv)
    num_stations = len(station_names)
    pair_count = num_stations * num_stations

    overall_acc = MetricAccumulator()
    period_acc = defaultdict(MetricAccumulator)
    day_acc = defaultdict(MetricAccumulator)
    hour_acc = defaultdict(MetricAccumulator)

    station_abs = np.zeros(num_stations, dtype=np.float64)
    station_sq = np.zeros(num_stations, dtype=np.float64)
    station_smape = np.zeros(num_stations, dtype=np.float64)
    station_true = np.zeros(num_stations, dtype=np.float64)
    station_pred = np.zeros(num_stations, dtype=np.float64)
    station_bias = np.zeros(num_stations, dtype=np.float64)
    station_count = 0

    pair_abs = np.zeros((num_stations, num_stations), dtype=np.float64)
    pair_sq = np.zeros((num_stations, num_stations), dtype=np.float64)
    pair_smape = np.zeros((num_stations, num_stations), dtype=np.float64)
    pair_true = np.zeros((num_stations, num_stations), dtype=np.float64)
    pair_pred = np.zeros((num_stations, num_stations), dtype=np.float64)
    pair_bias = np.zeros((num_stations, num_stations), dtype=np.float64)

    day_dirs = sorted(p for p in result_root.iterdir() if p.is_dir() and p.name.startswith("day_"))
    if args.max_days is not None:
        day_dirs = day_dirs[: args.max_days]

    processed_minutes = 0
    source_days = []

    for day_dir in day_dirs:
        day_idx = day_index_from_dir(day_dir)
        source_days.append(day_idx)
        pred_dir = day_dir / "minute"
        real_dir = day_dir / "real_minute"
        pred_files = sorted(pred_dir.glob("od_minute_*.csv"))
        total_files = len(pred_files)

        for file_idx, pred_file in enumerate(pred_files, start=1):
            minute = minute_from_file(pred_file)
            real_file = real_dir / pred_file.name
            if not real_file.exists():
                continue

            pred = read_matrix(pred_file)
            true = read_matrix(real_file)
            if pred.shape != (num_stations, num_stations) or true.shape != (num_stations, num_stations):
                raise ValueError(f"Unexpected shape at {pred_file}: pred={pred.shape}, true={true.shape}")

            diff = pred - true
            abs_diff = np.abs(diff)
            sq_diff = np.square(diff)
            smape_values = 2.0 * abs_diff / (np.abs(pred) + np.abs(true) + EPS)

            overall_acc.update(pred, true)
            period_acc[time_period(minute)].update(pred, true)
            day_acc[f"day_{day_idx:02d}"].update(pred, true)
            hour_acc[minute // 60].update(pred, true)

            pair_abs += abs_diff
            pair_sq += sq_diff
            pair_smape += smape_values
            pair_true += true
            pair_pred += pred
            pair_bias += diff

            pred_station = pred.sum(axis=1) + pred.sum(axis=0) - np.diag(pred)
            true_station = true.sum(axis=1) + true.sum(axis=0) - np.diag(true)
            station_diff = pred_station - true_station
            station_abs += np.abs(station_diff)
            station_sq += np.square(station_diff)
            station_smape += 2.0 * np.abs(station_diff) / (np.abs(pred_station) + np.abs(true_station) + EPS)
            station_true += true_station
            station_pred += pred_station
            station_bias += station_diff
            station_count += 1
            processed_minutes += 1

            if file_idx == 1 or file_idx % 120 == 0 or file_idx == total_files:
                print(f"[INFO] {day_dir.name}: processed {file_idx}/{total_files} minute files", flush=True)

    if processed_minutes == 0:
        raise RuntimeError("No comparable minute files were processed.")

    overall = overall_acc.to_dict()

    period_rows = []
    for label, _, _ in TIME_BINS:
        if label in period_acc:
            row = {"Time_Period": label}
            row.update(period_acc[label].to_dict())
            period_rows.append(row)
    period_df = pd.DataFrame(period_rows)

    day_rows = []
    for key in sorted(day_acc):
        row = {"Day": key}
        row.update(day_acc[key].to_dict())
        day_rows.append(row)
    day_df = pd.DataFrame(day_rows)

    hour_rows = []
    for hour in sorted(hour_acc):
        row = {"Hour": hour, "Hour_Label": f"{hour:02d}:00-{hour:02d}:59"}
        row.update(hour_acc[hour].to_dict())
        hour_rows.append(row)
    hour_df = pd.DataFrame(hour_rows)

    station_df = pd.DataFrame(
        {
            "station_id": np.arange(num_stations),
            "station_name": station_names,
            "Demand": station_true / station_count,
            "Pred_Demand": station_pred / station_count,
            "MAE": station_abs / station_count,
            "RMSE": np.sqrt(station_sq / station_count),
            "sMAPE": 100.0 * station_smape / station_count,
            "nMAE": (station_abs / station_count) / (station_true / station_count + EPS),
            "Bias": station_bias / station_count,
        }
    )

    pair_df = pd.DataFrame(
        {
            "origin_idx": np.repeat(np.arange(num_stations), num_stations),
            "dest_idx": np.tile(np.arange(num_stations), num_stations),
            "Demand": (pair_true / processed_minutes).reshape(pair_count),
            "Pred_Demand": (pair_pred / processed_minutes).reshape(pair_count),
            "MAE": (pair_abs / processed_minutes).reshape(pair_count),
            "RMSE": np.sqrt(pair_sq / processed_minutes).reshape(pair_count),
            "sMAPE": (100.0 * pair_smape / processed_minutes).reshape(pair_count),
            "nMAE": ((pair_abs / processed_minutes) / (pair_true / processed_minutes + EPS)).reshape(pair_count),
            "Bias": (pair_bias / processed_minutes).reshape(pair_count),
        }
    )

    station_df.to_csv(out_dir / "station_metrics_all.csv", index=False)
    period_df.to_csv(out_dir / "time_period_metrics.csv", index=False)
    day_df.to_csv(out_dir / "daily_metrics.csv", index=False)
    hour_df.to_csv(out_dir / "hourly_metrics.csv", index=False)

    if args.save_pair_metrics:
        pair_df.to_csv(out_dir / "od_pair_metrics_all.csv", index=False)

    nonself_pair_df = pair_df[pair_df["origin_idx"] != pair_df["dest_idx"]].copy()
    top_pairs = top_pair_table(nonself_pair_df, station_names)
    top_pairs.to_csv(out_dir / "top_demand_od_pairs.csv", index=False)

    group_df = station_group_summary(station_df)
    group_df.to_csv(out_dir / "station_demand_group_summary.csv", index=False)

    high_mae = station_df.sort_values("MAE", ascending=False).head(10)
    demand_cut = station_df["Demand"].quantile(0.80)
    high_demand = station_df[station_df["Demand"] >= demand_cut].copy()
    high_demand_stable = high_demand.sort_values(["sMAPE", "nMAE"], ascending=True).head(10)
    relative_vulnerable = high_demand.sort_values(["sMAPE", "nMAE"], ascending=False).head(10)

    selected_cols = ["station_name", "Demand", "MAE", "RMSE", "sMAPE", "nMAE", "Bias"]
    high_mae[selected_cols].to_csv(out_dir / "high_mae_stations.csv", index=False)
    high_demand_stable[selected_cols].to_csv(out_dir / "high_demand_stable_stations.csv", index=False)
    relative_vulnerable[selected_cols].to_csv(out_dir / "high_demand_relative_vulnerable_stations.csv", index=False)

    save_scatter(station_df, "Demand", "MAE", out_dir / "station_demand_vs_mae.png", "MAE")
    save_scatter(station_df, "Demand", "sMAPE", out_dir / "station_demand_vs_smape.png", "sMAPE (%)")
    save_scatter(station_df, "Demand", "nMAE", out_dir / "station_demand_vs_nmae.png", "nMAE")
    save_hourly_plot(hour_df, out_dir / "hourly_mae_smape_profile.png")

    source_info = {
        "result_root": str(result_root),
        "adj_csv": str(adj_csv),
        "days_processed": ",".join(f"day_{d:02d}" for d in source_days),
        "minutes_processed": processed_minutes,
        "station_count": num_stations,
        "time_coverage": "06:30-23:29",
    }
    (out_dir / "overall_metrics.json").write_text(
        json.dumps({"source": source_info, "overall": overall}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    write_report(
        out_dir=out_dir,
        overall=overall,
        period_df=period_df[
            [
                "Time_Period",
                "MAE",
                "RMSE",
                "sMAPE",
                "WAPE",
                "Bias",
                "True_Nonzero_Ratio",
                "Pred_Nonzero_Ratio",
                "Nonzero_Precision",
                "Nonzero_Recall",
            ]
        ],
        day_df=day_df[["Day", "MAE", "RMSE", "sMAPE", "WAPE", "Bias", "True_Total", "Pred_Total"]],
        hour_df=hour_df[["Hour_Label", "MAE", "RMSE", "sMAPE", "WAPE", "Bias", "True_Nonzero_Ratio"]],
        station_df=station_df,
        group_df=group_df,
        high_mae=high_mae[selected_cols],
        high_demand_stable=high_demand_stable[selected_cols],
        relative_vulnerable=relative_vulnerable[selected_cols],
        top_pairs=top_pairs,
        source_info=source_info,
    )

    print(f"[INFO] Analysis complete. Outputs written to {out_dir}", flush=True)
    print(json.dumps(overall, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
