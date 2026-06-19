import argparse
import glob
import json
import os
import re
import sys

import pandas as pd
import torch
from tqdm.auto import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from dataset import get_dataset
from graph_eval_utils import (
    _build_graph_week_loader,
    load_graph_week_model,
)


DEFAULT_CKPT_GLOB = os.path.join(
    ROOT,
    "ablation_runs",
    "progressive_core_v1",
    "S8_gate",
    "seed_*",
    "checkpoints",
    "best-*.ckpt",
)


def _parse_thresholds(raw):
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("At least one threshold must be provided.")
    return sorted(set(values))


def _infer_seed(path):
    match = re.search(r"seed_(\d+)", path)
    return int(match.group(1)) if match else None


def _find_checkpoints(pattern, max_seeds):
    ckpts = sorted(glob.glob(pattern), key=lambda path: (_infer_seed(path) is None, _infer_seed(path), path))
    if max_seeds is not None:
        ckpts = ckpts[:max_seeds]
    if len(ckpts) < 3:
        raise ValueError(f"Need at least 3 checkpoints for seed aggregation, found {len(ckpts)}.")
    return ckpts


def _new_metric_state():
    return {
        "sq_error_sum": 0.0,
        "abs_error_sum": 0.0,
        "true_abs_sum": 0.0,
        "smape_term_sum": 0.0,
        "count": 0,
        "pred_nonzero_count": 0,
        "true_nonzero_count": 0,
    }


def _update_state(state, pred_full, true_full, pred_mask, true_mask, eps):
    diff = true_full - pred_full
    abs_diff = torch.abs(diff)
    denom = (torch.abs(true_full) + torch.abs(pred_full)).clamp(min=eps)

    state["sq_error_sum"] += torch.sum(diff ** 2).item()
    state["abs_error_sum"] += torch.sum(abs_diff).item()
    state["true_abs_sum"] += torch.sum(torch.abs(true_full)).item()
    state["smape_term_sum"] += torch.sum(2.0 * abs_diff / denom).item()
    state["count"] += diff.numel()
    state["pred_nonzero_count"] += pred_mask.sum().item()
    state["true_nonzero_count"] += true_mask.sum().item()


def _finalize_state(state):
    if state["count"] == 0:
        raise ValueError("No evaluation samples were processed.")

    mse = state["sq_error_sum"] / state["count"]
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": state["abs_error_sum"] / state["count"],
        "smape": 100.0 * state["smape_term_sum"] / state["count"],
        "wmape": 100.0 * state["abs_error_sum"] / (state["true_abs_sum"] + 1e-8),
        "pred_nonzero_rate": state["pred_nonzero_count"] / state["count"],
        "true_nonzero_rate": state["true_nonzero_count"] / state["count"],
    }


def evaluate_checkpoint_thresholds(
    ckpt_path,
    dataset,
    device,
    od_csv,
    thresholds,
    batch_size,
    num_workers,
    mape_eps,
    progress,
):
    lightning_module, static_edge_index = load_graph_week_model(
        ckpt_path,
        device,
        od_csv,
        gate_tau=thresholds[0],
    )
    loader = _build_graph_week_loader(dataset, static_edge_index, batch_size, num_workers)
    states = {tau: _new_metric_state() for tau in thresholds}

    iterator = loader
    if progress:
        seed = _infer_seed(ckpt_path)
        desc = f"seed {seed}" if seed is not None else os.path.basename(ckpt_path)
        iterator = tqdm(loader, desc=desc, total=len(loader))

    static_edge_index = static_edge_index.to(device)
    with torch.no_grad():
        for batch in iterator:
            _, batch_graph, batch_size_actual, hist_steps, labels, time_hist, time_fut, weekday = batch
            batch_graph = batch_graph.to(device)
            labels = labels.to(device)
            time_hist = time_hist.to(device)
            time_fut = time_fut.to(device)
            weekday = weekday.to(device)

            mag_log, gate_logits = lightning_module.model(
                static_edge_index,
                batch_graph,
                batch_size_actual,
                hist_steps,
                time_hist,
                time_fut,
                weekday,
            )
            mag_full = torch.expm1(torch.clamp(mag_log, min=0.0))
            true_full = torch.expm1(labels)
            gate_prob = torch.sigmoid(gate_logits)
            true_mask = labels > 0

            for tau in thresholds:
                pred_mask = gate_prob > tau
                pred_full = mag_full * pred_mask
                _update_state(states[tau], pred_full, true_full, pred_mask, true_mask, mape_eps)

    return {tau: _finalize_state(state) for tau, state in states.items()}


def _summarize(seed_rows):
    df = pd.DataFrame(seed_rows)
    metric_cols = [
        "mse",
        "rmse",
        "mae",
        "smape",
        "wmape",
        "pred_nonzero_rate",
        "true_nonzero_rate",
    ]

    summary_rows = []
    for tau, group in df.groupby("gate_tau", sort=True):
        row = {
            "gate_tau": tau,
            "num_seeds": int(group["seed"].nunique()),
            "seeds": ",".join(str(int(seed)) for seed in sorted(group["seed"].dropna().unique())),
        }
        for metric in metric_cols:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
        summary_rows.append(row)

    return df.sort_values(["gate_tau", "seed"]), pd.DataFrame(summary_rows)


def _print_summary(summary_df):
    print("\nGate Threshold Sensitivity")
    print("=" * 112)
    print(
        f"{'tau':>5} | {'RMSE':>18} | {'MAE':>18} | {'SMAPE':>18} | "
        f"{'WMAPE':>18} | {'Pred NZ':>18}"
    )
    print("-" * 112)
    for _, row in summary_df.iterrows():
        print(
            f"{row['gate_tau']:>5.2f} | "
            f"{row['rmse_mean']:>8.4f} +- {row['rmse_std']:<7.4f} | "
            f"{row['mae_mean']:>8.4f} +- {row['mae_std']:<7.4f} | "
            f"{row['smape_mean']:>8.4f} +- {row['smape_std']:<7.4f} | "
            f"{row['wmape_mean']:>8.4f} +- {row['wmape_std']:<7.4f} | "
            f"{100.0 * row['pred_nonzero_rate_mean']:>7.3f}% +- {100.0 * row['pred_nonzero_rate_std']:<6.3f}%"
        )
    print("=" * 112)


def parse_args():
    parser = argparse.ArgumentParser(description="GATransformer gate threshold sensitivity analysis.")
    parser.add_argument("--ckpt_glob", default=DEFAULT_CKPT_GLOB)
    parser.add_argument("--max_seeds", type=int, default=5)
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9")
    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--test_subdir", default="test")
    parser.add_argument("--od_csv", default=os.path.join(ROOT, "AD_matrix_trimmed_common.csv"))
    parser.add_argument("--time_resolution", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mape_eps", type=float, default=1e-3)
    parser.add_argument("--device", default="")
    parser.add_argument("--output_dir", default=os.path.join(ROOT, "evaluate", "outputs", "gate_sensitivity"))
    parser.add_argument("--no_progress", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    thresholds = _parse_thresholds(args.thresholds)
    ckpts = _find_checkpoints(args.ckpt_glob, args.max_seeds)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Thresholds: {thresholds}")
    print("Checkpoints:")
    for ckpt in ckpts:
        print(f"  seed={_infer_seed(ckpt)} path={ckpt}")

    _, eval_dataset = get_dataset(
        data_root=args.data_root,
        train_subdir=args.test_subdir,
        val_subdir=args.test_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
        time_resolution=args.time_resolution,
    )

    seed_rows = []
    for ckpt in ckpts:
        seed = _infer_seed(ckpt)
        metrics_by_tau = evaluate_checkpoint_thresholds(
            ckpt_path=ckpt,
            dataset=eval_dataset,
            device=device,
            od_csv=args.od_csv,
            thresholds=thresholds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mape_eps=args.mape_eps,
            progress=not args.no_progress,
        )
        for tau, metrics in metrics_by_tau.items():
            seed_rows.append({"seed": seed, "checkpoint": ckpt, "gate_tau": tau, **metrics})

    per_seed_df, summary_df = _summarize(seed_rows)
    os.makedirs(args.output_dir, exist_ok=True)
    per_seed_path = os.path.join(args.output_dir, "per_seed.csv")
    summary_path = os.path.join(args.output_dir, "summary.csv")
    json_path = os.path.join(args.output_dir, "summary.json")

    per_seed_df.to_csv(per_seed_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(
            {
                "thresholds": thresholds,
                "checkpoints": ckpts,
                "per_seed": per_seed_df.to_dict(orient="records"),
                "summary": summary_df.to_dict(orient="records"),
            },
            fp,
            indent=2,
        )

    _print_summary(summary_df)
    print(f"\nSaved per-seed metrics: {per_seed_path}")
    print(f"Saved summary metrics : {summary_path}")
    print(f"Saved JSON summary    : {json_path}")


if __name__ == "__main__":
    main()
