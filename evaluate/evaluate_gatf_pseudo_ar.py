import argparse
import csv
import json
import os
import sys
import time

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import dense_to_sparse
from tqdm.auto import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from dataset import get_dataset, torch_time_sin_cos
from graph_eval_utils import build_static_edge_index, load_graph_week_model


def _new_state():
    return {
        "sq_error_sum": 0.0,
        "abs_error_sum": 0.0,
        "true_abs_sum": 0.0,
        "smape_term_sum": 0.0,
        "count": 0,
    }


def _update_state(state, pred_full, true_full, eps):
    diff = true_full - pred_full
    abs_diff = torch.abs(diff)
    denom = (torch.abs(true_full) + torch.abs(pred_full)).clamp(min=eps)
    state["sq_error_sum"] += torch.sum(diff ** 2).item()
    state["abs_error_sum"] += torch.sum(abs_diff).item()
    state["true_abs_sum"] += torch.sum(torch.abs(true_full)).item()
    state["smape_term_sum"] += torch.sum(2.0 * abs_diff / denom).item()
    state["count"] += diff.numel()


def _finalize_state(state):
    if state["count"] == 0:
        raise RuntimeError("No predictions were evaluated.")
    mse = state["sq_error_sum"] / state["count"]
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": state["abs_error_sum"] / state["count"],
        "smape": 100.0 * state["smape_term_sum"] / state["count"],
        "wmape": 100.0 * state["abs_error_sum"] / (state["true_abs_sum"] + 1e-8),
    }


def _result_path(study_dir, seed):
    return os.path.join(study_dir, "S8_gate", f"seed_{seed}", "run_result.json")


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _build_graph_from_history(x_hist, static_edge_index):
    del static_edge_index
    batch_size, hist_steps, num_nodes, _ = x_hist.shape
    data_list = []
    for batch_idx in range(batch_size):
        for step_idx in range(hist_steps):
            edge_index, edge_attr = dense_to_sparse(x_hist[batch_idx, step_idx])
            edge_attr = edge_attr.unsqueeze(-1)
            rev_edge_index = edge_index.flip(0)
            edge_index = torch.cat([edge_index, rev_edge_index], dim=1)
            edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
            data_list.append(
                Data(
                    x=torch.zeros(num_nodes, 1),
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                )
            )
    return Batch.from_data_list(data_list)


def _time_features(start_steps, offset, length, time_resolution, device):
    start = torch.as_tensor(start_steps, dtype=torch.long, device=device).view(-1, 1)
    steps = torch.arange(length, device=device).view(1, -1)
    minutes = ((start + offset + steps) * time_resolution) % 1440
    return torch_time_sin_cos(minutes)


def _predict_nar(lightning_module, static_edge_index, x_hist_cpu, labels, start_steps, weekday, time_resolution, device):
    batch_size, hist_steps = x_hist_cpu.shape[:2]
    future_steps = lightning_module.model.future_steps
    batch_graph = _build_graph_from_history(x_hist_cpu, static_edge_index).to(device)
    time_hist = _time_features(start_steps, 0, hist_steps, time_resolution, device)
    time_fut = _time_features(start_steps, hist_steps, future_steps, time_resolution, device)
    with torch.no_grad():
        mag_log, gate_logits = lightning_module.model(
            static_edge_index,
            batch_graph,
            batch_size,
            hist_steps,
            time_hist,
            time_fut,
            weekday,
        )
        mag_log_hard, _ = lightning_module._apply_gate(mag_log, gate_logits)
    pred_full = torch.expm1(torch.clamp(mag_log_hard, min=0.0))
    true_full = torch.expm1(labels)
    return pred_full, true_full


def _predict_ar(
    lightning_module,
    static_edge_index,
    x_hist_cpu,
    labels,
    start_steps,
    weekday,
    time_resolution,
    device,
    eps,
    state,
):
    batch_size, hist_steps = x_hist_cpu.shape[:2]
    future_steps = labels.shape[1]
    model_future_steps = lightning_module.model.future_steps
    rolling_hist = x_hist_cpu.clone()
    for horizon_idx in range(future_steps):
        batch_graph = _build_graph_from_history(rolling_hist, static_edge_index).to(device)
        time_hist = _time_features(start_steps, horizon_idx, hist_steps, time_resolution, device)
        time_fut = _time_features(
            start_steps,
            hist_steps + horizon_idx,
            model_future_steps,
            time_resolution,
            device,
        )
        with torch.no_grad():
            mag_log, gate_logits = lightning_module.model(
                static_edge_index,
                batch_graph,
                batch_size,
                hist_steps,
                time_hist,
                time_fut,
                weekday,
            )
            mag_log_hard, _ = lightning_module._apply_gate(mag_log, gate_logits)

        next_log = mag_log_hard[:, 0].detach()
        pred_step = torch.expm1(torch.clamp(next_log, min=0.0))
        true_step = torch.expm1(labels[:, horizon_idx])
        _update_state(state, pred_step, true_step, eps)

        rolling_hist = torch.cat([rolling_hist[:, 1:], next_log.cpu().unsqueeze(1)], dim=1)


def _iter_batches(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield indices[start : start + batch_size]


def evaluate_seed(args, dataset, seed, device):
    result_path = _result_path(args.study_dir, seed)
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Missing run_result.json: {result_path}")
    result = _load_json(result_path)
    ckpt = result.get("best_checkpoint")
    if not ckpt or not os.path.exists(ckpt):
        raise FileNotFoundError(f"Missing checkpoint for seed {seed}: {ckpt}")

    lightning_module, static_edge_index = load_graph_week_model(
        ckpt,
        device,
        args.od_csv,
        gate_tau=args.gate_tau,
    )
    static_edge_index = static_edge_index.to(device)

    indices = list(range(len(dataset)))
    if args.max_samples > 0:
        indices = indices[: args.max_samples]

    nar_state = _new_state()
    ar_state = _new_state()
    start_time = time.perf_counter()

    for batch_indices in tqdm(
        list(_iter_batches(indices, args.batch_size)),
        desc=f"S8 seed={seed} pseudo-AR",
    ):
        items = [dataset[idx] for idx in batch_indices]
        x_hist_cpu = torch.stack([item["x_tensor"] for item in items]).float()
        labels = torch.stack([item["y_tensor"] for item in items]).float().to(device)
        weekday = torch.stack([item["weekday_tensor"] for item in items]).long().to(device)
        start_steps = [dataset.info_list[idx]["start_step"] for idx in batch_indices]

        nar_pred, nar_true = _predict_nar(
            lightning_module,
            static_edge_index,
            x_hist_cpu,
            labels,
            start_steps,
            weekday,
            args.time_resolution,
            device,
        )
        _update_state(nar_state, nar_pred, nar_true, args.mape_eps)

        _predict_ar(
            lightning_module,
            static_edge_index,
            x_hist_cpu,
            labels,
            start_steps,
            weekday,
            args.time_resolution,
            device,
            args.mape_eps,
            ar_state,
        )

    nar_metrics = _finalize_state(nar_state)
    ar_metrics = _finalize_state(ar_state)
    elapsed = time.perf_counter() - start_time

    row = {
        "seed": seed,
        "checkpoint": ckpt,
        "evaluated_samples": len(indices),
        "batch_size": args.batch_size,
        "elapsed_sec": elapsed,
    }
    for prefix, metrics in [("nar", nar_metrics), ("pseudo_ar", ar_metrics)]:
        for key, value in metrics.items():
            row[f"{prefix}_{key}"] = value
    return row


def _write_outputs(rows, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    per_seed_csv = os.path.join(output_dir, "pseudo_ar_per_seed.csv")
    if rows:
        with open(per_seed_csv, "w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary = {}
    if rows:
        metric_cols = [key for key in rows[0] if key.startswith("nar_") or key.startswith("pseudo_ar_")]
        summary = {"runs": len(rows)}
        for col in metric_cols:
            values = torch.tensor([float(row[col]) for row in rows], dtype=torch.float64)
            summary[f"{col}_mean"] = float(values.mean())
            summary[f"{col}_std"] = float(values.std(unbiased=True)) if len(values) > 1 else 0.0

    output_json = os.path.join(output_dir, "pseudo_ar_summary.json")
    with open(output_json, "w", encoding="utf-8") as fp:
        json.dump({"rows": rows, "summary": summary}, fp, indent=2)
    return per_seed_csv, output_json


def parse_args():
    parser = argparse.ArgumentParser(description="Checkpoint-based pseudo-AR evaluation for S8/Metro-GATF.")
    parser.add_argument("--study_dir", default="/root/tmp/Trasport/review_runs/review_3way_50ep_20260512")
    parser.add_argument("--data_root", default="/root/tmp/Trasport/data_splits/od_minute_review_3way")
    parser.add_argument("--test_subdir", default="test")
    parser.add_argument("--od_csv", default=os.path.join(ROOT, "AD_matrix_trimmed_common.csv"))
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=10)
    parser.add_argument("--time_resolution", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--gate_tau", type=float, default=0.9)
    parser.add_argument("--mape_eps", type=float, default=1e-3)
    parser.add_argument("--max_samples", type=int, default=0, help="0 evaluates the full test split.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = args.output_dir or os.path.join(args.study_dir, "evaluation", "pseudo_ar")
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]

    _, dataset = get_dataset(
        data_root=args.data_root,
        train_subdir=args.test_subdir,
        val_subdir=args.test_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
        time_resolution=args.time_resolution,
        cache_in_mem=False,
    )

    rows = []
    for seed in seeds:
        rows.append(evaluate_seed(args, dataset, seed, device))
        per_seed_csv, output_json = _write_outputs(rows, output_dir)
        print(f"[Saved] {per_seed_csv}")
        print(f"[Saved] {output_json}")

    print(json.dumps({"rows": rows}, indent=2))


if __name__ == "__main__":
    main()
