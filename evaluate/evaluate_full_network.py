import argparse
import json
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from dataset import get_dataset, get_mpgcn_dataset
from graph_eval_utils import (
    _build_graph_week_loader,
    _run_graph_week_batch,
    load_graph_week_model,
)
from mpgcn_eval_utils import (
    _build_mpgcn_loader,
    _run_mpgcn_batch,
    load_mpgcn_model,
)


def _new_metric_state():
    return {
        "sq_error_sum": 0.0,
        "abs_error_sum": 0.0,
        "true_abs_sum": 0.0,
        "smape_term_sum": 0.0,
        "count": 0,
    }


def _update_metric_state(state, pred_full, true_full, eps):
    diff = true_full - pred_full
    abs_diff = torch.abs(diff)
    denom = (torch.abs(true_full) + torch.abs(pred_full)).clamp(min=eps)

    state["sq_error_sum"] += torch.sum(diff ** 2).item()
    state["abs_error_sum"] += torch.sum(abs_diff).item()
    state["true_abs_sum"] += torch.sum(torch.abs(true_full)).item()
    state["smape_term_sum"] += torch.sum(2.0 * abs_diff / denom).item()
    state["count"] += diff.numel()


def _finalize_metric_state(state):
    if state["count"] == 0:
        raise ValueError("No evaluation samples were processed.")

    mse = state["sq_error_sum"] / state["count"]
    rmse = mse ** 0.5
    mae = state["abs_error_sum"] / state["count"]
    smape = 100.0 * (state["smape_term_sum"] / state["count"])
    wmape = 100.0 * (state["abs_error_sum"] / (state["true_abs_sum"] + 1e-8))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "smape": smape,
        "wmape": wmape,
    }


def evaluate_train_graph_full(
    ckpt_path,
    dataset,
    device,
    od_csv,
    gate_tau,
    batch_size,
    num_workers,
    mape_eps,
):
    lightning_module, static_edge_index = load_graph_week_model(
        ckpt_path,
        device,
        od_csv,
        gate_tau=gate_tau,
    )
    loader = _build_graph_week_loader(dataset, static_edge_index, batch_size, num_workers)
    state = _new_metric_state()

    for batch in loader:
        pred_full, true_full = _run_graph_week_batch(lightning_module, batch, device)
        _update_metric_state(state, pred_full, true_full, eps=mape_eps)

    return _finalize_metric_state(state)


def evaluate_mpgcn_full(
    ckpt_path,
    train_dataset,
    eval_dataset,
    device,
    od_csv,
    dynamic_graph_cache,
    dynamic_bin_size,
    kernel_type,
    cheby_order,
    batch_size,
    num_workers,
    mape_eps,
):
    lightning_module = load_mpgcn_model(
        ckpt_path=ckpt_path,
        device=device,
        train_dataset=train_dataset,
        od_csv=od_csv,
        dynamic_graph_cache=dynamic_graph_cache,
        dynamic_bin_size=dynamic_bin_size,
        kernel_type=kernel_type,
        cheby_order=cheby_order,
    )
    loader = _build_mpgcn_loader(eval_dataset, batch_size, num_workers)
    state = _new_metric_state()

    for batch in loader:
        pred_full, true_full = _run_mpgcn_batch(lightning_module, batch, device)
        _update_metric_state(state, pred_full, true_full, eps=mape_eps)

    return _finalize_metric_state(state)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-network evaluation for TrainGraph and MPGCN checkpoints."
    )
    parser.add_argument("--gnn_ckpt", default="")
    parser.add_argument("--mpgcn_ckpt", default="")

    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--train_subdir", default="train")
    parser.add_argument("--test_subdir", default="test")
    parser.add_argument("--od_csv", default=os.path.join(ROOT, "AD_matrix_trimmed_common.csv"))
    parser.add_argument("--time_resolution", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=10)
    parser.add_argument("--graph_gate_tau", type=float, default=0.6)
    parser.add_argument("--mape_eps", type=float, default=1e-3)

    parser.add_argument("--mpgcn_dynamic_bin_size", type=int, default=60)
    parser.add_argument("--mpgcn_dynamic_graph_cache", default="")
    parser.add_argument(
        "--mpgcn_kernel_type",
        choices=[
            "chebyshev",
            "localpool",
            "random_walk_diffusion",
            "dual_random_walk_diffusion",
        ],
        default="random_walk_diffusion",
    )
    parser.add_argument("--mpgcn_cheby_order", type=int, default=1)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def _print_results(results):
    print("\n=================================================")
    print("         FULL-NETWORK EVALUATION RESULTS")
    print("=================================================")
    for model_name, metrics in results.items():
        print(f"\n[{model_name}]")
        print(f" MSE   : {metrics['mse']:.6f}")
        print(f" RMSE  : {metrics['rmse']:.6f}")
        print(f" MAE   : {metrics['mae']:.6f}")
        print(f" SMAPE : {metrics['smape']:.6f}")
        print(f" WMAPE : {metrics['wmape']:.6f}")
    print("\n=================================================")


def main():
    args = parse_args()
    if not args.gnn_ckpt and not args.mpgcn_ckpt:
        raise ValueError("At least one of --gnn_ckpt or --mpgcn_ckpt must be provided.")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    results = {}

    if args.gnn_ckpt:
        print("\n=== Loading TrainGraph evaluation dataset ===")
        _, gnn_eval_dataset = get_dataset(
            data_root=args.data_root,
            train_subdir=args.test_subdir,
            val_subdir=args.test_subdir,
            window_size=args.window_size,
            hop_size=args.hop_size,
            pred_size=args.pred_size,
            time_resolution=args.time_resolution,
        )
        print(f"[TrainGraph] Evaluating checkpoint: {args.gnn_ckpt}")
        results["TrainGraph"] = evaluate_train_graph_full(
            ckpt_path=args.gnn_ckpt,
            dataset=gnn_eval_dataset,
            device=device,
            od_csv=args.od_csv,
            gate_tau=args.graph_gate_tau,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mape_eps=args.mape_eps,
        )

    if args.mpgcn_ckpt:
        print("\n=== Loading MPGCN train/eval datasets ===")
        mpgcn_trainset, mpgcn_eval_dataset = get_mpgcn_dataset(
            data_root=args.data_root,
            train_subdir=args.train_subdir,
            val_subdir=args.test_subdir,
            window_size=args.window_size,
            hop_size=args.hop_size,
            pred_size=args.pred_size,
            time_resolution=args.time_resolution,
            dynamic_bin_size=args.mpgcn_dynamic_bin_size,
            cache_in_mem=False,
        )
        print(f"[MPGCN] Evaluating checkpoint: {args.mpgcn_ckpt}")
        results["MPGCN"] = evaluate_mpgcn_full(
            ckpt_path=args.mpgcn_ckpt,
            train_dataset=mpgcn_trainset,
            eval_dataset=mpgcn_eval_dataset,
            device=device,
            od_csv=args.od_csv,
            dynamic_graph_cache=args.mpgcn_dynamic_graph_cache,
            dynamic_bin_size=args.mpgcn_dynamic_bin_size,
            kernel_type=args.mpgcn_kernel_type,
            cheby_order=args.mpgcn_cheby_order,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            mape_eps=args.mape_eps,
        )

    _print_results(results)

    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2)
        print(f"[Saved] {args.output_json}")


if __name__ == "__main__":
    main()
