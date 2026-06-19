import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from dataset import get_dataset, get_mpgcn_dataset
from graph_eval_utils import _build_graph_week_loader, _run_graph_week_batch, load_graph_week_model
from mpgcn_eval_utils import _build_mpgcn_loader, _run_mpgcn_batch, load_mpgcn_model


def _sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_memory(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_gb(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    return None


def _count_trainable_params(module):
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def _checkpoint_mb(path):
    return os.path.getsize(path) / (1024 ** 2)


def _take_batches(loader, max_batches):
    for idx, batch in enumerate(loader):
        if max_batches is not None and idx >= max_batches:
            break
        yield idx, batch


def benchmark_graph(args, device):
    _, eval_dataset = get_dataset(
        data_root=args.data_root,
        train_subdir=args.test_subdir,
        val_subdir=args.test_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
        time_resolution=args.time_resolution,
        cache_in_mem=args.cache_dataset,
    )
    lightning_module, static_edge_index = load_graph_week_model(
        args.ckpt,
        device,
        args.od_csv,
        gate_tau=args.gate_tau,
    )
    loader = _build_graph_week_loader(eval_dataset, static_edge_index, args.batch_size, args.num_workers)

    # Warm up on a separate iterator so timed results include the same batch count every run.
    with torch.no_grad():
        for idx, batch in _take_batches(loader, args.warmup_batches):
            _run_graph_week_batch(lightning_module, batch, device)

    _sync(device)
    _reset_memory(device)
    start = time.perf_counter()
    num_batches = 0
    num_samples = 0
    num_od_matrices = 0

    with torch.no_grad():
        for _, batch in _take_batches(loader, args.max_batches):
            pred_full, _true_full = _run_graph_week_batch(lightning_module, batch, device)
            num_batches += 1
            num_samples += pred_full.shape[0]
            num_od_matrices += pred_full.shape[0] * pred_full.shape[1]

    _sync(device)
    elapsed = time.perf_counter() - start
    return {
        "model": "TrainGraph",
        "checkpoint": args.ckpt,
        "params": _count_trainable_params(lightning_module.model),
        "checkpoint_mb": _checkpoint_mb(args.ckpt),
        "peak_gpu_memory_gb": _peak_memory_gb(device),
        "elapsed_sec": elapsed,
        "num_batches": num_batches,
        "num_samples": num_samples,
        "num_od_matrices": num_od_matrices,
        "latency_ms_per_sample": 1000.0 * elapsed / max(num_samples, 1),
        "throughput_samples_per_sec": num_samples / max(elapsed, 1e-12),
        "throughput_od_matrices_per_sec": num_od_matrices / max(elapsed, 1e-12),
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
    }


def benchmark_mpgcn(args, device):
    train_dataset, eval_dataset = get_mpgcn_dataset(
        data_root=args.data_root,
        train_subdir=args.train_subdir,
        val_subdir=args.test_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
        time_resolution=args.time_resolution,
        dynamic_bin_size=args.mpgcn_dynamic_bin_size,
        cache_in_mem=args.cache_dataset,
    )
    lightning_module = load_mpgcn_model(
        ckpt_path=args.ckpt,
        device=device,
        train_dataset=train_dataset,
        od_csv=args.od_csv,
        dynamic_graph_cache=args.mpgcn_dynamic_graph_cache,
        dynamic_bin_size=args.mpgcn_dynamic_bin_size,
        kernel_type=args.mpgcn_kernel_type,
        cheby_order=args.mpgcn_cheby_order,
    )
    loader = _build_mpgcn_loader(eval_dataset, args.batch_size, args.num_workers)

    with torch.no_grad():
        for idx, batch in _take_batches(loader, args.warmup_batches):
            _run_mpgcn_batch(lightning_module, batch, device)

    _sync(device)
    _reset_memory(device)
    start = time.perf_counter()
    num_batches = 0
    num_samples = 0
    num_od_matrices = 0

    with torch.no_grad():
        for _, batch in _take_batches(loader, args.max_batches):
            pred_full, _true_full = _run_mpgcn_batch(lightning_module, batch, device)
            num_batches += 1
            num_samples += pred_full.shape[0]
            num_od_matrices += pred_full.shape[0] * pred_full.shape[1]

    _sync(device)
    elapsed = time.perf_counter() - start
    return {
        "model": "MPGCN",
        "checkpoint": args.ckpt,
        "params": _count_trainable_params(lightning_module.model),
        "checkpoint_mb": _checkpoint_mb(args.ckpt),
        "peak_gpu_memory_gb": _peak_memory_gb(device),
        "elapsed_sec": elapsed,
        "num_batches": num_batches,
        "num_samples": num_samples,
        "num_od_matrices": num_od_matrices,
        "latency_ms_per_sample": 1000.0 * elapsed / max(num_samples, 1),
        "throughput_samples_per_sec": num_samples / max(elapsed, 1e-12),
        "throughput_od_matrices_per_sec": num_od_matrices / max(elapsed, 1e-12),
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Measure model complexity and inference efficiency.")
    parser.add_argument("--model", choices=["graph", "mpgcn"], required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", default="/root/tmp/Trasport/data_splits/od_minute_review_3way")
    parser.add_argument("--train_subdir", default="train")
    parser.add_argument("--test_subdir", default="test")
    parser.add_argument("--od_csv", default=os.path.join(ROOT, "AD_matrix_trimmed_common.csv"))
    parser.add_argument("--time_resolution", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cache_dataset", action="store_true")
    parser.add_argument("--warmup_batches", type=int, default=5)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", default="")
    parser.add_argument("--gate_tau", type=float, default=0.9)

    parser.add_argument("--mpgcn_dynamic_bin_size", type=int, default=60)
    parser.add_argument("--mpgcn_dynamic_graph_cache", default=os.path.join(ROOT, "artifacts", "mpgcn_dyn_60m.pt"))
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
    parser.add_argument("--output_json", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.model == "graph":
        result = benchmark_graph(args, device)
    else:
        result = benchmark_mpgcn(args, device)

    print(json.dumps(result, indent=2))
    if args.output_json:
        output_dir = os.path.dirname(args.output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as fp:
            json.dump(result, fp, indent=2)


if __name__ == "__main__":
    main()
