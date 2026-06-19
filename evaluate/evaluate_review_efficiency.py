import argparse
import json
import os
import sys
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from dataset import get_dataset, get_mpgcn_dataset, get_odformer_dataset
from evaluate_review_checkpoints import (
    _load_json,
    _result_path,
    _static_edge_index,
)
from graph_eval_utils import _build_graph_week_loader, _run_graph_week_batch, load_graph_week_model
from mpgcn_eval_utils import _build_mpgcn_loader, _run_mpgcn_batch, load_mpgcn_model
from SCIE_Benchmark.Autoformer import AutoformerODFormal
from SCIE_Benchmark.GCN_LSTM import GCN_LSTM_OD
from SCIE_Benchmark.ODFormer import ODFormer
from train.train_autoformer import autoformer_collate_fn
from train.train_gcn_lstm import gcn_lstm_collate_fn
from train.trainer import MetroAutoformerODLM, MetroGCNLSTMLM, ODformerLM


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
        yield batch


def _run_timed(module, loader, run_batch, device, warmup_batches, max_batches):
    with torch.no_grad():
        for batch in _take_batches(loader, warmup_batches):
            run_batch(module, batch, device)

    _sync(device)
    _reset_memory(device)
    start = time.perf_counter()
    num_batches = 0
    num_samples = 0
    num_od_matrices = 0

    with torch.no_grad():
        for batch in _take_batches(loader, max_batches):
            pred = run_batch(module, batch, device)
            num_batches += 1
            num_samples += int(pred.shape[0])
            num_od_matrices += int(pred.shape[0] * pred.shape[1])

    _sync(device)
    elapsed = time.perf_counter() - start
    return {
        "elapsed_sec": elapsed,
        "num_batches": num_batches,
        "num_samples": num_samples,
        "num_od_matrices": num_od_matrices,
        "latency_ms_per_sample": 1000.0 * elapsed / max(num_samples, 1),
        "throughput_samples_per_sec": num_samples / max(elapsed, 1e-12),
        "throughput_od_matrices_per_sec": num_od_matrices / max(elapsed, 1e-12),
        "peak_gpu_memory_gb": _peak_memory_gb(device),
    }


def _graph_run_batch(module_bundle, batch, device):
    lightning_module, _static_edge_index = module_bundle
    pred, _true = _run_graph_week_batch(lightning_module, batch, device)
    return pred


def _mpgcn_run_batch(module, batch, device):
    pred, _true = _run_mpgcn_batch(module, batch, device)
    return pred


def _gcn_lstm_run_batch(module, batch, device):
    x, _y = batch
    pred = module.model(x.to(device))
    return torch.expm1(torch.clamp(pred, min=0.0))


def _autoformer_run_batch(module, batch, device):
    batch = {key: val.to(device) for key, val in batch.items()}
    pred = module(batch)
    return torch.expm1(torch.clamp(pred, min=0.0))


def _odformer_run_batch(module, batch, device):
    pred = module(batch["X"].to(device))
    if pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    return torch.expm1(pred)


def _load_gcn_lstm(ckpt, config, dataset, device, od_csv):
    sample = dataset[0]
    model = GCN_LSTM_OD(
        num_nodes=sample["x_tensor"].shape[-1],
        edge_index=_static_edge_index(od_csv),
        hid_dim=config.get("gcn_hidden", 64),
        lstm_layers=config.get("lstm_layers", 1),
        pred_steps=config.get("pred_size", 30),
    )
    return MetroGCNLSTMLM.load_from_checkpoint(
        ckpt,
        map_location=device,
        model=model,
        lr=config.get("lr", 1e-3),
    ).to(device).eval()


def _load_autoformer(ckpt, config, dataset, device):
    sample = dataset[0]
    model = AutoformerODFormal(
        num_nodes=sample["x_tensor"].shape[-1],
        d_model=config.get("d_model", 128),
        ff_dim=config.get("ff_dim", 256),
        enc_layers=config.get("enc_layers", 2),
        dec_layers=config.get("dec_layers", 2),
        pred_steps=config.get("pred_size", 30),
        rank=config.get("rank", 32),
        kernel_size=config.get("kernel_size", 25),
        top_k=config.get("top_k", 8),
        time_dim=sample["time_enc_hist"].shape[-1],
        use_weekday=True,
    )
    return MetroAutoformerODLM.load_from_checkpoint(
        ckpt,
        map_location=device,
        model=model,
        lr=config.get("lr", 1e-4),
    ).to(device).eval()


def _load_odformer(ckpt, config, dataset, device, od_csv):
    adj = pd.read_csv(od_csv, index_col=0).values
    sample = dataset[0]
    model = ODFormer(
        num_regions=sample["X"].shape[1],
        feature_dim=sample["X"].shape[-1],
        hidden_dim=config.get("hidden_dim", 64),
        out_feature_dim=1,
        alpha=config.get("alpha", 0.7),
        num_heads=config.get("num_heads", 4),
        pred_len=config.get("pred_size", 30),
    )
    return ODformerLM.load_from_checkpoint(
        ckpt,
        map_location=device,
        model=model,
        lr=config.get("lr", 1e-3),
        adj_matrix=adj,
    ).to(device).eval()


def _aggregate(rows):
    df = pd.DataFrame(rows)
    metric_cols = [
        "params",
        "checkpoint_mb",
        "peak_gpu_memory_gb",
        "latency_ms_per_sample",
        "throughput_samples_per_sec",
        "throughput_od_matrices_per_sec",
        "elapsed_sec",
    ]
    summary_rows = []
    for model_name, group in df.groupby("model", sort=False):
        row = {"model": model_name, "runs": int(len(group))}
        for col in metric_cols:
            row[f"{col}_mean"] = float(group[col].mean())
            row[f"{col}_std"] = float(0.0 if len(group) < 2 else group[col].std(ddof=1))
        summary_rows.append(row)
    return df, pd.DataFrame(summary_rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Measure efficiency for reduced reviewer checkpoints.")
    parser.add_argument("--study_dir", default="/root/tmp/Trasport/review_runs/review_3way_50ep_20260512")
    parser.add_argument("--data_root", default="/root/tmp/Trasport/data_splits/od_minute_review_3way")
    parser.add_argument("--train_subdir", default="train")
    parser.add_argument("--test_subdir", default="test")
    parser.add_argument("--models", default="S8_gate,ODFormer,MPGCN,GCN_LSTM,Autoformer")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--od_csv", default=os.path.join(ROOT, "AD_matrix_trimmed_common.csv"))
    parser.add_argument("--time_resolution", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=10)
    parser.add_argument("--gate_tau", type=float, default=0.9)
    parser.add_argument("--mpgcn_dynamic_bin_size", type=int, default=60)
    parser.add_argument("--mpgcn_dynamic_graph_cache", default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warmup_batches", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=20)
    parser.add_argument("--device", default="")
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_names = [part.strip() for part in args.models.split(",") if part.strip()]
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    output_dir = args.output_dir or os.path.join(args.study_dir, "efficiency")
    os.makedirs(output_dir, exist_ok=True)

    generic_test_dataset = None
    mpgcn_train_dataset = None
    mpgcn_test_dataset = None
    odformer_test_dataset = None
    rows = []

    for model_name in model_names:
        for seed in seeds:
            result_path = _result_path(args.study_dir, model_name, seed)
            if not os.path.exists(result_path):
                print(f"[Skip] missing result: {result_path}")
                continue
            result = _load_json(result_path)
            ckpt = result.get("best_checkpoint")
            if not ckpt or not os.path.exists(ckpt):
                print(f"[Skip] missing checkpoint for {model_name} seed={seed}: {ckpt}")
                continue
            config = result.get("args") or result.get("train_args") or {}
            print(f"[Efficiency] {model_name} seed={seed}")

            if model_name == "S8_gate":
                if generic_test_dataset is None:
                    _, generic_test_dataset = get_dataset(
                        data_root=args.data_root,
                        train_subdir=args.test_subdir,
                        val_subdir=args.test_subdir,
                        window_size=args.window_size,
                        hop_size=args.hop_size,
                        pred_size=args.pred_size,
                        time_resolution=args.time_resolution,
                        cache_in_mem=False,
                    )
                lm, static_edge_index = load_graph_week_model(
                    ckpt, device, args.od_csv, gate_tau=args.gate_tau
                )
                loader = _build_graph_week_loader(
                    generic_test_dataset, static_edge_index, args.batch_size, args.num_workers
                )
                module = (lm, static_edge_index)
                run_batch = _graph_run_batch
                params_module = lm.model
            elif model_name == "MPGCN":
                if mpgcn_train_dataset is None or mpgcn_test_dataset is None:
                    mpgcn_train_dataset, mpgcn_test_dataset = get_mpgcn_dataset(
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
                module = load_mpgcn_model(
                    ckpt_path=ckpt,
                    device=device,
                    train_dataset=mpgcn_train_dataset,
                    od_csv=args.od_csv,
                    dynamic_graph_cache=(
                        args.mpgcn_dynamic_graph_cache
                        or os.path.join(args.study_dir, "artifacts", "mpgcn_dyn_60m.pt")
                    ),
                    dynamic_bin_size=args.mpgcn_dynamic_bin_size,
                    kernel_type=config.get("kernel_type", "random_walk_diffusion"),
                    cheby_order=config.get("cheby_order", 1),
                )
                loader = _build_mpgcn_loader(mpgcn_test_dataset, args.batch_size, args.num_workers)
                run_batch = _mpgcn_run_batch
                params_module = module.model
            elif model_name == "GCN_LSTM":
                if generic_test_dataset is None:
                    _, generic_test_dataset = get_dataset(
                        data_root=args.data_root,
                        train_subdir=args.test_subdir,
                        val_subdir=args.test_subdir,
                        window_size=args.window_size,
                        hop_size=args.hop_size,
                        pred_size=args.pred_size,
                        time_resolution=args.time_resolution,
                        cache_in_mem=False,
                    )
                module = _load_gcn_lstm(ckpt, config, generic_test_dataset, device, args.od_csv)
                loader = DataLoader(
                    generic_test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=gcn_lstm_collate_fn,
                    pin_memory=True,
                )
                run_batch = _gcn_lstm_run_batch
                params_module = module.model
            elif model_name == "Autoformer":
                if generic_test_dataset is None:
                    _, generic_test_dataset = get_dataset(
                        data_root=args.data_root,
                        train_subdir=args.test_subdir,
                        val_subdir=args.test_subdir,
                        window_size=args.window_size,
                        hop_size=args.hop_size,
                        pred_size=args.pred_size,
                        time_resolution=args.time_resolution,
                        cache_in_mem=False,
                    )
                module = _load_autoformer(ckpt, config, generic_test_dataset, device)
                loader = DataLoader(
                    generic_test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    collate_fn=autoformer_collate_fn,
                    pin_memory=True,
                )
                run_batch = _autoformer_run_batch
                params_module = module.model
            elif model_name == "ODFormer":
                if odformer_test_dataset is None:
                    _, odformer_test_dataset = get_odformer_dataset(
                        data_root=args.data_root,
                        train_subdir=args.train_subdir,
                        val_subdir=args.test_subdir,
                        window_size=args.window_size,
                        hop_size=args.hop_size,
                        pred_size=args.pred_size,
                        use_time_feature=True,
                        cache_in_mem=False,
                    )
                module = _load_odformer(ckpt, config, odformer_test_dataset, device, args.od_csv)
                loader = DataLoader(
                    odformer_test_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True,
                )
                run_batch = _odformer_run_batch
                params_module = module.model
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            timed = _run_timed(
                module=module,
                loader=loader,
                run_batch=run_batch,
                device=device,
                warmup_batches=args.warmup_batches,
                max_batches=args.max_batches,
            )
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "checkpoint": ckpt,
                    "params": _count_trainable_params(params_module),
                    "checkpoint_mb": _checkpoint_mb(ckpt),
                    "batch_size": args.batch_size,
                    "warmup_batches": args.warmup_batches,
                    "max_batches": args.max_batches,
                    **timed,
                }
            )

            del module, loader
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No efficiency rows were produced.")

    per_run_df, summary_df = _aggregate(rows)
    per_run_path = os.path.join(output_dir, "per_run_efficiency.csv")
    summary_path = os.path.join(output_dir, "summary_efficiency.csv")
    json_path = os.path.join(output_dir, "efficiency.json")
    per_run_df.to_csv(per_run_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump({"per_run": rows, "summary": summary_df.to_dict(orient="records")}, fp, indent=2)

    print(f"[Saved] {per_run_path}")
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {json_path}")


if __name__ == "__main__":
    main()
