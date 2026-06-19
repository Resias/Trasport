import argparse
import json
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_geometric.utils import dense_to_sparse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from dataset import get_dataset, get_mpgcn_dataset, get_odformer_dataset
from evaluate_full_network import evaluate_mpgcn_full, evaluate_train_graph_full
from SCIE_Benchmark.Autoformer import AutoformerODFormal
from SCIE_Benchmark.GCN_LSTM import GCN_LSTM_OD
from SCIE_Benchmark.ODFormer import ODFormer
from train.train_autoformer import autoformer_collate_fn
from train.train_gcn_lstm import gcn_lstm_collate_fn
from train.trainer import MetroAutoformerODLM, MetroGCNLSTMLM, ODformerLM


def _new_state():
    return {
        "sq_error_sum": 0.0,
        "abs_error_sum": 0.0,
        "true_abs_sum": 0.0,
        "smape_term_sum": 0.0,
        "count": 0,
    }


def _update_state(state, pred_full, true_full, eps):
    diff = pred_full - true_full
    abs_diff = torch.abs(diff)
    denom = (torch.abs(pred_full) + torch.abs(true_full)).clamp(min=eps)
    state["sq_error_sum"] += torch.sum(diff ** 2).item()
    state["abs_error_sum"] += torch.sum(abs_diff).item()
    state["true_abs_sum"] += torch.sum(torch.abs(true_full)).item()
    state["smape_term_sum"] += torch.sum(2.0 * abs_diff / denom).item()
    state["count"] += diff.numel()


def _finalize_state(state):
    if state["count"] == 0:
        raise RuntimeError("No samples were evaluated.")
    mse = state["sq_error_sum"] / state["count"]
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "mae": state["abs_error_sum"] / state["count"],
        "smape": 100.0 * state["smape_term_sum"] / state["count"],
        "wmape": 100.0 * state["abs_error_sum"] / (state["true_abs_sum"] + 1e-8),
    }


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _result_path(study_dir, model_name, seed):
    subdir = {
        "S8_gate": "S8_gate",
        "ODFormer": "ODFormer",
        "MPGCN": "MPGCN",
        "GCN_LSTM": "GCN_LSTM",
        "Autoformer": "Autoformer",
    }[model_name]
    return os.path.join(study_dir, subdir, f"seed_{seed}", "run_result.json")


def _static_edge_index(od_csv):
    od_df = pd.read_csv(od_csv, index_col=0)
    adj = torch.tensor(od_df.values, dtype=torch.float32)
    edge_index, _ = dense_to_sparse(adj)
    return edge_index


def evaluate_gcn_lstm(ckpt, config, dataset, device, od_csv, batch_size, num_workers, eps):
    sample = dataset[0]
    num_nodes = sample["x_tensor"].shape[-1]
    model = GCN_LSTM_OD(
        num_nodes=num_nodes,
        edge_index=_static_edge_index(od_csv),
        hid_dim=config.get("gcn_hidden", 64),
        lstm_layers=config.get("lstm_layers", 1),
        pred_steps=config.get("pred_size", 30),
    )
    lm = MetroGCNLSTMLM.load_from_checkpoint(
        ckpt,
        map_location=device,
        model=model,
        lr=config.get("lr", 1e-3),
    ).to(device)
    lm.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=gcn_lstm_collate_fn,
        pin_memory=True,
    )
    state = _new_state()
    with torch.no_grad():
        for x, y in loader:
            pred = lm.model(x.to(device))
            true = y.to(device)
            _update_state(state, torch.expm1(torch.clamp(pred, min=0.0)), torch.expm1(true), eps)
    return _finalize_state(state)


def evaluate_autoformer(ckpt, config, dataset, device, batch_size, num_workers, eps):
    sample = dataset[0]
    num_nodes = sample["x_tensor"].shape[-1]
    time_dim = sample["time_enc_hist"].shape[-1]
    model = AutoformerODFormal(
        num_nodes=num_nodes,
        d_model=config.get("d_model", 128),
        ff_dim=config.get("ff_dim", 256),
        enc_layers=config.get("enc_layers", 2),
        dec_layers=config.get("dec_layers", 2),
        pred_steps=config.get("pred_size", 30),
        rank=config.get("rank", 32),
        kernel_size=config.get("kernel_size", 25),
        top_k=config.get("top_k", 8),
        time_dim=time_dim,
        use_weekday=True,
    )
    lm = MetroAutoformerODLM.load_from_checkpoint(
        ckpt,
        map_location=device,
        model=model,
        lr=config.get("lr", 1e-4),
    ).to(device)
    lm.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=autoformer_collate_fn,
        pin_memory=True,
    )
    state = _new_state()
    with torch.no_grad():
        for batch in loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            pred = lm(batch)
            true = batch["y"]
            _update_state(state, torch.expm1(torch.clamp(pred, min=0.0)), torch.expm1(true), eps)
    return _finalize_state(state)


def evaluate_odformer(ckpt, config, dataset, device, od_csv, batch_size, num_workers, eps):
    adj = pd.read_csv(od_csv, index_col=0).values
    sample = dataset[0]
    num_nodes = sample["X"].shape[1]
    feature_dim = sample["X"].shape[-1]
    model = ODFormer(
        num_regions=num_nodes,
        feature_dim=feature_dim,
        hidden_dim=config.get("hidden_dim", 64),
        out_feature_dim=1,
        alpha=config.get("alpha", 0.7),
        num_heads=config.get("num_heads", 4),
        pred_len=config.get("pred_size", 30),
    )
    lm = ODformerLM.load_from_checkpoint(
        ckpt,
        map_location=device,
        model=model,
        lr=config.get("lr", 1e-3),
        adj_matrix=adj,
    ).to(device)
    lm.eval()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    state = _new_state()
    with torch.no_grad():
        for batch in loader:
            x = batch["X"].to(device)
            true = batch["Y"].to(device)
            pred = lm(x)
            if pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            if true.shape[-1] == 1:
                true = true.squeeze(-1)
            _update_state(state, torch.expm1(pred), torch.expm1(true), eps)
    return _finalize_state(state)


def _aggregate(rows):
    df = pd.DataFrame(rows)
    metric_cols = [
        "mse",
        "rmse",
        "mae",
        "smape",
        "wmape",
        "train_time_sec",
        "train_time_hours",
    ]
    summary = []
    for model_name, group in df.groupby("model", sort=False):
        row = {"model": model_name, "runs": int(len(group))}
        for col in metric_cols:
            if col in group:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_std"] = float(0.0 if len(group) < 2 else group[col].std(ddof=1))
        summary.append(row)
    return df, pd.DataFrame(summary)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate reduced reviewer rerun checkpoints on test split.")
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
    parser.add_argument("--graph_gate_tau", type=float, default=0.9)
    parser.add_argument("--mape_eps", type=float, default=1e-3)
    parser.add_argument("--mpgcn_dynamic_bin_size", type=int, default=60)
    parser.add_argument("--mpgcn_dynamic_graph_cache", default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="")
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_names = [part.strip() for part in args.models.split(",") if part.strip()]
    seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
    output_dir = args.output_dir or os.path.join(args.study_dir, "evaluation")
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
            print(f"[Eval] {model_name} seed={seed}")

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
                metrics = evaluate_train_graph_full(
                    ckpt_path=ckpt,
                    dataset=generic_test_dataset,
                    device=device,
                    od_csv=args.od_csv,
                    gate_tau=args.graph_gate_tau,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    mape_eps=args.mape_eps,
                )
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
                metrics = evaluate_mpgcn_full(
                    ckpt_path=ckpt,
                    train_dataset=mpgcn_train_dataset,
                    eval_dataset=mpgcn_test_dataset,
                    device=device,
                    od_csv=args.od_csv,
                    dynamic_graph_cache=(
                        args.mpgcn_dynamic_graph_cache
                        or os.path.join(args.study_dir, "artifacts", "mpgcn_dyn_60m.pt")
                    ),
                    dynamic_bin_size=args.mpgcn_dynamic_bin_size,
                    kernel_type=config.get("kernel_type", "random_walk_diffusion"),
                    cheby_order=config.get("cheby_order", 1),
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    mape_eps=args.mape_eps,
                )
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
                metrics = evaluate_gcn_lstm(
                    ckpt, config, generic_test_dataset, device, args.od_csv, args.batch_size, args.num_workers, args.mape_eps
                )
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
                metrics = evaluate_autoformer(
                    ckpt, config, generic_test_dataset, device, args.batch_size, args.num_workers, args.mape_eps
                )
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
                metrics = evaluate_odformer(
                    ckpt, config, odformer_test_dataset, device, args.od_csv, args.batch_size, args.num_workers, args.mape_eps
                )
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "checkpoint": ckpt,
                    "train_time_sec": result.get("train_time_sec"),
                    "train_time_hours": result.get("train_time_hours"),
                    **metrics,
                }
            )

    if not rows:
        raise RuntimeError("No checkpoints were evaluated.")

    per_run_df, summary_df = _aggregate(rows)
    per_run_path = os.path.join(output_dir, "per_run_test_metrics.csv")
    summary_path = os.path.join(output_dir, "summary_test_metrics.csv")
    json_path = os.path.join(output_dir, "test_metrics.json")
    per_run_df.to_csv(per_run_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump({"per_run": rows, "summary": summary_df.to_dict(orient="records")}, fp, indent=2)

    print(f"[Saved] {per_run_path}")
    print(f"[Saved] {summary_path}")
    print(f"[Saved] {json_path}")


if __name__ == "__main__":
    main()
