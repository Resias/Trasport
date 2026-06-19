import argparse
import datetime
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import get_mpgcn_dataset
from SCIE_Benchmark.MPGCN import AdjProcessor, MPGCN, get_support_K
from trainer import MPGCNLM


def resolve_accelerator():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def mpgcn_collate_fn(batch):
    return {
        "x": torch.stack([item["x"] for item in batch], dim=0),
        "y": torch.stack([item["y"] for item in batch], dim=0),
        "future_keys": torch.stack([item["future_keys"] for item in batch], dim=0),
    }


def to_numpy(day_data):
    if isinstance(day_data, torch.Tensor):
        return day_data.cpu().numpy()
    return np.asarray(day_data)


def cosine_distance_graph(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, a_min=1e-6, a_max=None)
    similarity = np.clip(matrix @ matrix.T, -1.0, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    return distance.astype(np.float32, copy=False)


def build_dynamic_graph_bank(
    dataset,
    dynamic_bin_size: int,
    kernel_type: str,
    cheby_order: int,
):
    if dynamic_bin_size <= 0:
        raise ValueError("dynamic_bin_size must be positive")

    num_bins = math.ceil(1440 / dynamic_bin_size)
    sum_by_bin = {}
    count_by_bin = np.zeros(num_bins, dtype=np.int64)
    global_sum = None
    global_count = 0

    for day_idx, day_data in enumerate(dataset.day_data_cache):
        day_np = to_numpy(day_data)
        valid_mask = dataset.valid_masks[day_idx]

        for step_idx in range(day_np.shape[0]):
            if step_idx >= len(valid_mask) or not bool(valid_mask[step_idx]):
                continue

            minute = (step_idx * dataset.time_resolution) % 1440
            key = min(num_bins - 1, minute // dynamic_bin_size)
            od_t = day_np[step_idx].astype(np.float64, copy=False)

            if key not in sum_by_bin:
                sum_by_bin[key] = od_t.copy()
            else:
                sum_by_bin[key] += od_t
            count_by_bin[key] += 1

            if global_sum is None:
                global_sum = od_t.copy()
            else:
                global_sum += od_t
            global_count += 1

    if global_sum is None or global_count == 0:
        raise RuntimeError("No valid OD slices were found for MPGCN dynamic graph construction")

    global_avg = (global_sum / global_count).astype(np.float32)
    avg_od_by_bin = []
    for key in range(num_bins):
        if count_by_bin[key] > 0:
            avg_od = (sum_by_bin[key] / count_by_bin[key]).astype(np.float32)
        else:
            avg_od = global_avg
        avg_od_by_bin.append(avg_od)

    origin_dyn = np.stack(
        [cosine_distance_graph(avg_od) for avg_od in avg_od_by_bin],
        axis=0,
    )
    dest_dyn = np.stack(
        [cosine_distance_graph(avg_od.T) for avg_od in avg_od_by_bin],
        axis=0,
    )

    processor = AdjProcessor(kernel_type=kernel_type, cheby_order=cheby_order)
    origin_supports = processor.process(torch.from_numpy(origin_dyn).float())
    dest_supports = processor.process(torch.from_numpy(dest_dyn).float())
    meta = {
        "num_bins": num_bins,
        "dynamic_bin_size": dynamic_bin_size,
        "bin_counts": count_by_bin.tolist(),
    }
    return origin_supports, dest_supports, meta


def load_dynamic_graph_bank(
    cache_path: str,
    dataset,
    dynamic_bin_size: int,
    kernel_type: str,
    cheby_order: int,
):
    if cache_path and os.path.exists(cache_path):
        payload = torch.load(cache_path, map_location="cpu")
        expected_k = get_support_K(kernel_type, cheby_order)
        expected_bins = math.ceil(1440 / dynamic_bin_size)

        origin = payload["origin_dynamic_supports"].float()
        dest = payload["destination_dynamic_supports"].float()
        meta = payload.get("meta", {})

        cache_ok = (
            origin.dim() == 4
            and dest.dim() == 4
            and origin.shape == dest.shape
            and origin.shape[0] == expected_bins
            and origin.shape[1] == expected_k
            and meta.get("dynamic_bin_size") == dynamic_bin_size
            and meta.get("num_bins") == expected_bins
        )

        if cache_ok:
            return origin, dest, meta

        print(
            "Cached dynamic graph bank does not match current settings. "
            "Rebuilding cache..."
        )

    origin_supports, dest_supports, meta = build_dynamic_graph_bank(
        dataset=dataset,
        dynamic_bin_size=dynamic_bin_size,
        kernel_type=kernel_type,
        cheby_order=cheby_order,
    )

    if cache_path:
        out_dir = os.path.dirname(cache_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torch.save(
            {
                "origin_dynamic_supports": origin_supports.cpu(),
                "destination_dynamic_supports": dest_supports.cpu(),
                "meta": meta,
            },
            cache_path,
        )

    return origin_supports, dest_supports, meta


def load_static_supports(od_csv: str, kernel_type: str, cheby_order: int) -> torch.Tensor:
    od_df = pd.read_csv(od_csv, index_col=0)
    adj = torch.tensor(od_df.values, dtype=torch.float32)
    processor = AdjProcessor(kernel_type=kernel_type, cheby_order=cheby_order)
    return processor.process(adj).squeeze(0)


def get_loss_fn(loss_name: str):
    if loss_name == "MSE":
        return torch.nn.MSELoss()
    if loss_name == "MAE":
        return torch.nn.L1Loss()
    if loss_name == "Huber":
        return torch.nn.SmoothL1Loss()
    raise ValueError(f"Unsupported loss: {loss_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MPGCN benchmark on Transport OD data")
    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--train_subdir", default="train")
    parser.add_argument("--val_subdir", default="test")
    parser.add_argument("--cache_dataset", action="store_true")
    parser.add_argument("--dynamic_graph_cache", default="")

    parser.add_argument("--od_csv", default="./AD_matrix_trimmed_common.csv")
    parser.add_argument("--time_resolution", type=int, default=1)
    parser.add_argument("--dynamic_bin_size", type=int, default=60)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--loss", choices=["MSE", "MAE", "Huber"], default="MSE")

    parser.add_argument(
        "--kernel_type",
        choices=[
            "chebyshev",
            "localpool",
            "random_walk_diffusion",
            "dual_random_walk_diffusion",
        ],
        default="random_walk_diffusion",
    )
    parser.add_argument("--cheby_order", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--gcn_layers", type=int, default=2)
    parser.add_argument("--train_rollout_steps", type=int, default=1)

    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_root", default="./review_runs")
    parser.add_argument("--study_name", default="")
    parser.add_argument("--resume_if_complete", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default="")
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--wandb_project", default="metro-MPGCN")
    return parser.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    args.output_root = os.path.abspath(args.output_root)
    study_name = args.study_name or datetime.datetime.now().strftime("review_mpgcn_%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_root, study_name, "MPGCN", f"seed_{args.seed}")
    result_path = os.path.join(run_dir, "run_result.json")
    if args.resume_if_complete and os.path.exists(result_path):
        print(f"[Skip] completed run found: {result_path}")
        return

    os.makedirs(run_dir, exist_ok=True)
    L.seed_everything(args.seed, workers=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fp:
        json.dump(vars(args), fp, indent=2)

    trainset, valset = get_mpgcn_dataset(
        data_root=args.data_root,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
        time_resolution=args.time_resolution,
        dynamic_bin_size=args.dynamic_bin_size,
        cache_in_mem=True if args.cache_dataset else False,
    )

    static_supports = load_static_supports(
        od_csv=args.od_csv,
        kernel_type=args.kernel_type,
        cheby_order=args.cheby_order,
    )
    origin_dyn_supports, dest_dyn_supports, dyn_meta = load_dynamic_graph_bank(
        cache_path=args.dynamic_graph_cache,
        dataset=trainset,
        dynamic_bin_size=args.dynamic_bin_size,
        kernel_type=args.kernel_type,
        cheby_order=args.cheby_order,
    )

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=mpgcn_collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=mpgcn_collate_fn,
        pin_memory=True,
    )

    sample = trainset[0]
    num_nodes = sample["x"].shape[-1]
    support_k = get_support_K(args.kernel_type, args.cheby_order)

    if static_supports.shape[-1] != num_nodes:
        raise ValueError(
            f"Adjacency node count {static_supports.shape[-1]} does not match dataset node count {num_nodes}"
        )

    model = MPGCN(
        M=2,
        K=support_k,
        input_dim=1,
        lstm_hidden_dim=args.hidden_dim,
        lstm_num_layers=args.lstm_layers,
        gcn_hidden_dim=args.hidden_dim,
        gcn_num_layers=args.gcn_layers,
        num_nodes=num_nodes,
        use_bias=True,
        activation=torch.nn.ReLU,
    )

    lm = MPGCNLM(
        model=model,
        static_supports=static_supports,
        origin_dynamic_supports=origin_dyn_supports,
        destination_dynamic_supports=dest_dyn_supports,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss=get_loss_fn(args.loss),
        train_rollout_steps=args.train_rollout_steps,
    )

    accelerator = resolve_accelerator()
    devices = 1
    strategy = "auto"
    if args.use_ddp:
        strategy = "ddp_find_unused_parameters_true"
        devices = torch.cuda.device_count() if accelerator == "cuda" else os.cpu_count()

    run_name = (
        f"MPGCN_bs{args.batch_size}"
        f"_T{args.window_size}_P{args.pred_size}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    logger = False
    if not args.disable_wandb:
        logger = WandbLogger(
            project=args.wandb_project,
            name=run_name,
            config={**vars(args), **dyn_meta},
        )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="best-{epoch:03d}",
        monitor="val/smape",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint_callback]
    if args.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor="val/smape",
                mode="min",
                patience=args.early_stop_patience,
                min_delta=args.early_stop_min_delta,
                verbose=True,
            )
        )

    trainer = L.Trainer(
        default_root_dir=run_dir,
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
    )

    train_start = time.perf_counter()
    ckpt_path = args.resume_from_checkpoint or None
    if ckpt_path and not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
    if ckpt_path:
        print(f"[Resume] checkpoint: {ckpt_path}")
    trainer.fit(lm, train_loader, val_loader, ckpt_path=ckpt_path)
    train_time_sec = time.perf_counter() - train_start

    result = {
        "model": "MPGCN",
        "seed": args.seed,
        "run_dir": run_dir,
        "best_checkpoint": checkpoint_callback.best_model_path,
        "best_val_smape": (
            float(checkpoint_callback.best_model_score.item())
            if checkpoint_callback.best_model_score is not None
            else None
        ),
        "train_time_sec": train_time_sec,
        "train_time_hours": train_time_sec / 3600.0,
        "dynamic_graph_meta": dyn_meta,
        "args": vars(args),
    }
    with open(result_path, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)
    print(f"[Saved] {result_path}")


if __name__ == "__main__":
    main()
