# train/train_odformer.py
import argparse
import datetime
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import get_odformer_dataset
from SCIE_Benchmark.ODFormer import ODFormer   # 네가 구현한 ODFormer
from trainer import ODformerLM


def resolve_accelerator():
    return "cuda" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--train_subdir", default="train")
    parser.add_argument("--val_subdir", default="test")
    parser.add_argument("--adj_csv", default="./AD_matrix_trimmed_common.csv")

    # window
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=10)

    # model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.7)

    # training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_ddp", action="store_true")

    # logging
    parser.add_argument("--wandb_project", default="odformer-metro")

    args = parser.parse_args()

    # =====================
    # Dataset
    # =====================
    import pandas as pd
    adj = pd.read_csv(args.adj_csv, index_col=0).values
    print(f"Adjacency matrix shape: {adj.shape}")
    trainset, valset = get_odformer_dataset(
        data_root=args.data_root,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
        use_time_feature=True,
        cache_in_mem=False
    )

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # =====================
    # Model
    # =====================
    sample = trainset[0]
    N = sample["X"].shape[1]
    F = sample["X"].shape[-1]

    model = ODFormer(
        num_regions=N,
        feature_dim=F,
        hidden_dim=args.hidden_dim,
        out_feature_dim=1,
        alpha=args.alpha,
        num_heads=args.num_heads,
        pred_len=args.pred_size
    )

    # =====================
    # Lightning Module
    # =====================
    lm = ODformerLM(
        model=model,
        lr=args.lr,
        adj_matrix=adj,
    )

    # =====================
    # Trainer
    # =====================
    accelerator = resolve_accelerator()
    devices = torch.cuda.device_count() if args.use_ddp and accelerator == "cuda" else 1
    strategy = "ddp_find_unused_parameters_true" if args.use_ddp else "auto"

    run_name = f"ODformer_T{args.window_size}_P{args.pred_size}_{datetime.datetime.now():%Y%m%d_%H%M}"


    early_stop = EarlyStopping(
        monitor="val/loss",
        patience=3,          # 논문 기준: 8 epoch 이내 수렴
        mode="min",
        verbose=True
    )

    logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        config=vars(args)
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=logger,
        callbacks=[early_stop],
        log_every_n_steps=10,
        gradient_clip_val=1.0
    )

    trainer.fit(lm, train_loader, val_loader)


if __name__ == "__main__":
    main()
