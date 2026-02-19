# file: train_st_lstm.py
import argparse
import os
import numpy as np
import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import get_st_lstm_dataset
from SCIE_Benchmark.ST_LSTM import STLSTM
from trainer import STLSTMLM


def resolve_accelerator():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser("Train Paper-faithful ST-LSTM")

    # -------------------------
    # Data
    # -------------------------
    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--train_subdir", default="train")
    parser.add_argument("--val_subdir", default="test")

    # -------------------------
    # ST-LSTM parameters
    # -------------------------
    parser.add_argument("--H", type=int, default=60, help="Historical window")
    parser.add_argument("--h", type=int, default=30, help="Realtime window")
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--a", type=int, default=3, help="Number of historical days")

    parser.add_argument("--target_s", type=int, default=10)
    parser.add_argument("--target_e", type=int, default=25)

    # -------------------------
    # Training
    # -------------------------
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_epochs", type=int, default=300)

    # -------------------------
    # Model
    # -------------------------
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)

    # -------------------------
    # Logging
    # -------------------------
    parser.add_argument("--wandb_project", default="metro-st-lstm")
    parser.add_argument("--wandb_log_model", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    DEVICE = resolve_accelerator()

    print(f"=== Train ST-LSTM: OD {args.target_s} → {args.target_e} ===")
    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    
    trainset, valset = get_st_lstm_dataset(
        data_root=args.data_root,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        H=args.H,
        h=args.h,
        pred_size=args.pred_size,
        target_s=args.target_s,
        target_e=args.target_e,
        a=3,
        day_cluster_path="day_cluster.npy",
        top_x_od_path="top_x_od.npy",
        W_path="W.npy"
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

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    sample = trainset[0]
    input_dim = sample["x"].shape[-1]

    print(f"Input dim = {input_dim}")

    model = STLSTM(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pred_size=args.pred_size
    )

    lightning_module = STLSTMLM(
        model=model,
        lr=args.lr,
        loss=torch.nn.MSELoss()
    )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        log_model=args.wandb_log_model
    )
    wandb_logger.experiment.config.update(vars(args))

    trainer = L.Trainer(
        accelerator=DEVICE,
        devices=1,
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        log_every_n_steps=10
    )

    trainer.fit(lightning_module, train_loader, val_loader)


if __name__ == "__main__":
    main()
