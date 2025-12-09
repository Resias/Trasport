# file: train_TCN.py
import argparse
import os
import torch
import pandas as pd
import pytorch_lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from dataset import get_odpair_dataset
from benchmark_Model.TCNbased import TCN_Attention_LSTM
from trainer import TCNMetroLM

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def resolve_accelerator():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train TCN model with Metro pipeline")

    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--train_subdir", default="train")
    parser.add_argument("--val_subdir", default="test")

    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--pred_size", type=int, default=1)
    parser.add_argument("--hop_size", type=int, default=1)

    parser.add_argument("--od_i", type=int, default=10)
    parser.add_argument("--od_j", type=int, default=20)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=50)

    parser.add_argument("--wandb_project", default="metro-tcn")
    parser.add_argument("--wandb_log_model", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    DEVICE = resolve_accelerator()

    print("Loading dataset...")
    trainset, valset = get_odpair_dataset(
        data_root=args.data_root,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
        od_i=args.od_i,
        od_j=args.od_j
    )

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sample = trainset[0]
    input_dim = sample["x"].shape[-1]

    print(f"Detected input_dim = {input_dim}")

    model = TCN_Attention_LSTM(
        input_dim=input_dim,
        lstm_hidden=128,
        lstm_layers=3
    )

    lightning_module = TCNMetroLM(
        model=model,
        loss=torch.nn.MSELoss(),
        lr=args.lr
    )

    wandb_logger = WandbLogger(project=args.wandb_project, log_model=args.wandb_log_model)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=DEVICE,
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=50,
    )

    print("Start training...")
    trainer.fit(lightning_module, train_loader, val_loader)


if __name__ == "__main__":
    main()
