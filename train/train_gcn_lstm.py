# train/train_gcn_lstm.py
import argparse
import datetime
import torch
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import get_dataset
from torch_geometric.utils import dense_to_sparse

from SCIE_Benchmark.GCN_LSTM import GCN_LSTM_OD
from trainer import MetroGCNLSTMLM


def resolve_accelerator():
    return "cuda" if torch.cuda.is_available() else "cpu"


def gcn_lstm_collate_fn(batch):
    """
    batch: list of MetroDataset samples
    returns:
        x: (B,T,N,N)
        y: (B,T_out,N,N)
    """
    x = torch.stack([b["x_tensor"] for b in batch], dim=0)
    y = torch.stack([b["y_tensor"] for b in batch], dim=0)
    return x, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--train_subdir", default="train")
    parser.add_argument("--val_subdir", default="test")
    parser.add_argument("--cache_dataset", action="store_true")

    parser.add_argument("--od_csv", default="./AD_matrix_trimmed_common.csv")

    parser.add_argument("--time_resolution", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=10)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=200)

    parser.add_argument("--gcn_hidden", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=1)

    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--wandb_project", default="metro-GCN-LSTM")

    args = parser.parse_args()

    # =====================
    # Static adjacency
    # =====================
    od_df = pd.read_csv(args.od_csv, index_col=0)
    adj = torch.tensor(od_df.values, dtype=torch.float32)
    static_edge_index, _ = dense_to_sparse(adj)

    num_nodes = static_edge_index.max().item() + 1

    # =====================
    # Dataset
    # =====================
    trainset, valset = get_dataset(
        data_root=args.data_root,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
        time_resolution=args.time_resolution,
        cache_in_mem=True if args.cache_dataset else False
    )

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=gcn_lstm_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=gcn_lstm_collate_fn,
        pin_memory=True,
    )

    # =====================
    # Model
    # =====================
    model = GCN_LSTM_OD(
        num_nodes=num_nodes,
        edge_index=static_edge_index,
        hid_dim=args.gcn_hidden,
        lstm_layers=args.lstm_layers,
        pred_steps=args.pred_size,
    )

    # =====================
    # Lightning Module
    # =====================
    lm = MetroGCNLSTMLM(
        model=model,
        lr=args.lr,
    )

    # =====================
    # Trainer
    # =====================
    accelerator = resolve_accelerator()
    devices = 1
    strategy = "auto"

    if args.use_ddp:
        strategy = "ddp_find_unused_parameters_true"
        devices = torch.cuda.device_count() if accelerator == "cuda" else os.cpu_count()

    run_name = (
        f"GCN_LSTM_bs{args.batch_size}"
        f"_T{args.window_size}_P{args.pred_size}_"
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )

    logger = WandbLogger(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        logger=logger,
        log_every_n_steps=50,
    )

    trainer.fit(lm, train_loader, val_loader)


if __name__ == "__main__":
    main()
