# train/train_graph.py
import argparse
import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import get_dataset
from models.GNN import MetroGNNForecaster
from trainer import MetroGraphLM
from dataset import graph_collate_fn
from torch_geometric.utils import dense_to_sparse


def resolve_accelerator():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/workspace/od_minute")
    parser.add_argument("--train_subdir", default="train")
    parser.add_argument("--val_subdir", default="test")
    parser.add_argument("--od_csv", default="./AD_matrix_trimmed_common.csv")
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--gnn_hidden", type=int, default=64)
    parser.add_argument("--wandb_project", default="metro-gnn-graph")
    args = parser.parse_args()

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
    )

    # =====================
    # Static edge_index
    # =====================
    od_df = pd.read_csv(args.od_csv, index_col=0)
    adj = torch.tensor(od_df.values, dtype=torch.float32)
    edge_index, _ = dense_to_sparse(adj)

    # =====================
    # Dataloaders
    # =====================
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: graph_collate_fn(batch, edge_index),
        pin_memory=True,
    )

    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: graph_collate_fn(batch, edge_index),
        pin_memory=True,
    )

    # =====================
    # Model
    # =====================
    model = MetroGNNForecaster(
        num_nodes=edge_index.max().item() + 1,
        node_feat_dim=1,
        gat_hid_dim=args.gnn_hidden,
        num_future_steps=args.pred_size,
    )

    # edge_index를 모델에 보관
    model.edge_index = edge_index

    lm = MetroGraphLM(
        model=model,
        loss=torch.nn.MSELoss(),
        lr=args.lr
    )

    # =====================
    # Trainer
    # =====================
    logger = WandbLogger(project=args.wandb_project)

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=resolve_accelerator(),
        devices=1,
        logger=logger,
        log_every_n_steps=50,
    )

    trainer.fit(lm, train_loader, val_loader)


if __name__ == "__main__":
    main()