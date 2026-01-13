# train/train_graph.py
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

from dataset import get_dataset, graph_collate_fn, graph_week_collate_fn
from torch_geometric.utils import dense_to_sparse

from models.GATransformerdecoder import GATTransformerOD, GATTransformerODWeek
from trainer import MetroGraphLM, MetroGraphWeekLM

def resolve_accelerator():
    return "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--train_subdir", default="train")
    parser.add_argument("--val_subdir", default="test")
    parser.add_argument("--cache_dataset", action="store_true")
    parser.add_argument("--od_csv", default="./AD_matrix_trimmed_common.csv")
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--node_feat_dim", type=int, default=1)
    parser.add_argument("--decode_num_layers", type=int, default=2)
    parser.add_argument("--use_weekday", action="store_true")
    parser.add_argument("--use_ddp", action="store_true", help="Enable Distributed Data Parallel training")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--gat_hidden", type=int, default=64)
    parser.add_argument("--wandb_project", default="metro-gnn-od")
    args = parser.parse_args()

    # =====================
    # Static edge_index 로딩
    # =====================
    od_df = pd.read_csv(args.od_csv, index_col=0)
    adj = torch.tensor(od_df.values, dtype=torch.float32)
    static_edge_index, _ = dense_to_sparse(adj)

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
        cache_in_mem=True if args.cache_dataset else False
    )

    # =====================
    # Dataloaders
    # =====================
    if not args.use_weekday:
        train_loader = DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            persistent_workers=True,
            collate_fn=lambda batch: graph_collate_fn(batch, static_edge_index),
            pin_memory=True,
        )

        val_loader = DataLoader(
            valset,
            batch_size=args.batch_size,
            shuffle=False,
            persistent_workers=True,
            num_workers=args.num_workers,
            collate_fn=lambda batch: graph_collate_fn(batch, static_edge_index),
            pin_memory=True,
        )
        model = GATTransformerOD(
            num_nodes=static_edge_index.max().item() + 1,
            node_feat_dim=args.node_feat_dim,
            heads=args.gat_heads,
            gat_hid_dim=args.gat_hidden,
            num_future_steps=args.pred_size,
            decode_num_layers=args.decode_num_layers
        )

        model.static_edge_index = static_edge_index
    else:
        train_loader = DataLoader(
            trainset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            persistent_workers=True,
            collate_fn=lambda batch: graph_week_collate_fn(batch, static_edge_index),
            pin_memory=True,
        )
        val_loader = DataLoader(
            valset,
            batch_size=args.batch_size,
            shuffle=False,
            persistent_workers=True,
            num_workers=args.num_workers,
            collate_fn=lambda batch: graph_week_collate_fn(batch, static_edge_index),
            pin_memory=True,
        )
        model = GATTransformerODWeek(
            num_nodes=static_edge_index.max().item() + 1,
            heads=args.gat_heads,
            node_feat_dim=args.node_feat_dim,
            gat_hid_dim=args.gat_hidden,
            num_future_steps=args.pred_size,
            decode_num_layers=args.decode_num_layers
        )

        model.static_edge_index = static_edge_index

    # =====================
    # Lightning Module
    # =====================
    lm = MetroGraphWeekLM(
        model=model,
        loss=torch.nn.MSELoss(),
        lr=args.lr
    )

    # =====================
    # Trainer
    # =====================
    accelerator = resolve_accelerator()

    # 기본 strategy
    strategy = "auto"
    devices = 1
    
    if args.use_ddp:
        # DDP 설정
        strategy = "ddp_find_unused_parameters_true"
        if accelerator == "cuda":
            # GPU 여러 개일 때
            devices = torch.cuda.device_count()
        else:
            # CPU DDP
            devices = os.cpu_count()
    model_type = "GATTransformerODWeek" if args.use_weekday else "GATTransformerOD"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_type}_wd{args.use_weekday}_bs{args.batch_size}_T{args.window_size}_P{args.pred_size}_{timestamp}"

    logger = WandbLogger(project=args.wandb_project, name=run_name, config=vars(args))

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
