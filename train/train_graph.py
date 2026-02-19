# train/train_graph.py
import argparse
import datetime
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import get_dataset, graph_week_collate_fn
from torch_geometric.utils import dense_to_sparse

from models.GATransformerdecoder import GATTransformerODWeek
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
    parser.add_argument("--station_latlon_csv", type=str, default="./ad_station_latlon.csv", help="CSV file with columns: station_ad, lat, lon")
    parser.add_argument("--time_resolution", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gat_heads", type=int, default=4)
    parser.add_argument("--node_feat_dim", type=int, default=16)
    parser.add_argument("--decode_num_layers", type=int, default=2)
    parser.add_argument("--use_ddp", action="store_true", help="Enable Distributed Data Parallel training")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--gat_hidden", type=int, default=64)
    parser.add_argument("--wandb_project", default="metro-GATrasformer-od-week-latlon")
    args = parser.parse_args()

    # =====================
    # Static edge_index 로딩
    # =====================
    od_df = pd.read_csv(args.od_csv, index_col=0)
    adj = torch.tensor(od_df.values, dtype=torch.float32)
    static_edge_index, _ = dense_to_sparse(adj)

    # =====================
    # Load station lat/lon (JSON mapping)
    # =====================
    node_latlon = None

    if args.station_latlon_csv is not None:
        # lat/lon CSV
        latlon_raw = pd.read_csv(args.station_latlon_csv)  # station_ad, lat, lon

        # station -> idx json
        with open("station_to_idx.json", "r", encoding="utf-8") as f:
            station_to_idx = json.load(f)


        # ---- 핵심: full node frame 생성 ----
        full_nodes = pd.DataFrame({
            "station_ad": list(station_to_idx.keys()),
            "node_id": list(station_to_idx.values())
        })

        # lat/lon left merge (없는 역도 row 생성됨)
        latlon_df = full_nodes.merge(latlon_raw, on="station_ad", how="left")
        # adjacency numpy (이미 od_df 있음)
        adj_np = od_df.values

        # missing lat/lon 처리
        missing = latlon_df[latlon_df["lat"].isna() | latlon_df["lon"].isna()]

        if len(missing) > 0:
            print(f"Filling {len(missing)} missing station(s) by neighbor mean")

            for _, row in missing.iterrows():
                nid = int(row["node_id"])

                neighbors = (adj_np[nid] > 0).nonzero()[0]

                assert len(neighbors) > 0, f"No neighbors for node {nid}"

                neigh_latlon = latlon_df.iloc[neighbors][["lat", "lon"]].values

                mean_lat = neigh_latlon[:, 0].mean()
                mean_lon = neigh_latlon[:, 1].mean()

                latlon_df.loc[latlon_df["node_id"] == nid, "lat"] = mean_lat
                latlon_df.loc[latlon_df["node_id"] == nid, "lon"] = mean_lon

                print(f"Filled node {nid} using {len(neighbors)} neighbors")

        # sanity
        assert latlon_df["node_id"].isnull().sum() == 0
        assert latlon_df["lat"].isnull().sum() == 0
        assert latlon_df["lon"].isnull().sum() == 0

        # idx 기준 정렬
        latlon_df = latlon_df.sort_values("node_id")

        latlon = torch.tensor(
            latlon_df[["lat", "lon"]].values,
            dtype=torch.float32
        )

        # normalization
        latlon_min = latlon.min(dim=0, keepdim=True)[0]
        latlon_max = latlon.max(dim=0, keepdim=True)[0]
        latlon = (latlon - latlon_min) / (latlon_max - latlon_min + 1e-6)

        node_latlon = latlon
    
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

    # =====================
    # Dataloaders
    # =====================
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=False,
        collate_fn=lambda batch: graph_week_collate_fn(batch, static_edge_index),
        pin_memory=True,
    )
    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        persistent_workers=False,
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
        decode_num_layers=args.decode_num_layers,
        node_latlon=node_latlon
    )

    model.static_edge_index = static_edge_index

    # =====================
    # Lightning Module
    # =====================
    lm = MetroGraphWeekLM(
        model=model,
        loss=torch.nn.SmoothL1Loss(),
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
    model_type = "GATTransformerODWeek"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{model_type}_bs{args.batch_size}_T{args.window_size}_P{args.pred_size}_{timestamp}"

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
