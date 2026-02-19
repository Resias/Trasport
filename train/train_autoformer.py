# train/train_autoformer_od.py
import argparse
import datetime
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import get_dataset
from trainer import MetroAutoformerODLM
from SCIE_Benchmark.Autoformer import AutoformerODFormal


def resolve_accelerator():
    return "cuda" if torch.cuda.is_available() else "cpu"


def autoformer_collate_fn(batch):
    """
    batch: list of MetroDataset samples
    returns dict (LightningModule forward와 1:1 대응)
    """
    return {
        "x": torch.stack([b["x_tensor"] for b in batch], dim=0),
        "y": torch.stack([b["y_tensor"] for b in batch], dim=0),
        "weekday": torch.stack([b["weekday_tensor"] for b in batch], dim=0),
        "time_hist": torch.stack([b["time_enc_hist"] for b in batch], dim=0),
        "time_fut": torch.stack([b["time_enc_fut"] for b in batch], dim=0),
    }


def main():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--train_subdir", default="train")
    parser.add_argument("--val_subdir", default="test")
    parser.add_argument("--cache_dataset", action="store_true")

    parser.add_argument("--time_resolution", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=10)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=200)

    # Model
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--ff_dim", type=int, default=256)
    parser.add_argument("--enc_layers", type=int, default=2)
    parser.add_argument("--dec_layers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--kernel_size", type=int, default=25)

    # System
    parser.add_argument("--use_ddp", action="store_true")
    parser.add_argument("--wandb_project", default="metro-AutoformerOD")

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
        time_resolution=args.time_resolution,
        cache_in_mem=args.cache_dataset,
    )

    sample = trainset[0]
    num_nodes = sample["x_tensor"].shape[-1]
    time_dim = sample["time_enc_hist"].shape[-1]

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=autoformer_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=autoformer_collate_fn,
        pin_memory=True,
    )

    # =====================
    # Model
    # =====================
    model = AutoformerODFormal(
        num_nodes=num_nodes,
        d_model=args.d_model,
        ff_dim=args.ff_dim,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        pred_steps=args.pred_size,
        rank=args.rank,
        kernel_size=args.kernel_size,
        top_k=args.top_k,
        time_dim=time_dim,
        use_weekday=True,
    )

    lm = MetroAutoformerODLM(
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
        devices = torch.cuda.device_count()

    run_name = (
        f"AutoformerOD_bs{args.batch_size}"
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
