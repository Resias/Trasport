# train_st_damhgn.py
import argparse
import torch
import pytorch_lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import get_stdamhgn_dataset
from SCIE_Benchmark.STDAMHGN import STDAMHGN
from trainer import STDAMHGNLitModule


def resolve_accelerator():
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser("Train Paper-faithful ST-DAMHGN")

    # -------------------------
    # Data
    # -------------------------
    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--train_subdir", default="train")
    parser.add_argument("--val_subdir", default="test")
    parser.add_argument("--hypergraph_path", required=True)

    # -------------------------
    # Temporal setting
    # -------------------------
    parser.add_argument("--m", type=int, default=6, help="tendency length")
    parser.add_argument("--n", type=int, default=3, help="periodicity length")
    parser.add_argument("--hop_size", type=int, default=20)

    # -------------------------
    # Training
    # -------------------------
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=7.5e-4)
    parser.add_argument("--max_epochs", type=int, default=200)

    # -------------------------
    # Model
    # -------------------------
    parser.add_argument("--hidden_dim", type=int, default=64)

    # -------------------------
    # Logging
    # -------------------------
    parser.add_argument("--wandb_project", default="metro-st-damhgn")
    parser.add_argument("--wandb_log_model", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    DEVICE = resolve_accelerator()

    print("=== Train ST-DAMHGN ===")

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    trainset, valset = get_stdamhgn_dataset(
        data_root=args.data_root,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        hypergraph_path=args.hypergraph_path,
        m=args.m,
        n=args.n,
        hop_size=args.hop_size
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
    hg = torch.load(args.hypergraph_path)
    hypergraphs = [
        hg["tendency_closest"],
        hg["tendency_cluster"],
        hg["poi_closest"],
        hg["poi_cluster"]
    ]

    model = STDAMHGN(
        num_vertices=len(hg["valid_od_pairs"]),
        m=args.m,
        n=args.n,
        hid_dim=args.hidden_dim,
        hypergraphs=hypergraphs
    )

    lightning_module = STDAMHGNLitModule(
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
    L.seed_everything(42, workers=True)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=25,
            min_delta=1e-4,
            mode="min"
        ),
        ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min"
        )
    ]

    trainer = L.Trainer(
        accelerator=DEVICE,
        devices=1,
        precision=32,
        max_epochs=200,
        callbacks=callbacks,
        logger=wandb_logger,
    )

    trainer.fit(lightning_module, train_loader, val_loader)


if __name__ == "__main__":
    main()
