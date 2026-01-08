from dataset import get_dataset
from models.GNN import MetroGNNForecaster
from train.trainer import MetroLM

import argparse
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def resolve_accelerator():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train Metro GNN forecaster")
    parser.add_argument("--data_root", default="/workspace/od_minute", help="root directory with train/val data")
    parser.add_argument("--train_subdir", default="train", help="subdirectory name for training set")
    parser.add_argument("--val_subdir", default="test", help="subdirectory name for validation set")
    parser.add_argument("--od_csv", default="./AD_matrix_trimmed_common.csv", help="path to adjacency csv")
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_epochs", type=int, default=300)
    parser.add_argument("--gnn_hidden", type=int, default=64)
    parser.add_argument("--rnn_hidden", type=int, default=64)
    parser.add_argument("--weekday_emb_dim", type=int, default=8)
    parser.add_argument("--wandb_project", default="metro-gnn")
    parser.add_argument("--wandb_log_model", action="store_true", help="log artifacts to wandb")
    return parser.parse_args()


def main():
    args = parse_args()

    loss = torch.nn.MSELoss()

    train_path = os.path.join(args.data_root, args.train_subdir)
    val_path = os.path.join(args.data_root, args.val_subdir)
    print('Loading Dataset')
    trainset, valset = get_dataset(
        data_root=args.data_root,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size
    )
    
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    # od_df = pd.read_csv(args.od_csv, index_col='index_col')
    od_df = pd.read_csv(args.od_csv, index_col=0)
    od_df = od_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    sample = valset[0]
    in_feat = sample['x_tensor'].shape[-1]
    time_emb_dim = sample['time_enc_hist'].shape[-1]

    gnn_model = MetroGNNForecaster(
        od_df=od_df,
        in_feat=in_feat,
        gnn_hidden=args.gnn_hidden,
        rnn_hidden=args.rnn_hidden,
        weekday_emb_dim=args.weekday_emb_dim,
        time_emb_dim=time_emb_dim,
        window_size=args.window_size,
        pred_size=args.pred_size,
        device="cpu",
    )

    metroLM = MetroLM(
        model=gnn_model,
        loss=loss,
        lr=args.lr
    )
    accelerator = resolve_accelerator()
    wandb_logger = WandbLogger(project=args.wandb_project, log_model=args.wandb_log_model)
    wandb_logger.experiment.config.update({
        "window_size": args.window_size,
        "pred_size": args.pred_size,
        "hop_size": args.hop_size,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_files": train_path,
        "val_files": val_path,
        "gnn_hidden": args.gnn_hidden,
        "rnn_hidden": args.rnn_hidden,
        "weekday_emb_dim": args.weekday_emb_dim,
    })
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=accelerator,
        devices=1,
        log_every_n_steps=50,
        logger=wandb_logger,
        # precision="16-mixed",   # <- 추가
    )
    trainer.fit(metroLM, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
