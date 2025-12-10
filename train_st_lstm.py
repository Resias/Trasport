# file: train_st_lstm.py
import argparse
import os
import torch
import pandas as pd
import numpy as np
import pytorch_lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from dataset import get_st_lstm_dataset
from benchmark_Model.ST_LSTM import STLSTM
from trainer import STLSTMLM

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def resolve_accelerator():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="Train ST-LSTM model with Metro pipeline")

    # 데이터 경로 및 설정
    parser.add_argument("--data_root", default="/workspace/od_minute")
    parser.add_argument("--train_subdir", default="train")
    parser.add_argument("--val_subdir", default="test")
    
    # 행렬 파일 경로 (없으면 Dummy 생성)
    parser.add_argument("--dist_npy", default="./dist_matrix.npy", help="Path to distance matrix .npy file")
    parser.add_argument("--w_npy", default="./W_matrix.npy", help="Path to travel time limit W matrix .npy file")

    # 시계열 윈도우 설정
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=10)

    # ST-LSTM 타겟 설정
    parser.add_argument("--target_s", type=int, default=10, help="Start Station ID")
    parser.add_argument("--target_e", type=int, default=25, help="End Station ID")
    parser.add_argument("--top_k", type=int, default=3, help="Number of neighbor ODs")

    # 학습 설정
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--max_epochs", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)

    # 로깅 설정
    parser.add_argument("--wandb_project", default="metro-st-lstm")
    parser.add_argument("--wandb_log_model", action="store_true")

    return parser.parse_args()


def load_auxiliary_matrices(args, train_subdir_path):
    """
    거리 행렬(Dist)과 이동 시간 행렬(W)을 로드하거나 더미를 생성합니다.
    """
    # N(역 개수) 확인을 위해 샘플 파일 하나 로드
    try:
        sample_files = sorted(os.listdir(train_subdir_path))
        sample_path = os.path.join(train_subdir_path, sample_files[0])
        temp_data = np.load(sample_path)
        N = temp_data.shape[1]
    except Exception as e:
        print(f"Error loading sample data to determine N: {e}")
        N = 100 # Fallback

    # 1. Distance Matrix
    if args.dist_npy and os.path.exists(args.dist_npy):
        print(f"Loading Dist Matrix from {args.dist_npy}")
        dist_matrix = np.load(args.dist_npy)
    else:
        print("Warning: Distance matrix not found or not provided. Using Random Dummy Matrix.")
        dist_matrix = np.random.rand(N, N) * 10  # Dummy

    # 2. W Matrix
    if args.w_npy and os.path.exists(args.w_npy):
        print(f"Loading W Matrix from {args.w_npy}")
        W_matrix = np.load(args.w_npy)
    else:
        print("Warning: W matrix not found or not provided. Using Random Dummy Matrix.")
        W_matrix = np.random.randint(5, 30, size=(N, N)) # Dummy
    
    return dist_matrix, W_matrix


def main():
    args = parse_args()
    DEVICE = resolve_accelerator()

    train_path = os.path.join(args.data_root, args.train_subdir)
    loss = torch.nn.MSELoss()

    # 1. 행렬 로드 (Dist, W)
    dist_matrix, W_matrix = load_auxiliary_matrices(args, train_path)

    print(f"=== Training ST-LSTM for OD: {args.target_s} -> {args.target_e} ===")

    # 2. 데이터셋 생성 (get_st_lstm_dataset 활용)
    # 이 함수 내부에서 SpatialCorrelationSelector가 동작하여 이웃을 자동 선정함
    trainset, valset = get_st_lstm_dataset(
        data_root=args.data_root,
        train_subdir=args.train_subdir,
        val_subdir=args.val_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
        target_s=args.target_s,
        target_e=args.target_e,
        top_k=args.top_k,
        dist_matrix=dist_matrix,
        W_matrix=W_matrix
    )

    # 3. DataLoader 설정
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

    # 4. 입력 차원 자동 감지
    # ST-LSTM의 Input Dim은 이웃 수(Top-k) 등에 따라 달라지므로 샘플에서 확인하는 것이 안전함
    sample = trainset[0]
    input_dim = sample["x"].shape[-1]
    print(f"Detected input_dim: {input_dim} (Based on Top-K={args.top_k})")

    # 5. 모델 초기화
    model = STLSTM(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=args.pred_size
    )

    lightning_module = STLSTMLM(
        model=model,
        lr=args.lr,
        pred_size=args.pred_size,
        loss = loss
    )

    # 6. Logger 및 Trainer 설정
    wandb_logger = WandbLogger(project=args.wandb_project, log_model=args.wandb_log_model)
    # Wandb Config 업데이트
    wandb_logger.experiment.config.update(vars(args))

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=DEVICE,
        devices=1,
        logger=wandb_logger,
        log_every_n_steps=50,
        # precision="16-mixed" # 필요시 활성화
    )

    print("Start training...")
    trainer.fit(lightning_module, train_loader, val_loader)


if __name__ == "__main__":
    main()