# file: evaluate_all.py

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import argparse

from dataset import get_dataset, get_odpair_dataset
from benchmark_Model.TCNbased import TCN_Attention_LSTM
from trainer import TCNMetroLM, MetroLM
from GNN import MetroGNNForecaster


# ---------------------------------------
# Metric 계산 함수
# ---------------------------------------
def compute_metrics(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-3))) * 100

    return mse, rmse, mae, mape


# ---------------------------------------
# TCN evaluate
# ---------------------------------------
def evaluate_tcn(ckpt_path, dataset, batch_size, num_workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # dataset sample → input_dim
    sample = dataset[0]
    input_dim = sample["x"].shape[-1]

    # 모델 정의 (Lightning checkpoint 로딩)
    print(f"[TCN] Loading checkpoint: {ckpt_path}")
    model_module: TCNMetroLM = TCNMetroLM.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        model=TCN_Attention_LSTM(
            input_dim=input_dim,
            lstm_hidden=128,
            lstm_layers=3
        ),
        loss=torch.nn.MSELoss(),
        lr=1e-3
    )

    model_module.to(device)
    model_module.eval()

    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"][:, -1, :].to(device)

            pred, _ = model_module.model(x)

            preds.append(pred.cpu())
            trues.append(y.cpu())

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    return preds, trues


# ---------------------------------------
# GNN evaluate
# ---------------------------------------
def evaluate_gnn(ckpt_path, dataset, batch_size, num_workers, device, od_i, od_j, od_csv):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    sample = dataset[0]
    in_feat = sample["x_tensor"].shape[-1]
    time_emb_dim = sample["time_enc_hist"].shape[-1]

    # adjacency matrix
    od_df = pd.read_csv(od_csv, index_col=0)
    od_df = od_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"[GNN] Loading checkpoint: {ckpt_path}")
    model_module: MetroLM = MetroLM.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        model=MetroGNNForecaster(
            od_df=od_df,
            in_feat=in_feat,
            gnn_hidden=64,
            rnn_hidden=64,
            weekday_emb_dim=8,
            time_emb_dim=time_emb_dim,
            window_size=sample["x_tensor"].shape[0],
            pred_size=sample["y_tensor"].shape[0],
            device=device
        ),
        loss=torch.nn.MSELoss(),
        lr=1e-3
    )

    model_module.to(device)
    model_module.eval()

    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x_tensor"].to(device)
            weekday = batch["weekday_tensor"].to(device)
            time_hist = batch["time_enc_hist"].to(device)
            time_fut = batch["time_enc_fut"].to(device)

            pred_full = model_module.model(x, weekday, time_hist, time_fut)   # (B, T_out, N, N)
            true_full = batch["y_tensor"].to(device)

            pred = pred_full[:, -1, od_i, od_j]
            true = true_full[:, -1, od_i, od_j]

            preds.append(pred.cpu().unsqueeze(-1))
            trues.append(true.cpu().unsqueeze(-1))

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    return preds, trues


# ---------------------------------------
# Main: TCN + GNN 성능 비교
# ---------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tcn_ckpt", required=True)
    parser.add_argument("--gnn_ckpt", required=True)

    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--test_subdir", default="test")
    parser.add_argument("--od_csv", default="./AD_matrix_trimmed_common.csv")

    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--pred_size", type=int, default=1)
    parser.add_argument("--hop_size", type=int, default=1)

    parser.add_argument("--od_i", type=int, default=10)
    parser.add_argument("--od_j", type=int, default=20)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=== Loading Datasets ===")
    _, tcn_dataset  = get_odpair_dataset(
        data_root=args.data_root,
        train_subdir=args.test_subdir,
        val_subdir=args.test_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
        od_i=args.od_i,
        od_j=args.od_j
    )

    _, gnn_dataset = get_dataset(
        data_root=args.data_root,
        train_subdir=args.test_subdir,
        val_subdir=args.test_subdir,
        window_size=60,
        hop_size=10,
        pred_size=30
    )

    print("Evaluating TCN...")
    tcn_pred, tcn_true = evaluate_tcn(
        args.tcn_ckpt,
        tcn_dataset,
        args.batch_size,
        args.num_workers,
        device
    )

    print("Evaluating GNN...")
    gnn_pred, gnn_true = evaluate_gnn(
        args.gnn_ckpt,
        gnn_dataset,
        args.batch_size,
        args.num_workers,
        device,
        args.od_i,
        args.od_j,
        args.od_csv
    )

    # Metrics
    tcn_mse, tcn_rmse, tcn_mae, tcn_mape = compute_metrics(tcn_true, tcn_pred)
    gnn_mse, gnn_rmse, gnn_mae, gnn_mape = compute_metrics(gnn_true, gnn_pred)

    print("\n\n========================================")
    print("        🔥 MODEL PERFORMANCE COMPARE 🔥")
    print("========================================")
    print(f"Target OD Pair = ({args.od_i}, {args.od_j})\n")

    print("📌 TCN Model Results")
    print(f" MSE  : {tcn_mse:.4f}")
    print(f" RMSE : {tcn_rmse:.4f}")
    print(f" MAE  : {tcn_mae:.4f}")
    print(f" MAPE : {tcn_mape:.2f}%\n")

    print("📌 GNN Model Results")
    print(f" MSE  : {gnn_mse:.4f}")
    print(f" RMSE : {gnn_rmse:.4f}")
    print(f" MAE  : {gnn_mae:.4f}")
    print(f" MAPE : {gnn_mape:.2f}%")
    print("========================================")


if __name__ == "__main__":
    main()
