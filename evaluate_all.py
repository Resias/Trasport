# file: evaluate_all.py

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import argparse

from train_MPGCN import load_static_graphs
from dataset import get_dataset, get_odpair_dataset, get_mpgcn_dataset, get_st_lstm_dataset
from benchmark_Model.TCNbased import TCN_Attention_LSTM
from benchmark_Model.ST_LSTM import STLSTM
from benchmark_Model.MPGCN import MPGCN
from trainer import TCNMetroLM, MetroLM, STLSTMLM, MPGCNLM
from GNN import MetroGNNForecaster


# ===========================================================
# 🔍 1. Metric 계산 함수
# ===========================================================
def compute_metrics(y_true, y_pred):
    """
    y_true, y_pred: numpy array, shape (N, 1) or (N,)
    반환: MSE, RMSE, MAE, MAPE, SMAPE
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-3))) * 100

    # SMAPE 계산
    smape = np.mean(
        2.0 * np.abs(y_true - y_pred) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-3)
    ) * 100

    return mse, rmse, mae, mape, smape


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
    lm: TCNMetroLM = TCNMetroLM.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        model=TCN_Attention_LSTM(
            input_dim=10,
            lstm_hidden=128,
            lstm_layers=3
        ),
        loss=torch.nn.MSELoss(),
        lr=1e-3
    )

    # 실제 예측에 사용할 순수 모델
    model = lm.model.to(device)
    model.eval()

    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"][:, -1, :].to(device)

            pred, _ = model(x)

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
    lm: MetroLM = MetroLM.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        model=MetroGNNForecaster(
            od_df=od_df,
            in_feat=in_feat,
            gnn_hidden=32,
            rnn_hidden=32,
            weekday_emb_dim=16,
            time_emb_dim=time_emb_dim,
            window_size=sample["x_tensor"].shape[0],
            pred_size=sample["y_tensor"].shape[0],
            device=device
        ),
        loss=torch.nn.MSELoss(),
        lr=1e-3
    )

    model = lm.model.to(device)
    model.eval()

    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x_tensor"].to(device)
            weekday = batch["weekday_tensor"].to(device)
            time_hist = batch["time_enc_hist"].to(device)
            time_fut = batch["time_enc_fut"].to(device)

            pred_full = model(x, weekday, time_hist, time_fut)   # (B, T_out, N, N)
            true_full = batch["y_tensor"].to(device)

            pred = pred_full[:, -1, od_i, od_j]
            true = true_full[:, -1, od_i, od_j]

            preds.append(pred.cpu().unsqueeze(-1))
            trues.append(true.cpu().unsqueeze(-1))

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()

    return preds, trues


# ===========================================================
# 🔹 4. STLSTM evaluate
# ===========================================================
def evaluate_stlstm(ckpt_path, dataset, batch_size, num_workers, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    sample = dataset[0]
    input_dim = sample["x"].shape[-1]
    pred_size = sample["y"].shape[0]
    loss = torch.nn.MSELoss()


    print(f"[ST-LSTM] Loading checkpoint: {ckpt_path}")
    model_struct = STLSTM(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=30
    )
    # LightningModule(STLSTMLM) 기준으로 ckpt가 저장되었다고 가정
    lm: STLSTMLM = STLSTMLM.load_from_checkpoint(
        ckpt_path,
        model = model_struct,
        map_location=device,
        loss = loss
    )
    model = lm.model.to(device)
    model.eval()

    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            pred = model(x)
            pred = pred[:, -1].unsqueeze(-1)

            preds.append(pred.cpu())
            trues.append(y[:, -1, :].cpu())

    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


# ===========================================================
# 5. MPGCN Evaluate (★신규 추가★)
# ===========================================================
def evaluate_mpgcn(ckpt, dataset, batch, workers, device, od_i, od_j):
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers)

    sample = dataset[0]
    N = sample["x"].shape[1]   # (T,N,N)

    # 2. Model Init
    model_struct = MPGCN(
        N=N, 
        lstm_h=32, 
        gcn_h=32, 
        gcn_out=16, 
        K=3
    )
    
    # 3. Load & Set Static Graphs
    print("Setting up Static Graphs...")
    adj_O, adj_D, poi_O, poi_D = load_static_graphs("/workspace/od_minute", N)
    
    # Move graphs to device will be handled by Lightning or manually if needed?
    # Actually, model buffers need to be on same device. 
    # Lightning handles submodule parameters but we pass tensors to `set_static_graphs`.
    # We should do this inside `on_fit_start` or pass CPU tensors and let model register them as buffers.
    # Here, for simplicity, we pass them directly. Model will calculate polys and store.
    model_struct.set_static_graphs(adj_O, adj_D, poi_O, poi_D)
    
    lm: MPGCNLM = MPGCNLM.load_from_checkpoint(
        ckpt,
        map_location=device,
        model=model_struct,
        loss=torch.nn.MSELoss(),
        lr=1e-3,
        pred_size=1
    )

    model = lm.model.to(device)
    model.eval()

    preds, trues = [], []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)       # (B, T, N, N)
            y = batch["y"].to(device)       # (B, 1, N, N)

            pred_full = model(x)            # (B, N, N)

            pred = pred_full[:, od_i, od_j].unsqueeze(-1)
            true = y[:, -1, od_i, od_j].unsqueeze(-1)

            preds.append(pred.cpu())
            trues.append(true.cpu())

    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


# ===========================================================
# 5. Main Compare Function
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tcn_ckpt", default="/root/tmp/Trasport/metro-tcn/kqy0m396/checkpoints/epoch=499-step=404000.ckpt")
    parser.add_argument("--gnn_ckpt", default="/root/tmp/Trasport/metro-gnn/u7nf82rf/checkpoints/epoch=299-step=91800.ckpt")
    parser.add_argument("--stlstm_ckpt", default="/root/tmp/Trasport/metro-st-lstm/fn4yc8g8/checkpoints/epoch=499-step=4500.ckpt")
    parser.add_argument("--mpgcn_ckpt", default="/root/tmp/Trasport/metro-mpgcn/c9st2s73/checkpoints/epoch=2-step=39708.ckpt")

    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--test_subdir", default="test")
    parser.add_argument("--od_csv", default="./AD_matrix_trimmed_common.csv")

    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--pred_size", type=int, default=1)
    parser.add_argument("--hop_size", type=int, default=1)

    parser.add_argument("--od_i", type=int, default=10)
    parser.add_argument("--od_j", type=int, default=20)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n=== Loading OD Pair Dataset for TCN ===")
    _, tcn_dataset = get_odpair_dataset(
        data_root=args.data_root,
        train_subdir=args.test_subdir,
        val_subdir=args.test_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
        od_i=args.od_i,
        od_j=args.od_j
    )

    print("\n=== Loading GNN Dataset ===")
    _, gnn_dataset = get_dataset(
        data_root=args.data_root,
        train_subdir=args.test_subdir,
        val_subdir=args.test_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size
    )

    dist_matrix = np.load("dist_matrix.npy")
    W_matrix = np.load("W_matrix.npy")    
    print("\n=== Loading ST-LSTM Dataset ===")
    _, odpair_dataset = get_st_lstm_dataset(
        data_root=args.data_root,
        train_subdir=args.test_subdir,
        val_subdir=args.test_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=1,  # train과 동일
        target_s=10,
        target_e=20,
        top_k=3,
        dist_matrix=dist_matrix,
        W_matrix=W_matrix,
    )

    # MPGCN dataset 추가 로딩
    print("\n=== Loading MPGCN Dataset ===")
    _, mpgcn_dataset = get_mpgcn_dataset(
        data_root=args.data_root,
        train_subdir=args.test_subdir,
        val_subdir=args.test_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size
    )

    results = {}

    # ------------------------------
    # Evaluate TCN
    # ------------------------------
    if args.tcn_ckpt:
        pred, true = evaluate_tcn(args.tcn_ckpt, tcn_dataset, args.batch_size, args.num_workers, device)
        results["TCN"] = compute_metrics(true, pred)

    # ------------------------------
    # Evaluate STLSTM
    # ------------------------------
    if args.stlstm_ckpt:
        pred, true = evaluate_stlstm(args.stlstm_ckpt, odpair_dataset, args.batch_size, args.num_workers, device)
        results["STLSTM"] = compute_metrics(true, pred)

    # ------------------------------
    # Evaluate GNN
    # ------------------------------
    if args.gnn_ckpt:
        pred, true = evaluate_gnn(args.gnn_ckpt, gnn_dataset, args.batch_size, args.num_workers,
                                  device, args.od_i, args.od_j, args.od_csv)
        results["GNN"] = compute_metrics(true, pred)

    # ------------------------------
    # ★ Evaluate MPGCN
    # ------------------------------
    if args.mpgcn_ckpt:
        pred, true = evaluate_mpgcn(args.mpgcn_ckpt, mpgcn_dataset, args.batch_size, args.num_workers,
                                    device, args.od_i, args.od_j)
        results["MPGCN"] = compute_metrics(true, pred)

    # -----------------------------------------------------
    # Print Results
    # -----------------------------------------------------
    print("\n=================================================")
    print("        MODEL PERFORMANCE COMPARISON")
    print("=================================================")

    metrics_name = ["MSE", "RMSE", "MAE", "MAPE", "SMAPE"]

    for model_name, metric in results.items():
        print(f"\n📌 {model_name}")
        for n, v in zip(metrics_name, metric):
            print(f" {n:<6}: {v:.4f}")

    print("\n=================================================")


if __name__ == "__main__":
    main()