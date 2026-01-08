import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import argparse
import os

# 기존 모듈 임포트
from train_MPGCN import load_static_graphs
from dataset import get_dataset, get_odpair_dataset, get_mpgcn_dataset, get_st_lstm_dataset
from benchmark_Model.TCNbased import TCN_Attention_LSTM
from benchmark_Model.ST_LSTM import STLSTM
from benchmark_Model.MPGCN import MPGCN
from train.trainer import TCNMetroLM, MetroLM, STLSTMLM, MPGCNLM
from models.GNN import MetroGNNForecaster

# ===========================================================
# 🎯 평가 설정: 테스트할 OD 리스트
# ===========================================================
# [중요] 여기에 평가하고 싶은 OD 쌍들을 넣으세요.
# 반드시 Local Model이 학습한 OD(예: 10->20)가 포함되어야 합니다.
TARGET_OD_LIST = [
    # (523, 10),      # st-lstm
    # (523, 419),     # st-lstm
    # (20, 10),       # st-lstm
    # (25, 10),       # test용
    (10, 20)
]

# ===========================================================
# 1. Metric 계산 함수
# ===========================================================
def compute_metrics(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    diff = y_true - y_pred

    # --- Global MAPE (WAPE) ---
    numerator = np.sum(np.abs(diff))
    denominator = np.sum(np.abs(y_true)) + 1e-8
    global_mape = (numerator / denominator) * 100

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # 0으로 나누기 방지
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
        smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-3)) * 100
    
    return mse, rmse, mae, global_mape, smape



# ===========================================================
# 2. Global Model 로더 및 추론 함수 (메모리 효율화)
# ===========================================================
def load_metrognn_model(ckpt_path, dataset, device, od_csv):
    sample = dataset[0]
    in_feat = sample["x_tensor"].shape[-1]
    time_emb_dim = sample["time_enc_hist"].shape[-1]

    od_df = pd.read_csv(od_csv, index_col=0)
    od_df = od_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"[MetroGNN] Loading Global Model from {ckpt_path} ...")
    lm = MetroLM.load_from_checkpoint(
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
    return lm.model.to(device)

def run_metrognn_inference(model, dataset, device, od_i, od_j, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for batch in loader:
            x = batch["x_tensor"].to(device)
            weekday = batch["weekday_tensor"].to(device)
            time_hist = batch["time_enc_hist"].to(device)
            time_fut = batch["time_enc_fut"].to(device)
            y = batch["y_tensor"].to(device)

            # Global Inference (B, T, N, N)
            pred_full = model(x, weekday, time_hist, time_fut)
            
            # Local Slicing (B, T, 1, 1)
            p = pred_full[:, -1, od_i, od_j].unsqueeze(-1)
            t = y[:, -1, od_i, od_j].unsqueeze(-1)
            
            preds.append(p.cpu())
            trues.append(t.cpu())
            
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()

def load_mpgcn_model(ckpt_path, dataset, device):
    sample = dataset[0]
    N = sample["x"].shape[2]
    
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
    
    print(f"[MPGCN] Loading Global Model from {ckpt_path} ...")
    lm = MPGCNLM.load_from_checkpoint(
        ckpt_path,
        map_location=device,
        model=model_struct,
        loss=torch.nn.MSELoss(),
        lr=1e-3,
        pred_size=1
    )
    return lm.model.to(device)

def run_mpgcn_inference(model, dataset, device, od_i, od_j, batch_size, num_workers):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device) # (B, 1, N, N)
            
            # Global Inference
            pred_full = model(x) # (B, N, N)
            
            p = pred_full[:, od_i, od_j].unsqueeze(-1)
            t = y[:, -1, od_i, od_j].unsqueeze(-1)
            
            preds.append(p.cpu())
            trues.append(t.cpu())
            
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()

# ===========================================================
# 3. Local Model 평가 함수 (그대로 사용)
# ===========================================================
def evaluate_tcn_wrapper(ckpt, data_root, test_subdir, win, hop, pred, od_i, od_j, batch, workers, device):
    # Dataset 로드
    _, ds = get_odpair_dataset(data_root, test_subdir, test_subdir, win, hop, pred, od_i, od_j)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    
    # Model 로드
    lm = TCNMetroLM.load_from_checkpoint(
        ckpt, map_location=device,
        model=TCN_Attention_LSTM(input_dim=10, lstm_hidden=128, lstm_layers=3), # 설정 확인
        loss=torch.nn.MSELoss(),
        lr=1e-3
    )
    model = lm.model.to(device)
    model.eval()
    
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"][:, -1, :].to(device)
            p, _ = model(x)
            preds.append(p.cpu())
            trues.append(y.cpu())
            
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()

def evaluate_stlstm_wrapper(ckpt, data_root, test_subdir, win, hop, pred, od_i, od_j, batch, workers, device, dist, W):
    # Dataset 로드
    _, ds = get_st_lstm_dataset(data_root, test_subdir, test_subdir, win, hop, pred, 
                                target_s=od_i, target_e=od_j, top_k=3, dist_matrix=dist, W_matrix=W)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    
    sample = ds[0]
    input_dim = sample["x"].shape[-1]
    loss = torch.nn.MSELoss()
    
    # Model 로드
    
    model_struct = STLSTM(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=2,
        output_dim=30
    )
    # LightningModule(STLSTMLM) 기준으로 ckpt가 저장되었다고 가정
    lm: STLSTMLM = STLSTMLM.load_from_checkpoint(
        ckpt,
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
            p = model(x)
            p = p[:, -1].unsqueeze(-1)
            preds.append(p.cpu())
            trues.append(y[:, -1, :].cpu())
            
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()


# ===========================================================
# 4. Main Execution
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser()
    
    # Checkpoints
    parser.add_argument("--tcn_ckpt", default="", help="Path to TCN checkpoint")
    parser.add_argument("--stlstm_ckpt", default="", help="Path to ST-LSTM checkpoint")
    parser.add_argument("--gnn_ckpt", default="", help="Path to MetroGNN checkpoint")
    parser.add_argument("--mpgcn_ckpt", default="", help="Path to MPGCN checkpoint")
    
    # Data & Settings
    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--test_subdir", default="test")
    parser.add_argument("--od_csv", default="./AD_matrix_trimmed_common.csv")
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--pred_size", type=int, default=1)
    parser.add_argument("--hop_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)
    
    # [Local Model Info] 이 모델들이 학습된 특정 OD 정보
    parser.add_argument("--local_od_i", type=int, default=10, help="Start station TCN/STLSTM was trained on")
    parser.add_argument("--local_od_j", type=int, default=20, help="End station TCN/STLSTM was trained on")
    
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=========================================================")
    print(f"🚀 Evaluation Started | Device: {device}")
    print(f"   Local Model Target: {args.local_od_i} -> {args.local_od_j}")
    print(f"   Total Test Pairs  : {len(TARGET_OD_LIST)}")
    print("=========================================================\n")

    # 1. Load Auxiliary Matrices for ST-LSTM (한 번만 로드)
    dist_matrix = np.load("./dist_matrix.npy") if os.path.exists("./dist_matrix.npy") else None
    W_matrix = np.load("./W_matrix.npy") if os.path.exists("./W_matrix.npy") else None

    # 2. Load Global Models (한 번만 로드하여 메모리에 상주)
    metro_gnn = None
    metro_ds_template = None
    if args.gnn_ckpt:
        _, metro_ds_template = get_dataset(args.data_root, args.test_subdir, args.test_subdir, 
                                           args.window_size, args.hop_size, args.pred_size)
        metro_gnn = load_metrognn_model(args.gnn_ckpt, metro_ds_template, device, args.od_csv)

    mpgcn = None
    mpgcn_ds_template = None
    if args.mpgcn_ckpt:
        _, mpgcn_ds_template = get_mpgcn_dataset(args.data_root, args.test_subdir, args.test_subdir,
                                                 args.window_size, args.hop_size, args.pred_size)
        mpgcn = load_mpgcn_model(args.mpgcn_ckpt, mpgcn_ds_template, device)

    # 3. Evaluation Loop
    results = {
        "TCN": [], "ST-LSTM": [], 
        "MetroGNN": [], "MPGCN": []
    }
    
    local_target_metrics = {} # 10->20 에서의 결과만 따로 저장 (Head-to-Head 비교용)

    for idx, (s, e) in enumerate(TARGET_OD_LIST):
        print(f"\n[Step {idx+1}/{len(TARGET_OD_LIST)}] Evaluating Pair: {s} -> {e}")
        
        is_local_target = (s == args.local_od_i and e == args.local_od_j)
        
        # --- (A) Local Models Evaluation ---
        if is_local_target:
            print("   👉 Match found for Local Models! Running TCN & ST-LSTM...")
            
            if args.tcn_ckpt:
                p, t = evaluate_tcn_wrapper(args.tcn_ckpt, args.data_root, args.test_subdir,
                                            args.window_size, args.hop_size, args.pred_size,
                                            s, e, args.batch_size, args.num_workers, device)
                m = compute_metrics(t, p)
                results["TCN"].append(m)
                local_target_metrics["TCN"] = m
                print(f"      [TCN] RMSE: {m[1]:.4f} MAPE: {m[3]:.4f}")

            if args.stlstm_ckpt:
                p, t = evaluate_stlstm_wrapper(args.stlstm_ckpt, args.data_root, args.test_subdir,
                                               args.window_size, args.hop_size, args.pred_size,
                                               s, e, args.batch_size, args.num_workers, device, dist_matrix, W_matrix)
                m = compute_metrics(t, p)
                results["ST-LSTM"].append(m)
                local_target_metrics["ST-LSTM"] = m
                print(f"      [ST-LSTM] RMSE: {m[1]:.4f} MAPE: {m[3]:.4f}")
        else:
            print("   Running Global Models only (Local models skipped)...")

        # --- (B) Global Models Evaluation ---
        # MetroGNN
        if metro_gnn:
            p, t = run_metrognn_inference(metro_gnn, metro_ds_template, device, s, e, args.batch_size, args.num_workers)
            m = compute_metrics(t, p)
            results["MetroGNN"].append(m)
            if is_local_target: local_target_metrics["MetroGNN"] = m
            print(f"      [MetroGNN] RMSE: {m[1]:.4f} MAPE: {m[3]:.4f}")

        # MPGCN
        if mpgcn:
            p, t = run_mpgcn_inference(mpgcn, mpgcn_ds_template, device, s, e, args.batch_size, args.num_workers)
            m = compute_metrics(t, p)
            results["MPGCN"].append(m)
            if is_local_target: local_target_metrics["MPGCN"] = m
            print(f"      [MPGCN]    RMSE: {m[1]:.4f} MAPE: {m[3]:.4f}")


    # ===========================================================
    # 4. Final Report
    # ===========================================================
    print("\n\n")
    print("#########################################################")
    print("                 FINAL EVALUATION REPORT                 ")
    print("#########################################################")
    metrics_header = ["MSE", "RMSE", "MAE", "MAPE", "SMAPE"]

    # Table 1: Local Challenge (특정 구간 1:1 비교)
    print(f"\n[Table 1] Local Challenge (Target OD: {args.local_od_i}->{args.local_od_j})")
    print(f"{'Model':<12} | {'RMSE':<8} | {'MAE':<8} | {'SMAPE':<8}")
    print("-" * 45)
    for model_name in ["TCN", "ST-LSTM", "MPGCN", "MetroGNN"]:
        if model_name in local_target_metrics:
            m = local_target_metrics[model_name]
            print(f"{model_name:<12} | {m[1]:.4f}   | {m[2]:.4f}   | {m[4]:.4f}")
        else:
            print(f"{model_name:<12} |   N/A    |   N/A    |   N/A")
    
    # Table 2: Global Efficiency (전체 구간 평균 성능)
    print(f"\n\n[Table 2] Global Efficiency (Average over {len(TARGET_OD_LIST)} Pairs)")
    print(f"{'Model':<12} | {'RMSE':<8} | {'MAE':<8} | {'SMAPE':<8} | {'Note'}")
    print("-" * 65)
    
    for model_name in ["TCN", "ST-LSTM", "MPGCN", "MetroGNN"]:
        score_list = results[model_name]
        if len(score_list) > 0:
            avg = np.mean(np.array(score_list), axis=0)
            note = "Global Model" if len(score_list) == len(TARGET_OD_LIST) else "Local Only"
            print(f"{model_name:<12} | {avg[1]:.4f}   | {avg[2]:.4f}   | {avg[4]:.4f}   | {note}")
        else:
            print(f"{model_name:<12} |   N/A    |   N/A    |   N/A    | Not Evaluated")

    print("\nEvaluation Complete.")

if __name__ == "__main__":
    main()