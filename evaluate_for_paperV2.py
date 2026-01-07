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
from trainer import TCNMetroLM, MetroLM, STLSTMLM, MPGCNLM
from GNN import MetroGNNForecaster

# ===========================================================
# 🎯 평가 설정
# ===========================================================
TARGET_OD_LIST = [
    (523, 10),      # st-lstm 학습 OD
    # (523, 419),     # st-lstm 학습 OD
    # (20, 10),       # st-lstm 학습 OD
    # (25, 10),       # test용
]

# ===========================================================
# 1. Metric 계산 함수 (Local용)
# ===========================================================
def compute_metrics(y_true, y_pred):
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-3))) * 100
        smape = np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-3)) * 100
    
    return mse, rmse, mae, mape, smape

def compute_global_mape_correct(y_true, y_pred, eps=1e-8):
    """
    Global MAPE (a.k.a WAPE)
    = sum(|y - y_hat|) / sum(|y|)
    """
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)

    numerator = torch.sum(torch.abs(y_true - y_pred))
    denominator = torch.sum(torch.abs(y_true)) + eps

    return (numerator / denominator).item() * 100

# ===========================================================
# 2. ★ 신규 추가: 전체 네트워크(Full OD) 평가 함수
# ===========================================================
def evaluate_full_network(model, dataset, device, batch_size, num_workers, model_type="MetroGNN"):
    """
    모든 OD 쌍(NxN)에 대한 평균 성능을 평가합니다.
    메모리 부족 방지를 위해 오차의 합계만 누적합니다.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    
    total_mse_sum = 0
    total_mae_sum = 0
    total_count = 0
    total_abs_error = 0.0
    total_true_sum = 0.0

    
    
    print(f"   Running Full Network Evaluation for {model_type}...")
    
    with torch.no_grad():
        for batch in loader:
            # 1. Input Data 준비
            if model_type == "MetroGNN":
                x = batch["x_tensor"].to(device)
                weekday = batch["weekday_tensor"].to(device)
                time_hist = batch["time_enc_hist"].to(device)
                time_fut = batch["time_enc_fut"].to(device)
                y_true = batch["y_tensor"].to(device) # (B, T, N, N)
                
                # Inference
                y_pred = model(x, weekday, time_hist, time_fut) # (B, T, N, N)
                
                # 마지막 시점(-1)만 평가 (Local 평가 기준과 통일)
                y_pred = y_pred[:, -1, :, :]
                y_true = y_true[:, -1, :, :]
                
            elif model_type == "MPGCN":
                x = batch["x"].to(device)
                y_true = batch["y"].to(device) # (B, 1, N, N)
                
                # Inference
                y_pred = model(x) # (B, N, N) (MPGCN 구조상 바로 나옴)
                y_true = y_true[:, -1, :, :]
            
            # 2. Error Accumulation (전체 행렬 연산)
            # Flatten to (Batch * N * N)
            diff = (y_true - y_pred).reshape(-1)
            
            # Sum of Squared Errors
            total_mse_sum += torch.sum(diff ** 2).item()
            # Sum of Absolute Errors
            total_mae_sum += torch.sum(torch.abs(diff)).item()
            # Element Count
            total_count += diff.numel()
            total_abs_error += torch.sum(torch.abs(y_true.reshape(-1) - y_pred.reshape(-1))).item()
            total_true_sum += torch.sum(torch.abs(y_true.reshape(-1))).item()

    # 3. Final Calculation
    final_mse = total_mse_sum / total_count
    final_rmse = np.sqrt(final_mse)
    final_mae = total_mae_sum / total_count
    final_mape = (total_abs_error / total_true_sum + 1e-8) * 100
    
    # 주의: 전체 행렬은 0이 매우 많으므로 MAPE/SMAPE는 왜곡될 수 있어 제외하거나 주의 필요.
    # 여기서는 신뢰성 있는 RMSE, MAE만 반환
    return final_mse, final_rmse, final_mae, final_mape

# ===========================================================
# 3. 모델 로더 및 추론 함수들
# ===========================================================
def load_metrognn_model(ckpt_path, dataset, device, od_csv):
    sample = dataset[0]
    in_feat = sample["x_tensor"].shape[-1]
    time_emb_dim = sample["time_enc_hist"].shape[-1]
    od_df = pd.read_csv(od_csv, index_col=0)
    od_df = od_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"[MetroGNN] Loading Global Model from {ckpt_path} ...")
    lm = MetroLM.load_from_checkpoint(
        ckpt_path, map_location=device,
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

def run_metrognn_inference_local(model, dataset, device, od_i, od_j, batch_size, num_workers):
    # 기존 Local 평가용 추론 함수 (Slicing)
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
            
            pred_full = model(x, weekday, time_hist, time_fut)
            p = pred_full[:, -1, od_i, od_j].unsqueeze(-1)
            t = y[:, -1, od_i, od_j].unsqueeze(-1)
            preds.append(p.cpu())
            trues.append(t.cpu())
    
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()

def load_mpgcn_model(ckpt_path, dataset, device):
    sample = dataset[0]
    N = sample["x"].shape[2]

    model_struct = MPGCN(
        N=N, 
        lstm_h=32, 
        gcn_h=32, 
        gcn_out=16, 
        K=3
    )
    
    print("Setting up Static Graphs...")
    adj_O, adj_D, poi_O, poi_D = load_static_graphs("/workspace/od_minute", N)
    
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

def run_mpgcn_inference_local(model, dataset, device, od_i, od_j, batch_size, num_workers):
    # 기존 Local 평가용 추론 함수 (Slicing)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred_full = model(x)
            p = pred_full[:, od_i, od_j].unsqueeze(-1)
            t = y[:, -1, od_i, od_j].unsqueeze(-1)
            preds.append(p.cpu())
            trues.append(t.cpu())
    return torch.cat(preds).numpy(), torch.cat(trues).numpy()

# ... (Local Model Wrappers - evaluate_tcn_wrapper, evaluate_stlstm_wrapper는 위와 동일) ...
def evaluate_tcn_wrapper(ckpt, data_root, test_subdir, win, hop, pred, od_i, od_j, batch, workers, device):
    _, ds = get_odpair_dataset(data_root, test_subdir, test_subdir, win, hop, pred, od_i, od_j)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    lm = TCNMetroLM.load_from_checkpoint(ckpt, map_location=device,
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
    _, ds = get_st_lstm_dataset(data_root, test_subdir, test_subdir, win, hop, pred, target_s=od_i, target_e=od_j, top_k=3, dist_matrix=dist, W_matrix=W)
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    sample = ds[0]
    loss = torch.nn.MSELoss()
    lm = STLSTMLM.load_from_checkpoint(ckpt, map_location=device, model=STLSTM(input_dim=sample["x"].shape[-1], hidden_dim=64, num_layers=2, output_dim=30), loss=loss)
    model = lm.model.to(device)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            p = model(x)[:, -1].unsqueeze(-1)
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
    # Settings
    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--test_subdir", default="test")
    parser.add_argument("--od_csv", default="./AD_matrix_trimmed_common.csv")
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--pred_size", type=int, default=1)
    parser.add_argument("--hop_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    # Local Model Info
    parser.add_argument("--local_od_i", type=int, default=523)
    parser.add_argument("--local_od_j", type=int, default=10)
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"🚀 Evaluation Started | Device: {device}")
    
    # 1. Load Aux Data
    dist_matrix = np.load("./dist_matrix.npy") if os.path.exists("./dist_matrix.npy") else None
    W_matrix = np.load("./W_matrix.npy") if os.path.exists("./W_matrix.npy") else None

    # 2. Load Global Models
    metro_gnn = None; metro_ds_template = None
    if args.gnn_ckpt:
        _, metro_ds_template = get_dataset(args.data_root, args.test_subdir, args.test_subdir, 
                                           args.window_size, args.hop_size, args.pred_size)
        metro_gnn = load_metrognn_model(args.gnn_ckpt, metro_ds_template, device, args.od_csv)

    mpgcn = None; mpgcn_ds_template = None
    if args.mpgcn_ckpt:
        _, mpgcn_ds_template = get_mpgcn_dataset(args.data_root, args.test_subdir, args.test_subdir,
                                                 args.window_size, args.hop_size, args.pred_size)
        mpgcn = load_mpgcn_model(args.mpgcn_ckpt, mpgcn_ds_template, device)

    # 3. Local Challenge Loop
    local_metrics_store = {} # {OD_tuple: {Model: Metrics}}
    
    print("\n[Phase 1] Running Local Challenge (Specific ODs)...")
    for (s, e) in TARGET_OD_LIST:
        print(f"   Evaluating Pair: {s} -> {e}")
        local_metrics_store[(s,e)] = {}
        
        # Local Models (Only run if matches target)
        if s == args.local_od_i and e == args.local_od_j:
            if args.tcn_ckpt:
                p, t = evaluate_tcn_wrapper(args.tcn_ckpt, args.data_root, args.test_subdir,
                                            args.window_size, args.hop_size, args.pred_size,
                                            s, e, args.batch_size, args.num_workers, device)
                local_metrics_store[(s,e)]["TCN"] = compute_metrics(t, p)
            if args.stlstm_ckpt:
                p, t = evaluate_stlstm_wrapper(args.stlstm_ckpt, args.data_root, args.test_subdir,
                                               args.window_size, args.hop_size, args.pred_size,
                                               s, e, args.batch_size, args.num_workers, device, dist_matrix, W_matrix)
                local_metrics_store[(s,e)]["ST-LSTM"] = compute_metrics(t, p)
        
        # Global Models (Run Local Slicing)
        if metro_gnn:
            p, t = run_metrognn_inference_local(metro_gnn, metro_ds_template, device, s, e, args.batch_size, args.num_workers)
            local_metrics_store[(s,e)]["MetroGNN"] = compute_metrics(t, p)
        if mpgcn:
            p, t = run_mpgcn_inference_local(mpgcn, mpgcn_ds_template, device, s, e, args.batch_size, args.num_workers)
            local_metrics_store[(s,e)]["MPGCN"] = compute_metrics(t, p)

    # 4. ★ Global Full Network Evaluation (New!)
    print("\n[Phase 2] Running Full Network Evaluation (Global vs Global)...")
    global_results = {}
    
    if metro_gnn:
        mse, rmse, mae, mape = evaluate_full_network(metro_gnn, metro_ds_template, device, args.batch_size, args.num_workers, "MetroGNN")
        global_results["MetroGNN"] = (mse, rmse, mae, mape)
        print(f"   [MetroGNN] Full RMSE: {rmse:.4f} Global MAPE: {mape:.4f}%")
        
    if mpgcn:
        mse, rmse, mae, mape = evaluate_full_network(mpgcn, mpgcn_ds_template, device, args.batch_size, args.num_workers, "MPGCN")
        global_results["MPGCN"] = (mse, rmse, mae, mape)
        print(f"   [MPGCN]    Full RMSE: {rmse:.4f} Global MAPE: {mape:.4f}%")

    # ===========================================================
    # 5. Final Report Printing
    # ===========================================================
    print("\n\n################ FINAL EVALUATION REPORT ################")
    
    # [Table 1] & [Table 2]는 Local Challenge 결과 출력 (기존과 동일, 공간상 생략하거나 간단히 출력)
    print(f"\n[Table 1] Local Challenge (Target OD: {args.local_od_i}->{args.local_od_j})")
    target_key = (args.local_od_i, args.local_od_j)
    if target_key in local_metrics_store:
        print(f"{'Model':<12} | {'RMSE':<8} | {'MAE':<8}")
        print("-" * 35)
        for m_name, vals in local_metrics_store[target_key].items():
            print(f"{m_name:<12} | {vals[1]:.4f}   | {vals[2]:.4f}")
            
    # [Table 3] Global Network Evaluation
    print(f"\n\n[Table 3] Full Network Evaluation (All {metro_ds_template[0]['x_tensor'].shape[-1]**2} OD Pairs)")
    print(f"{'Model':<12} | {'RMSE':<8} | {'MAE':<8} | {'Note'}")
    print("-" * 50)
    
    for m_name in ["MPGCN", "MetroGNN"]:
        if m_name in global_results:
            vals = global_results[m_name] # (mse, rmse, mae)
            print(f"{m_name:<12} | {vals[1]:.4f}   | {vals[2]:.4f}   | Global Model")
        else:
            print(f"{m_name:<12} |   N/A    |   N/A    | Not Evaluated")
            
    print("\nEvaluation Complete.")

if __name__ == "__main__":
    main()