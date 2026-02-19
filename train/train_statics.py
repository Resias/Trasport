import os
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import get_dataset

# =========================
# Metric
# =========================
def rmse(pred, gt):
    return np.sqrt(((pred - gt) ** 2).mean())

def mae(pred, gt):
    return np.abs(pred - gt).mean()

def mape(pred, gt, eps=1e-5):
    mask = gt > eps
    return (np.abs((pred - gt)[mask] / gt[mask])).mean() * 100

def smape(pred, gt, eps=1e-5):
    return (2 * np.abs(pred - gt) / (np.abs(pred) + np.abs(gt) + eps)).mean() * 100



# ============================================================
# 1️⃣ HA (Historical Average) - STREAMING
# ============================================================
def train_HA_streaming(dataset):
    """
    Returns:
        ha_mean: dict[weekday] -> (T_out, N, N)
    """
    sums = {}
    counts = {}

    for i in tqdm(range(len(dataset)), desc="HA training"):
        item = dataset[i]
        y = np.expm1(item["y_tensor"].numpy())   # (T_out,N,N)
        # print(y.shape)
        # exit()
        wd = item["weekday_tensor"].item()

        if wd not in sums:
            sums[wd] = y.copy()
            counts[wd] = 1
        else:
            sums[wd] += y
            counts[wd] += 1

    ha_mean = {wd: sums[wd] / counts[wd] for wd in sums}
    return ha_mean


def eval_HA_streaming(ha_mean, dataset):
    se_sum = 0.0
    ae_sum = 0.0
    mape_sum = 0.0
    mape_cnt = 0
    smape_sum = 0.0
    smape_cnt = 0
    cnt = 0
    eps = 1e-5

    for i in tqdm(range(len(dataset)), desc="HA evaluation (streaming)"):
        item = dataset[i]
        wd = item["weekday_tensor"].item()

        pred = ha_mean[wd]  # (T_out,N,N)
        gt = np.expm1(item["y_tensor"].numpy())

        diff = pred - gt
        se_sum += (diff ** 2).sum()
        ae_sum += np.abs(diff).sum()
        cnt += diff.size

        mask = gt > eps
        if mask.any():
            mape_sum += (np.abs(diff)[mask] / gt[mask]).sum()
            mape_cnt += mask.sum()

        denom = np.abs(pred) + np.abs(gt) + eps
        smape_sum += (2.0 * np.abs(diff) / denom).sum()
        smape_cnt += diff.size

    rmse_val = np.sqrt(se_sum / cnt)
    mae_val = ae_sum / cnt
    mape_val = (mape_sum / max(mape_cnt, 1)) * 100.0
    smape_val = (smape_sum / max(smape_cnt, 1)) * 100.0
    return rmse_val, mae_val, mape_val, smape_val


# ============================================================
# 2️⃣ ARIMA - STREAMING (Top-K OD only)
# ============================================================
def estimate_topk_od(dataset, top_k):
    """
    Estimate important OD pairs by accumulated flow.
    """
    acc = None

    for i in tqdm(range(len(dataset)), desc="Estimating Top-K OD"):
        y = np.expm1(dataset[i]["y_tensor"].numpy())   # (T_out,N,N)
        flat = y.reshape(y.shape[0], -1).mean(axis=0)

        if acc is None:
            acc = flat
        else:
            acc += flat

    topk_idx = np.argsort(acc)[-top_k:]
    return topk_idx


def collect_arima_series(dataset, topk_idx, hop_size):
    """
    Collect 1D time-series only for Top-K OD pairs.
    Avoid heavy duplication due to overlapping windows.
    """
    series = {idx: [] for idx in topk_idx}

    first = True
    for i in tqdm(range(len(dataset)), desc="Collecting ARIMA series"):
        y = np.expm1(dataset[i]["y_tensor"].numpy())   # (T_out,N,N)
        flat = y.reshape(y.shape[0], -1)               # (T_out, N*N)

        if first:
            take = slice(None)          # first sample: take all T_out
            first = False
        else:
            take = slice(-hop_size, None)  # next samples: take only new tail

        for idx in topk_idx:
            series[idx].extend(flat[take, idx])

    return series


def train_ARIMA_streaming(series_dict, order=(1,1,1)):
    models = {}

    for idx, seq in tqdm(series_dict.items(), desc="Training ARIMA"):
        try:
            models[idx] = ARIMA(seq, order=order).fit()
        except Exception:
            pass

    return models


def eval_ARIMA_streaming(models, topk_idx, dataset, hop_size):
    """
    Rolling evaluation:
    - forecast T_out for each sample
    - then update model with first hop_size ground-truth points (time advances by hop_size)
    """
    se_sum = 0.0
    ae_sum = 0.0
    mape_sum = 0.0
    mape_cnt = 0
    smape_sum = 0.0
    smape_cnt = 0
    cnt = 0

    # keep rolling states
    rolling = dict(models)  # idx -> fitted result

    for i in tqdm(range(len(dataset)), desc="ARIMA evaluation (rolling)"):
        gt = np.expm1(dataset[i]["y_tensor"].numpy())   # (T_out,N,N)
        T_out, N, _ = gt.shape

        pred_flat = np.zeros((T_out, N * N), dtype=np.float32)

        # 1) forecast
        for idx, res in rolling.items():
            try:
                fc = res.forecast(T_out)           # (T_out,)
                pred_flat[:, idx] = np.asarray(fc)
            except Exception:
                pass

        pred = pred_flat.reshape(T_out, N, N)

        # 2) accumulate metrics (streaming)
        diff = pred - gt
        se_sum += (diff ** 2).sum()
        ae_sum += np.abs(diff).sum()
        cnt += diff.size

        # MAPE (gt>eps)
        eps = 1e-5
        mask = gt > eps
        if mask.any():
            mape_sum += (np.abs(diff)[mask] / gt[mask]).sum()
            mape_cnt += mask.sum()

        # sMAPE
        denom = np.abs(pred) + np.abs(gt) + eps
        smape_sum += (2.0 * np.abs(diff) / denom).sum()
        smape_cnt += diff.size

        # 3) update rolling states with newly observed values (first hop_size steps)
        # next sample target starts hop_size later
        new_obs = gt[:hop_size].reshape(hop_size, -1)  # (H, N*N)
        for idx, res in list(rolling.items()):
            try:
                res2 = res.append(new_obs[:, idx], refit=False)
                rolling[idx] = res2
            except Exception:
                pass

    rmse_val = np.sqrt(se_sum / cnt)
    mae_val = ae_sum / cnt
    mape_val = (mape_sum / max(mape_cnt, 1)) * 100.0
    smape_val = (smape_sum / max(smape_cnt, 1)) * 100.0
    return rmse_val, mae_val, mape_val, smape_val

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # -------------------------
    # Config
    # -------------------------
    # data_root = "/home/data/MTA_dataset_NY/od_matrices"
    data_root = "/home/data/od_minute"
    train_subdir = "train"
    test_subdir  = "test"

    window_size = 60
    hop_size = 10
    pred_size = 30
    time_resolution = 1

    top_k = 300

    # -------------------------
    # Load dataset
    # -------------------------
    trainset, testset = get_dataset(
        data_root,
        train_subdir,
        test_subdir,
        window_size,
        hop_size,
        pred_size,
        time_resolution,
        cache_in_mem=False
    )

    # =========================
    # HA
    # =========================
    print("\n[HA] Training (streaming)...")
    ha_mean = train_HA_streaming(trainset)

    print("[HA] Evaluating...")
    ha_rmse, ha_mae, ha_mape, ha_smape = eval_HA_streaming(ha_mean, testset)

    # =========================
    # ARIMA
    # =========================
    print("\n[ARIMA] Estimating Top-K OD...")
    topk_idx = estimate_topk_od(trainset, top_k)

    print("[ARIMA] Collecting series...")
    series = collect_arima_series(trainset, topk_idx, hop_size)

    print("[ARIMA] Training...")
    arima_models = train_ARIMA_streaming(series)

    print("[ARIMA] Evaluating...")
    arima_rmse, arima_mae, arima_mape, arima_smape = eval_ARIMA_streaming(arima_models, topk_idx, testset, hop_size)


    print(f"HA RMSE : {ha_rmse:.4f}")
    print(f"HA MAE  : {ha_mae:.4f}")
    print(f"HA MAPE : {ha_mape:.2f}%")
    print(f"HA sMAPE: {ha_smape:.2f}%")

    print(f"ARIMA RMSE: {arima_rmse:.4f}")
    print(f"ARIMA MAE : {arima_mae:.4f}")
    print(f"ARIMA MAPE: {arima_mape:.2f}%")
    print(f"ARIMA sMAPE: {arima_smape:.2f}%")