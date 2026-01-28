# infer_odformer.py
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataset import get_odformer_dataset
from SCIE_Benchmark.ODFormer import ODFormer
from train.trainer import ODformerLM
from tqdm import tqdm

# ===============================
# Per-horizon RMSE
# ===============================
def per_horizon_rmse(y_true, y_pred):
    """
    y_true, y_pred: (B, T, D)
    return: (T,)
    """
    y_true = torch.expm1(y_true)
    y_pred = torch.expm1(y_pred)

    B, T, N, _, F = y_true.shape

    # flatten OD
    y_true = y_true.view(B, T, -1)
    y_pred = y_pred.view(B, T, -1)


    mask = (y_true > 0).float()

    rmses = []

    for t in range(T):
        diff2 = (y_true[:, t] - y_pred[:, t]) ** 2
        diff2 = diff2 * mask[:, t]

        mse_t = diff2.sum() / mask[:, t].sum()
        rmses.append(torch.sqrt(mse_t))

    return torch.stack(rmses)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--test_subdir", default="test")
    parser.add_argument("--adj_csv", default="./AD_matrix_trimmed_common.csv")

    parser.add_argument("--window_size", type=int, default=96)
    parser.add_argument("--pred_size", type=int, default=192)
    parser.add_argument("--hop_size", type=int, default=12)

    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================
    # Adjacency
    # =====================
    adj = pd.read_csv(args.adj_csv, index_col=0).values

    # =====================
    # Dataset
    # =====================
    _, testset = get_odformer_dataset(
        data_root=args.data_root,
        train_subdir="train",
        val_subdir=args.test_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
        use_time_feature=True,
        cache_in_mem=False
    )

    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    sample = testset[0]
    N = sample["X"].shape[1]
    F = sample["X"].shape[-1]

    # =====================
    # Load model
    # =====================
    lm = ODformerLM.load_from_checkpoint(
        args.ckpt,
        model=ODFormer(
            num_regions=N,
            feature_dim=F,
            out_feature_dim=1,
            hidden_dim=64,
            alpha=0.7,
            num_heads=4,
            pred_len=args.pred_size
        ),
        adj_matrix=adj
    )

    lm.eval().to(device)

    # =====================
    # Inference
    # =====================
    metric_buf = []
    horizon_buf = []
    zero_buf = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            X = batch["X"].to(device)
            y = batch["Y"].to(device)

            y_pred = lm(X)

            # global metrics
            metric_buf.append(torch.stack(lm._compute_metrics(y, y_pred)).cpu())

            # horizon RMSE
            horizon_buf.append(per_horizon_rmse(y, y_pred).cpu())

            # zero ratio
            y_real = torch.expm1(y).squeeze(-1).view(y.size(0), y.size(1), -1)
            zero_buf.append((y_real < 1).float().mean().cpu())

    metrics = torch.stack(metric_buf).mean(0)
    horizon_rmse = torch.stack(horizon_buf).mean(0)
    zero_ratio = torch.stack(zero_buf).mean()

    # =====================
    # Print summary
    # =====================
    print("\n====== ODFormer Test Metrics ======")
    print(f"MSE   : {metrics[0]:.4f}")
    print(f"MAE   : {metrics[1]:.4f}")
    print(f"MAPE  : {metrics[2]:.2f}%")
    print(f"SMAPE : {metrics[3]:.2f}%")
    print(f"RMSE  : {metrics[4]:.4f}")
    print(f"Zero ratio (<1): {zero_ratio:.4f}")

    # =====================
    # Save horizon RMSE
    # =====================
    output_dir = os.path.join(os.getcwd(),"results","odformer_inference")
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame({
        "horizon": list(range(1, len(horizon_rmse) + 1)),
        "rmse": horizon_rmse.numpy()
    }).to_csv(os.path.join(output_dir, "per_horizon_rmse.csv"), index=False)

    plt.figure(figsize=(8,4))
    plt.plot(horizon_rmse.numpy())
    plt.xlabel("Prediction Horizon")
    plt.ylabel("RMSE")
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_horizon_rmse.png"), dpi=200)

    print("\nSaved:")
    print(" results/per_horizon_rmse.csv")
    print(" results/per_horizon_rmse.png")


if __name__ == "__main__":
    main()
