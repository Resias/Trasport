import os
import argparse
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader

from dataset import get_dataset
from GNN import MetroGNNForecaster
from trainer import MetroLM

# 추론결과 저장되는것은 각 일자별 06:30 ~ 11:29분까지의 OD 행렬(390~1409까지)

def parse_args():
    parser = argparse.ArgumentParser("MetroGNN Inference (NO DAY MIXING)")
    parser.add_argument("--data_root", default="/home/data/od_minute")
    parser.add_argument("--test_subdir", default="test")
    parser.add_argument("--od_csv", default="./AD_matrix_trimmed_common.csv")

    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--pred_size", type=int, default=30)
    parser.add_argument("--hop_size", type=int, default=30)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--save_dir", default="./inference_results_no_mix")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # =====================================================
    # Dataset (DO NOT MODIFY)
    # =====================================================
    _, testset = get_dataset(
        data_root=args.data_root,
        train_subdir="train",
        val_subdir=args.test_subdir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
    )

    test_loader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=False,              # ★ 매우 중요
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # =====================================================
    # Load adjacency
    # =====================================================
    od_df = pd.read_csv(args.od_csv, index_col=0)
    od_df = od_df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # =====================================================
    # Build model
    # =====================================================
    sample = testset[0]
    in_feat = sample["x_tensor"].shape[-1]
    time_emb_dim = sample["time_enc_hist"].shape[-1]

    model = MetroGNNForecaster(
        od_df=od_df,
        in_feat=in_feat,
        gnn_hidden=32,
        rnn_hidden=32,
        weekday_emb_dim=16,
        time_emb_dim=time_emb_dim,
        window_size=args.window_size,
        pred_size=args.pred_size,
        device=device,
    )

    lightning_model = MetroLM.load_from_checkpoint(
        checkpoint_path=args.ckpt_path,
        model=model,
        loss=torch.nn.MSELoss(),
        lr=1e-3,
    )
    lightning_model.to(device)
    lightning_model.eval()

    # =====================================================
    # Inference (NO DAY MIXING 핵심부)
    # =====================================================
    # timeline[file_idx] = [(minute_in_day, OD_matrix)]
    timeline = defaultdict(list)

    global_idx = 0  # ★ dataset index 직접 추적

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            batch = {k: v.to(device) for k, v in batch.items()}

            B = batch["x_tensor"].shape[0]

            y_pred, _, _ = lightning_model.forward_batch(batch)
            y_pred = y_pred.cpu().numpy()  # (B, pred, N, N)

            for b in range(B):
                info = testset.info_list[global_idx]
                file_idx = info["file_idx"]
                start_idx = info["start_idx"]

                for h in range(args.pred_size):
                    minute_in_day = start_idx + args.window_size + h
                    timeline[file_idx].append(
                        (minute_in_day, y_pred[b, h])
                    )

                global_idx += 1

    # =====================================================
    # Save results (PER DAY, MINUTE + HOURLY)
    # =====================================================
    for file_idx, entries in timeline.items():
        entries.sort(key=lambda x: x[0])

        day_dir = os.path.join(args.save_dir, f"day_{file_idx:02d}")
        minute_dir = os.path.join(day_dir, "minute")
        hour_dir = os.path.join(day_dir, "hourly")

        os.makedirs(minute_dir, exist_ok=True)
        os.makedirs(hour_dir, exist_ok=True)

        hourly_sum = defaultdict(lambda: None)

        for minute, od in entries:
            od = np.clip(od, 0, None)
            od = np.rint(od).astype(np.int64)

            # ---------- minute save ----------
            pd.DataFrame(
                od,
                index=od_df.index,
                columns=od_df.columns
            ).to_csv(
                os.path.join(minute_dir, f"od_minute_{minute:04d}.csv")
            )

            # ---------- hourly accumulate ----------
            hour = minute // 60
            if hourly_sum[hour] is None:
                hourly_sum[hour] = od.copy()
            else:
                hourly_sum[hour] += od

        # ---------- hourly save ----------
        for hour, mat in sorted(hourly_sum.items()):
            pd.DataFrame(
                mat,
                index=od_df.index,
                columns=od_df.columns
            ).to_csv(
                os.path.join(hour_dir, f"od_hour_{hour:02d}.csv")
            )
    # =====================================================
    # Save REAL results in SAME FORMAT as prediction
    # =====================================================
    real_root = os.path.join(args.data_root, args.test_subdir)

    for file_idx, entries in timeline.items():
        # --- load real 하루 ---
        real_path = sorted(os.listdir(real_root))[file_idx]
        real_day = np.load(os.path.join(real_root, real_path))  # (1440, N, N)

        day_dir = os.path.join(args.save_dir, f"day_{file_idx:02d}")
        real_minute_dir = os.path.join(day_dir, "real_minute")
        real_hour_dir = os.path.join(day_dir, "real_hourly")

        os.makedirs(real_minute_dir, exist_ok=True)
        os.makedirs(real_hour_dir, exist_ok=True)

        real_hourly_sum = defaultdict(lambda: None)

        for minute, _ in entries:
            if minute >= real_day.shape[0]:
                continue

            od = real_day[minute]
            od = np.clip(od, 0, None)
            od = np.rint(od).astype(np.int64)

            # ---------- real minute save ----------
            pd.DataFrame(
                od,
                index=od_df.index,
                columns=od_df.columns
            ).to_csv(
                os.path.join(real_minute_dir, f"od_minute_{minute:04d}.csv")
            )

            # ---------- hourly accumulate ----------
            hour = minute // 60
            if real_hourly_sum[hour] is None:
                real_hourly_sum[hour] = od.copy()
            else:
                real_hourly_sum[hour] += od

        # ---------- real hourly save ----------
        for hour, mat in sorted(real_hourly_sum.items()):
            pd.DataFrame(
                mat,
                index=od_df.index,
                columns=od_df.columns
            ).to_csv(
                os.path.join(real_hour_dir, f"od_hour_{hour:02d}.csv")
            )



if __name__ == "__main__":
    main()
