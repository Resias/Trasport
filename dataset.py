# dataset.py

import os
import datetime
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


def build_time_sin_cos(minute_indices, period=1440):
    """
    minute_indices: 1D numpy array or list of int (분단위 인덱스, 0~1439)
    period: 하나의 주기 길이 (1440분 = 24시간)
    
    return: [T, 2] tensor (sin, cos)
    """
    minute_indices = np.asarray(minute_indices, dtype=np.float32)
    # 0~2π 스케일로 변환
    angles = 2.0 * np.pi * minute_indices / float(period)
    sin_vals = np.sin(angles)
    cos_vals = np.cos(angles)
    enc = np.stack([sin_vals, cos_vals], axis=-1)  # [T, 2]
    return torch.tensor(enc, dtype=torch.float32)



def weekday_onehot(weekday):
    # weekday: 0~6  (월~일)
    onehot = np.zeros(7, dtype=np.float32)
    onehot[weekday] = 1.0
    return torch.tensor(onehot, dtype=torch.float32)  # (7,)


class ODPairDataset(Dataset):
    """
    OD Pair 기반 시계열 예측 Dataset
    - 원본 데이터는 (1440, N, N)
    - 특정 (i,j) OD 흐름만 추출하여 (1440,) 시계열 생성
    - 윈도우 슬라이싱 후 (T_in, F) 형태로 반환
    """

    def __init__(self,
                 data_root,
                 window_size,
                 hop_size,
                 pred_size,
                 od_i,
                 od_j,
                 use_weekday=True,
                 use_time_enc=True):
        super().__init__()
        self.data_root = data_root
        self.window_size = window_size
        self.hop_size = hop_size
        self.pred_size = pred_size
        self.i = od_i
        self.j = od_j
        self.use_weekday = use_weekday
        self.use_time_enc = use_time_enc

        self.day_start_minute = 5 * 60 + 30  # 05:30
        self.day_end_minute = 24 * 60        # 1440

        info_list = []
        data_list = []

        file_names = sorted(os.listdir(data_root))

        for file_idx, file_name in enumerate(file_names):

            ymd = file_name.split('_')[-1].split('.')[0]
            date = datetime.date(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]))
            weekday = date.weekday()

            file_path = os.path.join(data_root, file_name)

            for start_idx in range(
                self.day_start_minute,
                self.day_end_minute - (window_size + pred_size),
                hop_size
            ):
                info_list.append({
                    "file_idx": file_idx,
                    "start_idx": start_idx,
                    "weekday": weekday,
                })

            data_list.append(file_path)

        self.info_list = info_list
        self.data_list = data_list

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, index):
        info = self.info_list[index]

        file_idx = info["file_idx"]
        start_idx = info["start_idx"]
        weekday = info["weekday"]
        file_path = self.data_list[file_idx]

        # Load daily OD matrix (1440, N, N)
        day_data = np.load(file_path)  # (1440, 637, 637)

        # Extract OD pair time series → (1440,)
        full_seq = day_data[:, self.i, self.j]

        # Windowing
        x_vals = full_seq[start_idx:start_idx + self.window_size]           # (T_in,)
        y_vals = full_seq[start_idx + self.window_size:
                          start_idx + self.window_size + self.pred_size]    # (T_out,)

        # Time features
        hist_minutes = np.arange(start_idx, start_idx + self.window_size) % 1440  # (T_in,)
        fut_minutes = np.arange(start_idx + self.window_size,
                                start_idx + self.window_size + self.pred_size) % 1440

        # build time encoding
        time_enc_hist = build_time_sin_cos(hist_minutes)  # (T_in, 2)

        # weekday encoding (global to window)
        weekday_oh = weekday_onehot(weekday).unsqueeze(0).repeat(self.window_size, 1)  # (T_in, 7)

        # OD flow values → (T_in, 1)
        flow_vals = torch.tensor(x_vals, dtype=torch.float32).unsqueeze(-1)

        # ------------------------------
        # Final feature concat: (T_in, F)
        # F = 2 (sin/cos) + 7 (weekday) + 1 (flow) = 10
        # ------------------------------
        feat_list = [flow_vals]

        if self.use_time_enc:
            feat_list.append(time_enc_hist)    # (T_in, 2)

        if self.use_weekday:
            feat_list.append(weekday_oh)       # (T_in, 7)

        x_feat = torch.cat(feat_list, dim=1)  # (T_in, F)

        y_feat = torch.tensor(y_vals, dtype=torch.float32).unsqueeze(-1)  # (T_out, 1)

        return {
            "x": x_feat,      # (T_in, F)
            "y": y_feat,      # (T_out, 1)
            "weekday": weekday,
            "time_hist": time_enc_hist,
        }

class MetroDataset(Dataset):
    """
    지하철 OD 데이터셋

    - data_root: 하루 단위 .npy 파일들이 있는 디렉토리
        각 파일: [1440, N, N] (분단위, 1분 1스텝, 00:00 ~ 24:00)
    - window_size: 입력 시퀀스 길이 (분 단위)
    - hop_size: 슬라이딩 윈도우 이동 간격 (분 단위)
    - pred_size: 예측 시퀀스 길이 (분 단위)
    
    운영시간만 사용: 05:30(=330분) ~ 24:00(=1440분)
    """

    def __init__(self, data_root, window_size, hop_size, pred_size):
        super().__init__()

        self.data_root = data_root
        self.window_size = window_size
        self.hop_size = hop_size
        self.pred_size = pred_size

        # 하루 중 사용할 구간 (분)
        self.day_start_minute = 5 * 60 + 30  # 05:30 -> 330
        self.day_end_minute = 24 * 60        # 24:00 -> 1440

        info_list = []
        data_list = []

        # 파일들을 날짜 순으로 정렬
        file_names = sorted(os.listdir(data_root))

        for file_idx, file_name in enumerate(file_names):

            # 파일명에서 날짜 추출 (예: something_YYYYMMDD.npy)
            ymd = file_name.split('_')[-1].split('.')[0]
            date = datetime.date(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]))
            weekday = date.weekday()  # 0=월, 6=일

            file_path = os.path.join(data_root, file_name)
            

            for start_idx in range(self.day_start_minute, self.day_end_minute-(self.pred_size+self.window_size), hop_size):
                info_list.append({
                    "file_idx": file_idx,
                    "start_idx": start_idx,
                    "weekday": weekday,
                })

            data_list.append(file_path)

        self.info_list = info_list
        self.data_list = data_list

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, index):
        info = self.info_list[index]
        file_idx = info["file_idx"]
        start_idx = info["start_idx"]
        weekday = info["weekday"]

        file_path = self.data_list[file_idx]  # [T_day, N, N]
        day_data = np.load(file_path)  # [1440, N, N] 가정
        # 입력/출력 시퀀스 추출
        x_np = day_data[start_idx:start_idx + self.window_size]  # [T_in, N, N]
        y_np = day_data[start_idx + self.window_size:
                        start_idx + self.window_size + self.pred_size]  # [T_out, N, N]

        # 입력 구간 시간
        hist_minutes = np.arange(start_idx, start_idx+self.window_size, 1)
        # 출력 구간 시간
        fut_minutes = np.arange(start_idx+self.window_size, start_idx + self.window_size + self.pred_size, 1)

        # 혹시 24:00를 넘어갈 일은 현재 설계상 없지만,
        # 일반성을 위해 modulo 1440을 취해도 됨.
        hist_minutes = hist_minutes % 1440
        fut_minutes = fut_minutes % 1440

        # --- 분단위 sin/cos 인코딩 생성 ---
        time_enc_hist = build_time_sin_cos(hist_minutes)  # [T_in, 2]
        time_enc_fut = build_time_sin_cos(fut_minutes)    # [T_out, 2]

        # --- tensor 변환 ---
        x_tensor = torch.tensor(x_np, dtype=torch.float32)  # [T_in, N, N]
        y_tensor = torch.tensor(y_np, dtype=torch.float32)  # [T_out, N, N]
        weekday_tensor = torch.tensor(weekday, dtype=torch.long)

        sample = {
            "x_tensor": x_tensor,
            "y_tensor": y_tensor,
            "weekday_tensor": weekday_tensor,
            "time_enc_hist": time_enc_hist,  # [T_in, 2]
            "time_enc_fut": time_enc_fut,    # [T_out, 2]
        }
        return sample
    

class CacheDataset(Dataset):
    def __init__(self, pt_path):
        self.data = torch.load(pt_path, weights_only=True)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]

def get_dataset(data_root, train_subdir, val_subdir, window_size, hop_size, pred_size):
    
    # train_pt = os.path.join(data_root, 'train.pt')
    # val_pt = os.path.join(data_root, 'val.pt')
    
    # if os.path.exists(train_pt) and os.path.exists(val_pt):
    #     print(f'data load from {train_pt} and {val_pt}')
    #     trainset = CacheDataset(train_pt)
    #     valset = CacheDataset(val_pt)
        
    #     return trainset, valset
    
    train_path = os.path.join(data_root, train_subdir)
    val_path = os.path.join(data_root, val_subdir)
    
    trainset = MetroDataset(
        data_root=train_path,
        window_size=window_size,
        hop_size=hop_size,
        pred_size=pred_size
    )
    valset = MetroDataset(
        data_root=val_path,
        window_size=window_size,
        hop_size=hop_size,
        pred_size=pred_size
    )
    
    # print('data caching...')
    # train_list = []
    # for item in tqdm(trainset):
    #     train_list.append(item)
    # train_tensor = torch.tensor(train_list)
    # val_list = []
    # for item in tqdm(valset):
    #     val_list.append(item)
    # val_tensor = torch.tensor(val_list)
    
    # torch.save(train_tensor, train_pt)
    # torch.save(val_tensor, val_pt)
    
    return trainset, valset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./od_minute")
    parser.add_argument("--train_dir", type=str, default="train")
    parser.add_argument("--val_dir", type=str, default="test")
    parser.add_argument("--window_size", type=int, default=60)
    parser.add_argument("--hop_size", type=int, default=10)
    parser.add_argument("--pred_size", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    print("=== Dataset Test Start ===")
    print(f"data_root      : {args.data_root}")
    print(f"train_subdir   : {args.train_dir}")
    print(f"val_subdir     : {args.val_dir}")
    print(f"window_size    : {args.window_size}")
    print(f"hop_size       : {args.hop_size}")
    print(f"pred_size      : {args.pred_size}")

    # ---- Load datasets ----
    trainset, valset = get_dataset(
        data_root=args.data_root,
        train_subdir=args.train_dir,
        val_subdir=args.val_dir,
        window_size=args.window_size,
        hop_size=args.hop_size,
        pred_size=args.pred_size,
    )

    print(f"\nTrain samples : {len(trainset)}")
    print(f"Val samples   : {len(valset)}")

    # ---- Loaders ----
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=args.batch_size, shuffle=False)

    # ---- Fetch one batch ----
    batch = next(iter(train_loader))

    print("\n=== Sample Batch ===")
    for k, v in batch.items():
        if torch.is_tensor(v):
            print(f"{k:15s}: {tuple(v.shape)}")
        else:
            print(f"{k:15s}: {v}")

    print("\n=== Time Encoding Check ===")
    print("time_enc_hist[0]:", batch["time_enc_hist"][0][:5])
    print("\nDataset test complete.")