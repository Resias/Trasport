# dataset.py

import os
import datetime
import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader


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

        # 파일들을 날짜 순으로 정렬
        file_names = sorted(os.listdir(data_root))
        self.data_paths = [os.path.join(data_root, f) for f in file_names]

        # ---- 2) 파일 전체 로드 (메모리 캐싱) ----
        print("Caching OD matrices into memory...")
        self.day_data_cache = []
        for path in tqdm(self.data_paths):
            arr = np.load(path)  # shape: [1440, N, N]
            self.day_data_cache.append(torch.tensor(arr, dtype=torch.float32))
        print("Caching completed.")
        
        # ---- 3) sliding window 정보 생성 ----
        self.info_list = []
        for file_idx, file_name in enumerate(file_names):
            # 파일명에서 날짜 추출 (예: something_YYYYMMDD.npy)
            ymd = file_name.split('_')[-1].split('.')[0]
            date = datetime.date(int(ymd[0:4]), int(ymd[4:6]), int(ymd[6:8]))
            weekday = date.weekday()  # 0=월, 6=일

            for start_idx in range(self.day_start_minute, self.day_end_minute-(self.pred_size+self.window_size), hop_size):
                self.info_list.append({
                    "file_idx": file_idx,
                    "start_idx": start_idx,
                    "weekday": weekday,
                })

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, index):
        info = self.info_list[index]
        file_idx = info["file_idx"]
        start_idx = info["start_idx"]
        weekday = info["weekday"]

        # -------- Memory Cache에서 바로 가져오기 ----------
        day_data = self.day_data_cache[file_idx]  # (1440, N, N)
        
        # slicing
        x_tensor = day_data[start_idx:start_idx+self.window_size]  
        y_tensor = day_data[start_idx+self.window_size:
                            start_idx+self.window_size+self.pred_size]

        # time encoding
        hist_minutes = torch.arange(start_idx, start_idx+self.window_size) % 1440
        fut_minutes = torch.arange(start_idx+self.window_size,
                                   start_idx+self.window_size+self.pred_size) % 1440

        time_enc_hist = build_time_sin_cos(hist_minutes.numpy())
        time_enc_fut = build_time_sin_cos(fut_minutes.numpy())

        return {
            "x_tensor": x_tensor,      # (T_in, N, N)
            "y_tensor": y_tensor,
            "weekday_tensor": torch.tensor(weekday),
            "time_enc_hist": time_enc_hist,
            "time_enc_fut": time_enc_fut
        }


class ODPairDatasetV2(Dataset):
    """
    MetroDataset 구조를 그대로 참고한 OD Pair Dataset
    - 특정 OD pair (i, j)에 대한 시계열만 추출
    - sliding window 적용
    - 최종 입력: (T_in, F)
    - 최종 출력: (T_out, 1)
    """

    def __init__(self, data_root, window_size, hop_size, pred_size,
                 od_i, od_j,
                 use_weekday=True,
                 use_time_encoding=True):
        super().__init__()

        self.data_root = data_root
        self.window_size = window_size
        self.hop_size = hop_size
        self.pred_size = pred_size
        self.i = od_i
        self.j = od_j
        self.use_weekday = use_weekday
        self.use_time_encoding = use_time_encoding

        self.day_start_minute = 5 * 60 + 30  # 330
        self.day_end_minute = 24 * 60        # 1440

        self.info_list = []
        self.data_list = []

        file_names = sorted(os.listdir(data_root))
        self.data_paths = [os.path.join(data_root, f) for f in file_names]

        # ---- 1) 하루치 파일 전체 메모리 캐싱 ----
        print("Caching OD matrices for ODPairDataset...")
        self.day_data_cache = []
        self.weekday_list = []
        for file_name in file_names:
            ymd = file_name.split('_')[-1].split('.')[0]
            date = datetime.date(int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8]))
            self.weekday_list.append(date.weekday())

        for path in tqdm(self.data_paths):
            arr = np.load(path)
            self.day_data_cache.append(torch.tensor(arr, dtype=torch.float32))  
        print("Caching completed.")

        # ---- 2) sliding window 정의 ----
        self.info_list = []
        for file_idx, wd in enumerate(self.weekday_list):
            for start_idx in range(
                self.day_start_minute,
                self.day_end_minute - (self.window_size + self.pred_size),
                self.hop_size
            ):
                self.info_list.append({
                    "file_idx": file_idx,
                    "weekday": wd,
                    "start_idx": start_idx,
                })

    def __len__(self):
        return len(self.info_list)

    def __getitem__(self, index):
        info = self.info_list[index]
        file_idx = info["file_idx"]
        weekday = info["weekday"]
        start_idx = info["start_idx"]

        # -------- 캐싱된 메모리에서 가져오기 --------
        day_data = self.day_data_cache[file_idx]  # (1440, N, N)

        # 특정 OD pair 시계열 추출
        od_seq = day_data[:, self.i, self.j]  # (1440,)

        # window slicing
        x_vals = od_seq[start_idx:start_idx+self.window_size]
        y_vals = od_seq[start_idx+self.window_size:
                        start_idx+self.window_size+self.pred_size]

        # Build features
        # (T_in, 1)
        flow_feature = x_vals.unsqueeze(-1)

        feat_list = [flow_feature]

        # 시간 인코딩
        if self.use_time_encoding:
            hist_minutes = np.arange(start_idx, start_idx+self.window_size) % 1440
            time_enc_hist = build_time_sin_cos(hist_minutes)
            feat_list.append(time_enc_hist)

        # 요일 원핫
        if self.use_weekday:
            weekday_oh = weekday_onehot(weekday).unsqueeze(0).repeat(self.window_size, 1)
            feat_list.append(weekday_oh)

        x_feat = torch.cat(feat_list, dim=1)
        y_feat = y_vals.unsqueeze(-1)

        return {
            "x": x_feat.float(),     # (T, F)
            "y": y_feat.float(),     # (1, 1)
        }


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


def get_odpair_dataset(data_root, train_subdir, val_subdir,
                       window_size, hop_size, pred_size, od_i, od_j):
    trainset = ODPairDatasetV2(
        os.path.join(data_root, train_subdir),
        window_size, hop_size, pred_size,
        od_i, od_j
    )
    valset = ODPairDatasetV2(
        os.path.join(data_root, val_subdir),
        window_size, hop_size, pred_size,
        od_i, od_j
    )
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