# stats_dataset.py
import os
import re
import datetime as dt
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def _safe_load_npy(path: str, mmap: bool = True):
    if mmap:
        return np.load(path, mmap_mode="r")
    return np.load(path)


def _parse_file_meta(fname: str) -> Dict:
    """
    지원하는 파일명:
      1) YYYYMMDD.npy                 -> month, weekday는 실제 날짜에서 계산
      2) YYYYMM_dowX.npy              -> month, weekday는 파일명에서 직접
      3) YYYYMMDD_anything.npy        -> 앞 8자리 날짜 파싱 시도
    """
    stem = fname.replace(".npy", "")

    # case2: YYYYMM_dowX
    m = re.match(r"^(\d{6})_dow([0-6])$", stem)
    if m:
        yyyymm = m.group(1)
        dow = int(m.group(2))
        year = int(yyyymm[:4])
        month = int(yyyymm[4:6])
        return {"year": year, "month": month, "weekday": dow, "date_str": None}

    # case1/3: YYYYMMDD at start
    m = re.match(r"^(\d{8})", stem)
    if m:
        yyyymmdd = m.group(1)
        year = int(yyyymmdd[:4])
        month = int(yyyymmdd[4:6])
        day = int(yyyymmdd[6:8])
        weekday = dt.date(year, month, day).weekday()  # Mon=0..Sun=6
        return {"year": year, "month": month, "weekday": weekday, "date_str": yyyymmdd}

    raise ValueError(f"Cannot parse meta from filename: {fname}")


def _is_consecutive(hours: np.ndarray) -> bool:
    """
    hours: (L,) int array.
    True if strictly consecutive by +1 each step.
    """
    if len(hours) <= 1:
        return True
    return np.all(np.diff(hours) == 1)


class IntraDayODStatsDataset(Dataset):
    """
    ✅ Task A: intra-day OD forecasting dataset (hourly)

    - Each file is one "day-profile sample": (T_day, N, N)
      (주의: 실제 연속 날짜가 아니어도 상관없음. 이 데이터는 월/요일/시간 패턴 샘플임)

    - We create sliding windows inside each file:
        X: (window_size, N, N)
        Y: (pred_size,   N, N)

    Returns dict keys (통계모델 스크립트 호환 목적):
      - x_tensor: torch.FloatTensor (T_in, N, N)   (optionally log1p)
      - y_tensor: torch.FloatTensor (T_out, N, N)  (optionally log1p)
      - weekday_tensor: torch.LongTensor scalar (0=Mon..6=Sun)
      - month_tensor:   torch.LongTensor scalar (1..12)
      - hour_hist: torch.LongTensor (T_in,)   (hour-of-day indices from .time.npy)
      - hour_fut:  torch.LongTensor (T_out,)
      - file_key: str  (debug)

    Important:
      - Uses .mask.npy and .time.npy if present.
      - Only keeps windows where:
          * mask window all True
          * hours are consecutive for (window_size+pred_size)
    """

    def __init__(
        self,
        data_dir: str,
        window_size: int,
        pred_size: int,
        hop_size: int = 1,
        log1p: bool = True,
        cache_in_mem: bool = False,
        mmap: bool = True,
        verbose: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.window_size = int(window_size)
        self.pred_size = int(pred_size)
        self.hop_size = int(hop_size)
        self.log1p = bool(log1p)
        self.cache_in_mem = bool(cache_in_mem)
        self.mmap = bool(mmap)

        assert self.window_size > 0
        assert self.pred_size > 0
        assert self.hop_size > 0
        assert (self.window_size + self.pred_size) <= 24, (
            "Hourly intra-day Task A에서는 window_size + pred_size <= 24 를 권장/가정합니다."
        )

        # -------------------------
        # 1) scan files
        # -------------------------
        fnames = sorted(
            f for f in os.listdir(data_dir)
            if f.endswith(".npy")
            and not f.endswith(".time.npy")
            and not f.endswith(".mask.npy")
        )
        if len(fnames) == 0:
            raise FileNotFoundError(f"No .npy day files found in: {data_dir}")

        self.files: List[Dict] = []
        for f in fnames:
            meta = _parse_file_meta(f)
            base = os.path.join(data_dir, f)

            time_path = base.replace(".npy", ".time.npy")
            mask_path = base.replace(".npy", ".mask.npy")

            self.files.append({
                "path": base,
                "time_path": time_path if os.path.exists(time_path) else None,
                "mask_path": mask_path if os.path.exists(mask_path) else None,
                "year": meta["year"],
                "month": meta["month"],
                "weekday": meta["weekday"],
                "file_key": os.path.basename(base).replace(".npy", ""),
            })

        # -------------------------
        # 2) optional caching
        # -------------------------
        self._cache_od: List[Optional[np.ndarray]] = [None] * len(self.files)
        self._cache_time: List[Optional[np.ndarray]] = [None] * len(self.files)
        self._cache_mask: List[Optional[np.ndarray]] = [None] * len(self.files)

        if self.cache_in_mem and verbose:
            print(f"[IntraDayODStatsDataset] Caching into RAM: {len(self.files)} files")

        # -------------------------
        # 3) build window index
        # -------------------------
        self.index: List[Tuple[int, int]] = []  # (file_idx, start_step)
        iterator = range(len(self.files))
        if verbose:
            iterator = tqdm(iterator, desc=f"Indexing windows: {os.path.basename(data_dir)}")

        for file_idx in iterator:
            od, hours, mask = self._load_file(file_idx)

            T_day = od.shape[0]
            if T_day < (self.window_size + self.pred_size):
                continue

            last_start = T_day - (self.window_size + self.pred_size)
            # ✅ include last_start itself -> +1
            for s in range(0, last_start + 1, self.hop_size):
                e = s + self.window_size + self.pred_size

                mask_slice = mask[s:e]
                if not np.all(mask_slice):
                    continue

                hour_slice = hours[s:e]
                if not _is_consecutive(hour_slice):
                    continue

                self.index.append((file_idx, s))

        if verbose:
            print(f"[IntraDayODStatsDataset] total windows: {len(self.index)}")

        # infer N
        od0, _, _ = self._load_file(0)
        self.num_nodes = int(od0.shape[1])

    def __len__(self):
        return len(self.index)

    def _load_file(self, file_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          od:    (T_day, N, N) float
          hours: (T_day,) int  (e.g., 0..23)
          mask:  (T_day,) bool
        """
        if self.cache_in_mem and self._cache_od[file_idx] is not None:
            return self._cache_od[file_idx], self._cache_time[file_idx], self._cache_mask[file_idx]

        info = self.files[file_idx]
        od = _safe_load_npy(info["path"], mmap=self.mmap)

        # hours
        if info["time_path"] is not None:
            hours = np.load(info["time_path"])
        else:
            hours = np.arange(od.shape[0], dtype=np.int64)

        # mask
        if info["mask_path"] is not None:
            mask = np.load(info["mask_path"]).astype(bool)
        else:
            mask = np.ones(od.shape[0], dtype=bool)

        # sanity
        if od.ndim != 3:
            raise ValueError(f"OD tensor must be (T,N,N). got {od.shape} @ {info['path']}")
        if hours.shape[0] != od.shape[0]:
            raise ValueError(f"time.npy length mismatch: {hours.shape} vs {od.shape} @ {info['path']}")
        if mask.shape[0] != od.shape[0]:
            raise ValueError(f"mask.npy length mismatch: {mask.shape} vs {od.shape} @ {info['path']}")

        # optional cache
        if self.cache_in_mem:
            od = np.array(od, dtype=np.float32, copy=True)   # materialize from mmap
            hours = np.array(hours, dtype=np.int64, copy=True)
            mask = np.array(mask, dtype=bool, copy=True)
            self._cache_od[file_idx] = od
            self._cache_time[file_idx] = hours
            self._cache_mask[file_idx] = mask

        return od, hours, mask

    def __getitem__(self, idx: int) -> Dict:
        file_idx, s = self.index[idx]
        info = self.files[file_idx]
        od, hours, _mask = self._load_file(file_idx)

        e_in = s + self.window_size
        e_out = e_in + self.pred_size

        x = od[s:e_in]     # (T_in, N, N)
        y = od[e_in:e_out] # (T_out,N,N)

        # convert to float32 tensor
        x_t = torch.tensor(x, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        # optional log1p (baseline 코드에서 expm1로 복구 가능)
        if self.log1p:
            x_t = torch.log1p(torch.clamp(x_t, min=0.0))
            y_t = torch.log1p(torch.clamp(y_t, min=0.0))

        hour_hist = torch.tensor(hours[s:e_in], dtype=torch.long)
        hour_fut  = torch.tensor(hours[e_in:e_out], dtype=torch.long)

        return {
            "x_tensor": x_t,
            "y_tensor": y_t,
            "weekday_tensor": torch.tensor(info["weekday"], dtype=torch.long),
            "month_tensor": torch.tensor(info["month"], dtype=torch.long),
            "hour_hist": hour_hist,
            "hour_fut": hour_fut,
            "file_key": info["file_key"],
        }


def get_stats_datasets(
    data_root: str,
    train_subdir: str,
    test_subdir: str,
    window_size: int,
    pred_size: int,
    hop_size: int = 1,
    log1p: bool = True,
    cache_in_mem: bool = False,
    mmap: bool = True,
    verbose: bool = True,
):
    train_dir = os.path.join(data_root, train_subdir)
    test_dir  = os.path.join(data_root, test_subdir)

    trainset = IntraDayODStatsDataset(
        data_dir=train_dir,
        window_size=window_size,
        pred_size=pred_size,
        hop_size=hop_size,
        log1p=log1p,
        cache_in_mem=cache_in_mem,
        mmap=mmap,
        verbose=verbose,
    )
    testset = IntraDayODStatsDataset(
        data_dir=test_dir,
        window_size=window_size,
        pred_size=pred_size,
        hop_size=hop_size,
        log1p=log1p,
        cache_in_mem=cache_in_mem,
        mmap=mmap,
        verbose=verbose,
    )
    return trainset, testset
