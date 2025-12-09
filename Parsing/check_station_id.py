import pandas as pd
import glob
from tqdm import tqdm
import json
from od_making import normalize_station_name

# station2id 로드
with open("station2id.json", "r", encoding="utf-8") as f:
    station2id = json.load(f)

valid_stations = set(station2id.keys())

parquet_files = sorted(glob.glob("./202205/train_pars/*.parquet"))

missing_names = set()

for f in tqdm(parquet_files, desc="역명 검사"):
    df = pd.read_parquet(f, columns=["승차역명", "하차역명"])
    df["승차역명_norm"] = df["승차역명"].apply(normalize_station_name)
    df["하차역명_norm"] = df["하차역명"].apply(normalize_station_name)

    # station2id에 없는 역명 찾기
    missing_o = set(df["승차역명_norm"]) - valid_stations
    missing_d = set(df["하차역명_norm"]) - valid_stations
    
    missing_names.update(missing_o)
    missing_names.update(missing_d)

print("\n🔥 station2id에 없는 역명 목록")
for name in sorted(missing_names):
    print(name)

print(f"\n총 {len(missing_names)}개의 역명이 station2id에 없음")
