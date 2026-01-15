import os
import glob
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
# ----------------------------------------------------------
# 🔥 1) 명시적으로 변경해야 하는 역명 매핑 테이블
# ----------------------------------------------------------
SPECIAL_MAP = {
    "4.19민주묘지": "4·19민주묘지",
    "4·19민주묘지": "4·19민주묘지",
    "관악산": "관악산(서울대)",
    "관악산(서울대)": "관악산(서울대)",
    "남동인더스파크": "인더스파크남동",
    "당고개": "불암산",
    "동대문역사문화공원": "문화공원동대문역사",
    "뚝섬유원지": "자양",
    "시청용인대": "시청·용인대",
    "신대방삼거리": "삼거리신대방",
    "운동장송담대": "용인중앙시장",
    "이수": "총신대입구",
    "인천국제공항1터미널": "인천공항1터미널",
    "인천국제공항2터미널": "인천공항2터미널",
    "전대에버랜드": "전대·에버랜드",
    "지제": "평택지제",
    "화전": "한국항공대",
    "흑석(중앙대입구)": "흑석",
    # 특수 케이스: 청량리(지상), 청량리(지하)
    "청량리(지상)": "청량리",
    "청량리(지하)": "청량리",
    # 아시아드경기장
    "아시아드경기장": "아시아드경기장",
    "아시아드경기장(공촌사거리": "아시아드경기장",
    "아시아드경기장(공촌사거리)": "아시아드경기장"
}

# ----------------------------------------------------------
# 🔥 2) 괄호 제거 함수
# ----------------------------------------------------------
import re
def remove_parentheses(name: str):
    """괄호와 그 안 내용 제거"""
    return re.sub(r"\(.*?\)", "", name).strip()
    

# ----------------------------------------------------------
# 🔥 3) 최종 역명 정규화 함수
# ----------------------------------------------------------
def normalize_station_name(name: str):
    if not isinstance(name, str):
        return "Unknown"

    name = name.strip()

    # 1️⃣ 특수 매핑 먼저 적용
    if name in SPECIAL_MAP:
        return SPECIAL_MAP[name]

    # 2️⃣ 특수 매핑 안 된 경우 → 괄호 제거
    cleaned = remove_parentheses(name)

    # 3️⃣ 청량리 같은 case는 위에서 처리됨
    return cleaned


def build_station_dict(sample_file):
    df = pd.read_parquet(sample_file)

    df["승차역명"] = df["승차역명"].fillna("Unknown")
    df["하차역명"] = df["하차역명"].fillna("Unknown")

    stations = sorted(set(df["승차역명"]) | set(df["하차역명"]))
    station2id = {s: i for i, s in enumerate(stations)}

    return station2id


def compute_minute_OD(day_files, station2id):
    N = len(station2id)
    minute_slots = 1440
    OD = np.zeros((minute_slots, N, N), dtype=np.int32)

    for f in tqdm(day_files, desc="minute-OD 계산"):
        df = pd.read_parquet(f, columns=["승차역명", "하차역명", "승차일시"])
        df["승차역명"] = df["승차역명"].apply(normalize_station_name)
        df["하차역명"] = df["하차역명"].apply(normalize_station_name)
        # df = df[df["승차역명"].isin(station2id)]
        # df = df[df["하차역명"].isin(station2id)]
        
        df["승차역명"] = df["승차역명"].fillna("Unknown")
        df["하차역명"] = df["하차역명"].fillna("Unknown")

        df = df[(df["승차역명"] != "Unknown") & (df["하차역명"] != "Unknown")]
        

        df["승차일시_dt"] = pd.to_datetime(df["승차일시"], errors="coerce")
        df = df.dropna(subset=["승차일시_dt"])

        df["minute_idx"] = df["승차일시_dt"].dt.hour * 60 + df["승차일시_dt"].dt.minute
        df = df[(df["minute_idx"] >= 0) & (df["minute_idx"] < 1440)]

        origins = df["승차역명"].map(station2id).values
        dests   = df["하차역명"].map(station2id).values
        mins    = df["minute_idx"].values

        for o, d, m in zip(origins, dests, mins):
            OD[m, o, d] += 1

    return OD

def compute_hourly_OD(day_files, station2id):
    N = len(station2id)
    # 총 24시간
    OD_by_hour = {h: np.zeros((N, N), dtype=np.int64) for h in range(24)}

    for f in tqdm(day_files, desc="시간대별 OD 계산중"):
        df = pd.read_parquet(f, columns=["승차역명", "하차역명", "승차일시"])

        df["승차역명"] = df["승차역명"].fillna("Unknown")
        df["하차역명"] = df["하차역명"].fillna("Unknown")

        df["hour"] = pd.to_datetime(df["승차일시"]).dt.hour

        for _, row in df.iterrows():
            o = station2id[row["승차역명"]]
            d = station2id[row["하차역명"]]
            h = row["hour"]

            OD_by_hour[h][o, d] += 1

    return OD_by_hour

def compute_daily_OD(day_files, station2id):
    N = len(station2id)
    OD = np.zeros((N, N), dtype=np.int64)

    for f in tqdm(day_files, desc="OD 계산중"):
        df = pd.read_parquet(f, columns=["승차역명", "하차역명"])

        df["승차역명"] = df["승차역명"].fillna("Unknown")
        df["하차역명"] = df["하차역명"].fillna("Unknown")

        origins = df["승차역명"].map(station2id).values
        dests   = df["하차역명"].map(station2id).values

        for o, d in zip(origins, dests):
            OD[o, d] += 1

    return OD

def build_global_station_dict(parquet_files):
    stations = set()
    for f in tqdm(parquet_files, desc="전체 역 목록 스캔"):
        df = pd.read_parquet(f, columns=["승차역명", "하차역명"])
        df["승차역명"] = df["승차역명"].apply(normalize_station_name)
        df["하차역명"] = df["하차역명"].apply(normalize_station_name)

        df["승차역명"] = df["승차역명"].fillna("Unknown")
        df["하차역명"] = df["하차역명"].fillna("Unknown")

        stations.update(df["승차역명"].tolist())
        stations.update(df["하차역명"].tolist())
    # Unknown 제거
    if "Unknown" in stations:
        stations.remove("Unknown")

    station2id = {s: i for i, s in enumerate(sorted(stations))}
    return station2id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processing", default="./202205/train_pars")
    parser.add_argument("--outdir", default="./od_minute")
    args = parser.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.processing, "*.parquet")))

    # 날짜별 grouping
    date2files = {}
    for f in all_files:
        fname = os.path.basename(f)
        date = fname.split("_")[1][:8]  # e.g., TCD_20220501.parquet → 20220501
        date2files.setdefault(date, []).append(f)

    # 역 사전은 첫 파일에서 생성
    station2id = build_global_station_dict(all_files)

    print("총 역 개수:", len(station2id))

    os.makedirs(args.outdir, exist_ok=True)

    # 날짜별 처리
    for date, files in date2files.items():
        OD = compute_minute_OD(files, station2id)
        
        save_path = os.path.join(args.outdir, f"OD_minute_{date}.npy")
        np.save(save_path, OD)
        
        print(f"저장 완료 → {save_path}")


if __name__ == "__main__":
    main()