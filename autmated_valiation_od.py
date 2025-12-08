import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import json

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

import re
def remove_parentheses(name: str):
    return re.sub(r"\(.*?\)", "", name).strip()

def normalize_station_name(name: str):
    if not isinstance(name, str):
        return "Unknown"

    name = name.strip()
    if name in SPECIAL_MAP:
        return SPECIAL_MAP[name]

    return remove_parentheses(name)


# =====================================================
# 1) Parquet 기본 구조 검사
# =====================================================
def validate_parquet_structure(parquet_files):
    print("\n==============================")
    print("📌 [1] Parquet 구조 검증")
    print("==============================")

    required_cols = ["승차역명", "하차역명", "승차일시"]

    sample = pd.read_parquet(parquet_files[0])
    print("샘플 파일:", parquet_files[0])

    missing = [c for c in required_cols if c not in sample.columns]
    if missing:
        print("❌ 필수 컬럼 누락:", missing)
    else:
        print("✔ 필수 컬럼 OK")

    print("\nNaN 비율 점검:")
    for col in required_cols:
        print(f"  - {col}: {sample[col].isna().mean():.4f}")

    t = pd.to_datetime(sample["승차일시"], errors="coerce")
    print(f"\n✔ 시간 파싱 성공률: {(~t.isna()).mean():.4f}")

    print("\n샘플 5개:")
    print(sample.head())
    print()


# =====================================================
# 2) station2id 사전 검증
# =====================================================
def validate_station_dict(station2id):
    print("\n==============================")
    print("📌 [2] 역 사전(station2id) 검증")
    print("==============================")

    print("총 역 개수:", len(station2id))
    print("Unknown 개수:", sum(1 for s in station2id if s == "Unknown"))

    if len(station2id) == len(set(station2id.keys())):
        print("✔ 중복 없음")
    else:
        print("❌ 중복 있음")


# =====================================================
# 3) OD 매트릭스 검증
# =====================================================
def validate_minute_od(od_path, parquet_files, station2id):

    print("\n==============================")
    print("📌 [3] OD 매트릭스 검증")
    print("==============================")

    OD = np.load(od_path)
    print("OD shape:", OD.shape)

    minute_slots, N1, N2 = OD.shape
    N = len(station2id)

    print("✔ 1440분 OK" if minute_slots == 1440 else "❌ minute 오류")
    print("✔ 역 개수 OK" if (N1 == N and N2 == N) else f"❌ 역 개수 불일치: OD={N1}, dict={N}")

    # Unknown 제외한 전체 승차 수 계산
    total_rides = 0
    for f in parquet_files:
        df = pd.read_parquet(f, columns=["승차역명", "하차역명"])
        df["승차역명"] = df["승차역명"].apply(normalize_station_name)
        df["하차역명"] = df["하차역명"].apply(normalize_station_name)
        df = df[(df["승차역명"] != "Unknown") & (df["하차역명"] != "Unknown")]
        total_rides += len(df)

    od_sum = OD.sum()

    print(f"\n총 승차 수(Unknown 제외): {total_rides:,}")
    print(f"OD 총합:                 {od_sum:,}")

    if abs(total_rides - od_sum) <= max(1, total_rides * 0.001):
        print("✔ 총합 일치")
    else:
        print("❌ 총합 불일치 — 파싱 규칙 차이 가능")

    # 분당 분포 예시
    print("\n시간대별 분포(0~10분):")
    print(OD.sum(axis=(1,2))[:10])
    print()


# =====================================================
# 4) station2id에 포함되지 않는 역명 확인
# =====================================================
def check_missing_station_names(parquet_files, station2id):

    print("\n==============================")
    print("📌 [4] station2id에 없는 역명 검사")
    print("==============================")

    valid_stations = set(station2id.keys())
    missing = set()

    for f in tqdm(parquet_files, desc="역명 검사"):
        df = pd.read_parquet(f, columns=["승차역명","하차역명"])
        df["승차역명_norm"] = df["승차역명"].apply(normalize_station_name)
        df["하차역명_norm"] = df["하차역명"].apply(normalize_station_name)

        missing.update(set(df["승차역명_norm"]) - valid_stations)
        missing.update(set(df["하차역명_norm"]) - valid_stations)

    missing.discard("Unknown")

    print("\n🔥 station2id에 없는 역명 목록:")
    for n in sorted(missing):
        print(" -", n)

    print(f"\n총 {len(missing)}개의 역명이 station2id에 없음\n")


# =====================================================
# Main
# =====================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="./202205/train_pars")
    parser.add_argument("--od_dir", default="./od_minute")
    parser.add_argument("--save_json", default="./station2id.json")
    args = parser.parse_args()

    parquet_files = sorted(glob.glob(os.path.join(args.data, "*.parquet")))
    if not parquet_files:
        print("❌ parquet 없음")
        return

    # 1) parquet 구조 검증
    validate_parquet_structure(parquet_files)

    # 2) station2id 생성 및 검증
    from od_making import build_global_station_dict
    station2id = build_global_station_dict(parquet_files)

    validate_station_dict(station2id)

    # 저장
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(station2id, f, ensure_ascii=False, indent=2)
    print(f"📁 station2id 저장 완료 → {args.save_json}")

    # 3) OD 검증
    od_files = sorted(glob.glob(os.path.join(args.od_dir, "OD_minute_*.npy")))
    if not od_files:
        print("❌ OD 파일 없음")
        return

    import random
    idx = random.randrange(0,len(od_files))
    sample_od = od_files[idx]
    date = os.path.basename(sample_od).split("_")[2].split(".")[0]
    day_files = [f for f in parquet_files if date in f]

    validate_minute_od(sample_od, day_files, station2id)

    # 4) station2id에 없는 역명 검사
    check_missing_station_names(parquet_files, station2id)

    print("\n==============================")
    print("🎉 전체 검증 완료")
    print("==============================\n")


if __name__ == "__main__":
    main()
