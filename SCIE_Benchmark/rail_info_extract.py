# rail_info_extract.py
import re
import pandas as pd
import numpy as np

# ======================================================
# PATHS
# ======================================================
AD_PATH = "./AD_matrix_trimmed_common.csv"
EXCEL_PATH = "./ALL_subway_train_info_20250930.xlsx"
SAVE_PATH = "./ad_station_latlon.csv"
MISSING_PATH = "./missing_station_geo.csv"

# ======================================================
# 1. LOAD AD MATRIX (GROUND TRUTH)
# ======================================================
ad_df = pd.read_csv(AD_PATH, index_col=0)
ad_stations = list(ad_df.index)
ad_station_set = set(ad_stations)

print(f"[INFO] AD matrix stations: {len(ad_stations)}")

# ======================================================
# 2. LOAD STATION INFO EXCEL
# ======================================================
station_df = pd.read_excel(EXCEL_PATH, engine="openpyxl")

USE_COLS = {
    "역사명": "station_name",
    "역위도": "lat",
    "역경도": "lon",
}
station_df = station_df[list(USE_COLS.keys())].rename(columns=USE_COLS)

# ======================================================
# 3. ALIAS MAP (AD CANONICAL → EXCEL VARIANT)
# ======================================================
ALIAS_MAP = {
    "4·19민주묘지": "4.19민주묘지",
    "관악산(서울대)": "관악산",
    "문화공원동대문역사": "동대문역사문화공원",
    "삼거리신대방": "신대방삼거리",
    "총신대입구": "이수",
    "불암산": "당고개",
    "평택지제": "지제",
    "한국항공대": "화전",
    "전대·에버랜드": "전대에버랜드",
    "인천공항1터미널": "인천국제공항1터미널",
    "인천공항2터미널": "인천국제공항2터미널",
    "인더스파크남동": "남동인더스파크"
}

# ======================================================
# 4. NORMALIZATION FUNCTIONS
# ======================================================
def normalize_raw(name: str) -> str | None:
    """기본 문자열 정규화 (엑셀/AD 공통)"""
    if pd.isna(name):
        return None

    name = str(name)
    name = re.sub(r"\s+", "", name)
    name = name.replace("（", "(").replace("）", ")")
    name = re.sub(r"\(.*?\)", "", name)     # 괄호 제거
    name = name.replace(".", "·")           # 기호 통일
    if name == "서울역":
        return name
    if name.endswith("역"):
        name = name[:-1]
    return name


def normalize_to_ad(excel_name: str) -> str | None:
    """
    엑셀 역명을 AD 기준 역명으로 매핑
    """
    base = normalize_raw(excel_name)

    # 1. 직접 매칭
    if base in ad_station_set:
        return base

    # 2. alias 매칭
    for ad_name, excel_variant in ALIAS_MAP.items():
        if base == normalize_raw(excel_variant):
            return ad_name

    return None


# ======================================================
# 5. APPLY NORMALIZATION & MAPPING
# ======================================================
station_df["station_ad"] = station_df["station_name"].apply(normalize_to_ad)

matched = station_df["station_ad"].notna().sum()
print(f"[INFO] Matched stations after normalization: {matched}")

# ======================================================
# 6. FILTER VALID + DROP MISSING GEO
# ======================================================
filtered_df = station_df[
    station_df["station_ad"].notna()
].dropna(subset=["lat", "lon"]).copy()

# ======================================================
# 7. MERGE DUPLICATES (e.g. 청량리 지상/지하)
# ======================================================
geo_df = (
    filtered_df
    .groupby("station_ad")[["lat", "lon"]]
    .mean()
)

# ======================================================
# 8. REORDER TO MATCH AD MATRIX
# ======================================================
available_stations = [s for s in ad_stations if s in geo_df.index]
missing_stations = [s for s in ad_stations if s not in geo_df.index]

print(f"[INFO] Stations with geo info: {len(available_stations)} / {len(ad_stations)}")

geo_df = geo_df.loc[available_stations]

# ======================================================
# 9. SAVE RESULTS
# ======================================================
geo_df.to_csv(SAVE_PATH, encoding="utf-8-sig")

pd.Series(missing_stations).to_csv(
    MISSING_PATH,
    index=False,
    header=["station_name"],
    encoding="utf-8-sig"
)

print(f"[DONE] Saved AD-aligned station geo file → {SAVE_PATH}")
print(f"[WARN] Missing stations saved → {MISSING_PATH}")
print(f"[FINAL] Stations saved: {len(geo_df)}")
