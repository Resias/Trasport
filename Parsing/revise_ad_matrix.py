import pandas as pd
import json

# ---------------------------
# 1) Load parsed stations
# ---------------------------
with open("station2id.json", "r", encoding="utf-8") as f:
    station2id = json.load(f)

parsed_stations = set(station2id.keys())
print("파싱 역 개수:", len(parsed_stations))


# ---------------------------
# 2) Load full AD matrix
# ---------------------------
df = pd.read_csv("AD_matrix.csv", index_col=0)
all_ad_stations = set(df.index)
print("AD matrix 역 개수:", len(all_ad_stations))



# ---------------------------
# 3) 교집합 및 차집합 계산
# ---------------------------
common_stations = sorted(parsed_stations & all_ad_stations)
dropped_stations = sorted(parsed_stations - all_ad_stations)

print("교집합 역 개수:", len(common_stations))
print("AD_matrix에 없는 파싱역 개수:", len(dropped_stations))


# ---------------------------
# 🔥 여기서 실제 목록 출력!
# ---------------------------
print("\n=== AD_matrix에 없는 파싱역 목록 ===")
for s in dropped_stations:
    print(s)

print("\n총", len(dropped_stations), "개\n")


# ---------------------------
# 4) 행렬 필터링 (교집합만)
# ---------------------------
AD_trimmed = df.loc[common_stations, common_stations]

print("Trimmed shape:", AD_trimmed.shape)
# ---------------------------
# 5) 저장
# ---------------------------
AD_trimmed.to_csv("AD_matrix_trimmed_common.csv", encoding="utf-8-sig")
print("저장 완료 → AD_matrix_trimmed_common.csv")
