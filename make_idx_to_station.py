import pandas as pd
import json

CSV_PATH = "AD_matrix_trimmed_common.csv"
OUT_PATH = "station_to_idx.json"

df = pd.read_csv(CSV_PATH, index_col=0)

# index = station names
stations = list(df.index)

station_to_idx = {
    station: idx
    for idx, station in enumerate(stations)
}

with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(station_to_idx, f, ensure_ascii=False, indent=2)

print("Saved:", OUT_PATH)
print("Total stations:", len(station_to_idx))
print("First 5:")
for k in list(station_to_idx.keys())[:5]:
    print(k, "->", station_to_idx[k])