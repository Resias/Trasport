# Data Preservation Manifest

Date: 2026-06-19

Policy:

- Preserve these assets locally for Appl. Sci. 2026 reproducibility.
- Keep large arrays, split directories, and caches ignored by git.
- Before deleting or moving an asset, update this manifest and verify the replacement path.

## Split

| Asset | Size | Checksum | Role | Status |
| --- | ---: | --- | --- | --- |
| `data_splits/od_minute_review_3way/` | 16K | symlink directory | Paper split root, 19 train days, 5 validation days, 7 test days | preserve-local |

The split directory contains symlinks to `/home/data/od_minute/train` and `/home/data/od_minute/test`.
It does not store the raw OD arrays directly.

Expected day coverage:

- Train: `OD_minute_20220501.npy` through `OD_minute_20220519.npy`
- Validation: `OD_minute_20220520.npy` through `OD_minute_20220524.npy`
- Test: `OD_minute_20220525.npy` through `OD_minute_20220531.npy`

## Network And Metadata

| Asset | Size | SHA-256 | Role | Status |
| --- | ---: | --- | --- | --- |
| `AD_matrix.csv` | 848K | `434a59dc1fbcccb1ca276ea55c6e18698ac44b3821640da1eea535b86935e1cf` | Full station adjacency matrix | preserve-local |
| `AD_matrix_trimmed_common.csv` | 808K | `4b2fff47bb40784022c9de1fceffba610b8b4ceb93003830275a184f6000c830` | 637-station adjacency used by main models | preserve-local |
| `station_to_idx.json` | 16K | `ce5ac622dc512f6a598a437bcf06b925aecf9a85e4d3fa0471846174d702e359` | Station ID to model index mapping | preserve-local |
| `ad_station_latlon.csv` | 20K | `a3ad0e74794eb9c11297bc2bc97691eee89627d5b39ef64e6c9b8d198b068d15` | Station coordinates for Metro-GATF geo feature | preserve-local |
| `missing_station_geo.csv` | 4K | `e725eb1ed488f161cae6b47aab98fa86dee3abe36fddd2e2ec8989955523b00a` | Coordinate repair/reference metadata | preserve-local |
| `ALL_subway_train_info_20250930.xlsx` | 420K | `64f6d323050c4e6a99a1644fe1a9f9432570df8b1eccb0874cf6af607406d67d` | Line/station source metadata | preserve-local |
| `Parsing/line_info.xlsx` | 52K | `324a659f0bfe260b40b96bc3c41564b9baf868e55fc8c3de2a9a28efe427bd75` | Line metadata used by preprocessing scripts | preserve-local |

## Baseline Artifacts

| Asset | Size | SHA-256 | Role | Status |
| --- | ---: | --- | --- | --- |
| `dist_matrix.npy` | 3.1M | `9fde69493f7971e54685dfc9b857bb2f03fb1d5f3004a153694b3c2763a44572` | ST-LSTM distance matrix | preserve-local |
| `W_matrix.npy` | 3.1M | `b52cfe4a7650869120716e5e97b86cde3e6a2fe6b601e418e88535cf4a91c06d` | ST-LSTM weight matrix | preserve-local |
| `W.npy` | 28K | `4612465a09fb6f9e87ce74159d5a668fe09c96e3d0cd65a5fb5340f1c5316bae` | ST-LSTM auxiliary weight artifact | preserve-local |
| `day_cluster.npy` | 4K | `5762d27001f3d5014404e4752ed6e30a7252c9c228463faae318a578697c8b5a` | ST-LSTM day cluster artifact | preserve-local |
| `top_x_od.npy` | 93M | `a0e6bd538f050670fd433f68c293d9ca3330a5bc9a9d104f19e0b86201683068` | ST-LSTM top OD artifact | preserve-local |
| `artifacts/mpgcn_dyn_60m.pt` | 149M | `0ae29cd6d82e79581a913ed87a90e82ae19a5006e30d731dcdc3b8ab885f60a6` | MPGCN dynamic graph cache | preserve-local |

## ST-LSTM MTA Hourly Artifact Folder

| Asset | Size bytes | SHA-256 | Role | Status |
| --- | ---: | --- | --- | --- |
| `st_lstm_artifacts_mta_hourly/W.npy` | 284 | `8af60dfcb1169d48efef8f0c458eff73a66c3187562ca0b82849677315caa693` | ST-LSTM auxiliary artifact | preserve-local |
| `st_lstm_artifacts_mta_hourly/adj_knn.npy` | 181604 | `fcee56d789d0363d889b930128c06e2aadfb3882ebc7a29d81eef6c8465ecc81` | ST-LSTM KNN adjacency | preserve-local |
| `st_lstm_artifacts_mta_hourly/day_cluster.test.npy` | 337 | `75e23ac5c7a26815657fc3c3c78a2e9bb77d61849e91e224a52bb3c3e0248d62` | ST-LSTM test day cluster | preserve-local |
| `st_lstm_artifacts_mta_hourly/day_cluster.train.npy` | 525 | `6605d510f71d3b4be55742adce0d4e8a4f88a8925f9d0741c510727b2c6b1450` | ST-LSTM train day cluster | preserve-local |
| `st_lstm_artifacts_mta_hourly/dist_hop.npy` | 726032 | `76ac5c0cab73c0aaf0b842e405ae01ee9915e4cade0cb7514d18ab2dbfcfb0be` | ST-LSTM hop distance | preserve-local |
| `st_lstm_artifacts_mta_hourly/meta.json` | 499 | `b4330034af108256e2d434f694ad9295254441f5253374fdbf473dc39b9dba03` | ST-LSTM artifact metadata | preserve-local |
| `st_lstm_artifacts_mta_hourly/top_x_od.npy` | 356 | `14a8099d8911e6bf5353a56a5fced09cb5927d08fe701868c60cf3aba7e010cc` | ST-LSTM top OD metadata | preserve-local |

## Regeneration Notes

- Main OD arrays are expected under `/home/data/od_minute`.
- `evaluate/create_review_split.sh` can recreate the symlink split if source OD arrays exist.
- `make_st_lstm_dist_w.py` and `SCIE_Benchmark/st_lstm_mta_preprocessed.py` are related to ST-LSTM artifacts.
- `train/train_mpgcn.py` can regenerate `artifacts/mpgcn_dyn_60m.pt` when given the same split and MPGCN settings.
