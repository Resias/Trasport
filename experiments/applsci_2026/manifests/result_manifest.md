# Result Preservation Manifest

Date: 2026-06-19

Policy:

- Preserve final paper summaries locally.
- Small summaries can later be copied into `experiments/applsci_2026/results/` after removing machine-specific paths.
- Generated plots and large logs remain ignored by git unless explicitly curated.

## Main Review Results

| Asset | Size bytes | SHA-256 | Paper role | Status |
| --- | ---: | --- | --- | --- |
| `review_runs/review_3way_50ep_20260512/evaluation/summary_test_metrics.csv` | 1613 | `5405fe4b4a2a5037636fc92707c6c46b49748f1eb470d6f2634a4325d5e379e2` | Main test metric summary | preserve-local |
| `review_runs/review_3way_50ep_20260512/evaluation/test_metrics.json` | 9408 | `aaa144c0060dc438e3ea78d51c770e60a096c924bd504742ac25f404cfe1ea9f` | Per-model test metrics | preserve-local |
| `review_runs/review_3way_50ep_20260512/summary/report.md` | 834 | `5a1eb4033abf11eec10216e4a42f37f28d11c0f248d6aeec4e9d159cb0076c7f` | Human-readable review summary | preserve-local |

## Efficiency Results

| Asset | Size bytes | SHA-256 | Paper role | Status |
| --- | ---: | --- | --- | --- |
| `review_runs/review_3way_50ep_20260512/efficiency/summary_efficiency.csv` | 1500 | `d3f744af090ee17addb09f93a46619993a1796aabf412ac472a8f7c3851752a7` | Efficiency table summary | preserve-local |
| `review_runs/review_3way_50ep_20260512/efficiency/efficiency.json` | 12909 | `7e183386fc53d9c975a91e5d08e4071baa0935a7f9af04c7742786f8324e347b` | Full efficiency records | preserve-local |

## Gate Sensitivity Results

| Asset | Size bytes | SHA-256 | Paper role | Status |
| --- | ---: | --- | --- | --- |
| `evaluate/outputs/gate_sensitivity/summary.csv` | 2725 | `65f3fa75ef281b05d77874690fd76083def4d40cbed568c9b633fc4ecd369cc9` | Gate-threshold sensitivity summary | preserve-local |
| `evaluate/outputs/gate_sensitivity/summary.json` | 18339 | `883aa74ebee04cf720d60d12b52677641b8ee8480b2c76dd8eb7834646d17153` | Gate-threshold sensitivity details | preserve-local |
| `evaluate/outputs/gate_sensitivity/per_seed.csv` | 6512 | not recorded | Per-seed gate sensitivity rows | preserve-local |

## Station Analysis Results

| Asset | Size bytes | SHA-256 | Paper role | Status |
| --- | ---: | --- | --- | --- |
| `final_station_analysis/analysis_report.md` | 14317 | `4a3cbac54ac9477951c1abb993ee83b0bb961b6317223c6e18b83e6b88bb98e9` | Final station analysis report | preserve-local |
| `final_station_analysis/overall_metrics.json` | 770 | `49afe70cd6985ca33e4563d4318b51c824f455aa95dfb6f787cac4efd752162d` | Overall station-analysis metrics | preserve-local |
| `final_station_analysis/*.csv` | mixed | not recorded | Station/time/OD result tables | preserve-local |
| `final_station_analysis/*.png` | mixed | not recorded | Final report figures | preserve-local |

## Pseudo-AR And Auxiliary Results

The following folders are preserved as auxiliary result sources until the final paper table mapping is confirmed:

- `review_runs/review_3way_50ep_20260512/evaluation/pseudo_ar_full/`
- `review_runs/review_3way_50ep_20260512/evaluation/pseudo_ar_subset30/`
- `review_runs/review_3way_50ep_20260512/evaluation/pseudo_ar_bench_b4/`
- `review_runs/review_3way_50ep_20260512/evaluation/pseudo_ar_bench_b8/`
- `review_runs/review_3way_50ep_20260512/evaluation/pseudo_ar_sanity/`
