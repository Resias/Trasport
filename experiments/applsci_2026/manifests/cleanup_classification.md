# Cleanup Classification

Date: 2026-06-19

This file classifies large local directories before any destructive cleanup. No directory in this
manifest has been deleted by this refactor.

## Preserve

| Path | Approx size | Reason |
| --- | ---: | --- |
| `data_splits/od_minute_review_3way/` | 16K | Paper split symlinks |
| `metro-GATrasformer-od-week-latlon/` | 90M | Metro-GATF checkpoint candidates |
| `metro-GCN-LSTM/` | 1.2M | Paper baseline checkpoint candidates |
| `metro-MPGCN/` | 228K | Paper baseline checkpoint candidates |
| `metro-st-lstm/` | 1.4M | Pair-level ST-LSTM checkpoint candidates |
| `artifacts/mpgcn_dyn_60m.pt` | 149M | MPGCN dynamic graph cache |
| `st_lstm_artifacts_mta_hourly/` | 916K | ST-LSTM reproducibility artifacts |
| `review_runs/review_3way_50ep_20260512/evaluation/` | mixed | Main review evaluation results |
| `review_runs/review_3way_50ep_20260512/efficiency/` | mixed | Efficiency results |
| `review_runs/review_3way_50ep_20260512/summary/` | mixed | Review summary |
| `evaluate/outputs/gate_sensitivity/` | mixed | Gate sensitivity outputs |
| `final_station_analysis/` | 2.1M | Final station analysis and figures |

## Preserve Candidate, Reduce Later

| Path | Approx size | Reason |
| --- | ---: | --- |
| `metro-AutoformerOD/` | 1.3G | Paper baseline checkpoints, but exact final seeds need selection |
| `ablation_runs/progressive_core_v1/` | mixed | Paper S0-S8 progressive ablation source |
| `review_runs/review_3way_50ep_20260512/` | mixed | Contains final summaries plus logs and auxiliary artifacts |

## Archive Candidate

| Path | Approx size | Reason |
| --- | ---: | --- |
| `before_train_code_saved_model/` | 3.5G | Historical snapshot, not part of final paper path |
| `metro-st-damhgn/` | 28M | Related-work model not in the main comparison |
| `analysis_results/` | 6.8M | Older analysis outputs, likely superseded by final station analysis |
| `daily_compare/` | 20M | Intermediate comparison outputs |
| `inference_results_no_mix/` | 12G | Large intermediate inference outputs, needs final-result mapping check |

## Remove Candidate After Manifest Verification

| Path | Approx size | Reason |
| --- | ---: | --- |
| `wandb/` | 2.0G | Reproducibility should come from configs/manifests, not raw local W&B cache |
| `train/wandb/` | unknown | Local W&B cache |
| `__pycache__/` and nested `__pycache__/` | small | Generated Python bytecode |
| root `checkpoints/` | 792K | Not mapped to paper checkpoint manifest |
| `*.log`, `*.pid`, `*.logpath` | mixed | Runtime markers/logs |

## Source Archive Candidates

Do not move these until import checks and paper entrypoints are stable:

- `SCIE_Benchmark/STDAMHGN.py`
- `SCIE_Benchmark/hyper_graph_preprocessed.py`
- `train/train_st_damhgn.py`
- `models/GNN.py`
- `models/TransformerConv.py`
- `inference_SCIE/infer_odformer.py`
- `inference_metrognn.py`
- `evaluate.py`
- `evaluate/evaluate_all.py`
- `evaluate/evaluate_for_paper.py`
- `evaluate/evaluate_for_paperV2.py`
- `analyze/analysis_more.py`
- `analyze/analyze_od_prediction.py`
- `analyze/create_analysis_table.py`
