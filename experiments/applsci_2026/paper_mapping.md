# Paper To Repository Mapping

## Metro-GATF

Paper name: `Metro-GATF`

Current repository implementation:

- `models/GATTransformerdecoder.py`
- `train/train_graph.py`
- `train/trainer.py::MetroGraphWeekLM`
- `evaluate/graph_eval_utils.py`
- `evaluate/gate_threshold_sensitivity.py`
- `evaluate/evaluate_full_network.py`

Architecture components from the paper:

- GATv2 static spatial encoder
- Dynamic sparse OD graph encoder
- Non-autoregressive Transformer decoder
- Origin-destination factorized reconstruction
- Sparsity-aware gate head
- Gate threshold `t = 0.9` for final evaluation

Current local run path:

- `metro-GATrasformer-od-week-latlon/`

The path spelling is historical. Keep it stable until checkpoint references are migrated.

## Baseline Mapping

| Paper model | Source | Training or evaluation entrypoint |
| --- | --- | --- |
| HA | `evaluate/statistical_baselines_review.py`, `stats_dataset.py` | `evaluate/statistical_baselines_review.py` |
| ARIMA | `evaluate/statistical_baselines_review.py`, `stats_dataset.py` | `evaluate/statistical_baselines_review.py` |
| GCN-LSTM | `SCIE_Benchmark/GCN_LSTM.py` | `train/train_gcn_lstm.py` |
| Autoformer | `SCIE_Benchmark/Autoformer.py` | `train/train_autoformer.py` |
| ODFormer | `SCIE_Benchmark/ODFormer.py` | `train/train_odformer.py` |
| MPGCN | `SCIE_Benchmark/MPGCN.py` | `train/train_mpgcn.py` |
| ST-LSTM | `SCIE_Benchmark/ST_LSTM.py` | `train/train_st_lstm.py` |

## Non-Paper Source Candidates

These files are retained for now but classified as non-main-experiment cleanup candidates:

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
