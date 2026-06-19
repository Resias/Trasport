#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

python "$ROOT/evaluate/evaluate_review_efficiency.py" \
  --study_dir "${STUDY_DIR:-$ROOT/review_runs/review_3way_50ep_20260512}" \
  --data_root "${DATA_ROOT:-$ROOT/data_splits/od_minute_review_3way}" \
  --train_subdir "${TRAIN_SUBDIR:-train}" \
  --test_subdir "${TEST_SUBDIR:-test}" \
  --models "${MODELS:-S8_gate,ODFormer,MPGCN,GCN_LSTM,Autoformer}" \
  --seeds "${SEEDS:-0,1,2}" \
  --od_csv "${OD_CSV:-$ROOT/AD_matrix_trimmed_common.csv}" \
  --time_resolution "${TIME_RESOLUTION:-1}" \
  --window_size "${WINDOW_SIZE:-60}" \
  --pred_size "${PRED_SIZE:-30}" \
  --hop_size "${HOP_SIZE:-10}" \
  --gate_tau "${METRO_GATF_GATE_TAU:-0.9}" \
  --mpgcn_dynamic_graph_cache "${MPGCN_DYNAMIC_GRAPH_CACHE:-$ROOT/artifacts/mpgcn_dyn_60m.pt}" \
  --mpgcn_dynamic_bin_size "${MPGCN_DYNAMIC_BIN_SIZE:-60}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --warmup_batches "${WARMUP_BATCHES:-2}" \
  --max_batches "${MAX_BATCHES:-20}" \
  --device "${DEVICE:-}" \
  --output_dir "${OUTPUT_DIR:-}"
