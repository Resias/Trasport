#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

python "$ROOT/evaluate/gate_threshold_sensitivity.py" \
  --ckpt_glob "${CKPT_GLOB:-$ROOT/ablation_runs/progressive_core_v1/S8_gate/seed_*/checkpoints/best-*.ckpt}" \
  --max_seeds "${MAX_SEEDS:-3}" \
  --thresholds "${THRESHOLDS:-0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9}" \
  --data_root "${DATA_ROOT:-$ROOT/data_splits/od_minute_review_3way}" \
  --test_subdir "${TEST_SUBDIR:-val}" \
  --od_csv "${OD_CSV:-$ROOT/AD_matrix_trimmed_common.csv}" \
  --time_resolution "${TIME_RESOLUTION:-1}" \
  --window_size "${WINDOW_SIZE:-60}" \
  --pred_size "${PRED_SIZE:-30}" \
  --hop_size "${HOP_SIZE:-10}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --mape_eps "${MAPE_EPS:-1e-3}" \
  --device "${DEVICE:-}" \
  --output_dir "${OUTPUT_DIR:-$ROOT/evaluate/outputs/gate_sensitivity}"
