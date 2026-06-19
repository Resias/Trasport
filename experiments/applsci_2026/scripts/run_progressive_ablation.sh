#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
STAGES="${STAGES:-${EXPERIMENT_NAMES:-S0_minimal,S1_factorization,S2_multiscale_static,S3_dynamic,S4_transformer,S5_time,S6_weekday,S7_geo,S8_gate}}"

python "$ROOT/train/train_ablation.py" \
  --experiment_plan progressive_core \
  --stages "$STAGES" \
  --seeds "${SEEDS:-0,1,2}" \
  --data_root "${DATA_ROOT:-$ROOT/data_splits/od_minute_review_3way}" \
  --train_subdir "${TRAIN_SUBDIR:-train}" \
  --val_subdir "${VAL_SUBDIR:-val}" \
  --od_csv "${OD_CSV:-$ROOT/AD_matrix_trimmed_common.csv}" \
  --station_latlon_csv "${STATION_LATLON_CSV:-$ROOT/ad_station_latlon.csv}" \
  --time_resolution "${TIME_RESOLUTION:-1}" \
  --window_size "${WINDOW_SIZE:-60}" \
  --pred_size "${PRED_SIZE:-30}" \
  --hop_size "${HOP_SIZE:-10}" \
  --gate_tau "${METRO_GATF_GATE_TAU:-0.9}" \
  --max_epochs "${MAX_EPOCHS:-200}" \
  --output_root "${OUTPUT_ROOT:-$ROOT/ablation_runs}" \
  --study_name "${STUDY_NAME:-progressive_core_v1}"
