#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_JSON="${OUT_JSON:-$ROOT/review_runs/review_3way_50ep_20260512/statistical_baselines/ha_arima_test_metrics.json}"
EXTRA_ARGS=()
if [[ -n "${EXTRA_STAT_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=($EXTRA_STAT_ARGS)
fi

python "$ROOT/evaluate/statistical_baselines_review.py" \
  --data_root "${DATA_ROOT:-$ROOT/data_splits/od_minute_review_3way}" \
  --train_subdir "${TRAIN_SUBDIR:-train}" \
  --test_subdir "${TEST_SUBDIR:-test}" \
  --time_resolution "${TIME_RESOLUTION:-1}" \
  --window_size "${WINDOW_SIZE:-60}" \
  --hop_size "${HOP_SIZE:-10}" \
  --pred_size "${PRED_SIZE:-30}" \
  --mape_eps "${MAPE_EPS:-1e-3}" \
  --arima_order "${ARIMA_ORDER:-1,1,1}" \
  --arima_pair_chunk_size "${ARIMA_PAIR_CHUNK_SIZE:-0}" \
  --output_json "$OUT_JSON" \
  "${EXTRA_ARGS[@]}"
