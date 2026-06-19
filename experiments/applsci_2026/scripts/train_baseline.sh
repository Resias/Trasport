#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

MODEL="${MODEL:-}"
if [[ -z "$MODEL" ]]; then
  echo "Set MODEL to one of: gcn_lstm, autoformer, odformer, mpgcn, st_lstm" >&2
  exit 2
fi

DATA_ROOT="${DATA_ROOT:-$ROOT/data_splits/od_minute_review_3way}"
TRAIN_SUBDIR="${TRAIN_SUBDIR:-train}"
VAL_SUBDIR="${VAL_SUBDIR:-val}"
OD_CSV="${OD_CSV:-$ROOT/AD_matrix_trimmed_common.csv}"

COMMON_ARGS=(
  --data_root "$DATA_ROOT"
  --train_subdir "$TRAIN_SUBDIR"
  --val_subdir "$VAL_SUBDIR"
  --window_size "${WINDOW_SIZE:-60}"
  --pred_size "${PRED_SIZE:-30}"
  --hop_size "${HOP_SIZE:-10}"
)

case "$MODEL" in
  gcn_lstm)
    python "$ROOT/train/train_gcn_lstm.py" \
      "${COMMON_ARGS[@]}" \
      --od_csv "$OD_CSV" \
      --time_resolution "${TIME_RESOLUTION:-1}" \
      --batch_size "${BATCH_SIZE:-32}" \
      --max_epochs "${MAX_EPOCHS:-200}" \
      --num_workers "${NUM_WORKERS:-4}"
    ;;
  autoformer)
    python "$ROOT/train/train_autoformer.py" \
      "${COMMON_ARGS[@]}" \
      --time_resolution "${TIME_RESOLUTION:-1}" \
      --batch_size "${BATCH_SIZE:-32}" \
      --max_epochs "${MAX_EPOCHS:-200}" \
      --num_workers "${NUM_WORKERS:-2}"
    ;;
  odformer)
    python "$ROOT/train/train_odformer.py" \
      "${COMMON_ARGS[@]}" \
      --adj_csv "$OD_CSV" \
      --batch_size "${BATCH_SIZE:-8}" \
      --max_epochs "${MAX_EPOCHS:-100}" \
      --num_workers "${NUM_WORKERS:-4}"
    ;;
  mpgcn)
    python "$ROOT/train/train_mpgcn.py" \
      "${COMMON_ARGS[@]}" \
      --od_csv "$OD_CSV" \
      --time_resolution "${TIME_RESOLUTION:-1}" \
      --dynamic_graph_cache "${MPGCN_DYNAMIC_GRAPH_CACHE:-$ROOT/artifacts/mpgcn_dyn_60m.pt}" \
      --dynamic_bin_size "${MPGCN_DYNAMIC_BIN_SIZE:-60}" \
      --batch_size "${BATCH_SIZE:-1}" \
      --max_epochs "${MAX_EPOCHS:-200}" \
      --num_workers "${NUM_WORKERS:-2}" \
      --output_root "${OUTPUT_ROOT:-$ROOT/review_runs}"
    ;;
  st_lstm)
    python "$ROOT/train/train_st_lstm.py" \
      --data_root "$DATA_ROOT" \
      --train_subdir "$TRAIN_SUBDIR" \
      --val_subdir "$VAL_SUBDIR" \
      --H "${WINDOW_SIZE:-60}" \
      --h "${PRED_SIZE:-30}" \
      --pred_size "${PRED_SIZE:-30}" \
      --target_s "${TARGET_S:-16}" \
      --target_e "${TARGET_E:-523}" \
      --batch_size "${BATCH_SIZE:-256}" \
      --max_epochs "${MAX_EPOCHS:-300}" \
      --num_workers "${NUM_WORKERS:-4}"
    ;;
  *)
    echo "Unsupported MODEL=$MODEL" >&2
    exit 2
    ;;
esac
