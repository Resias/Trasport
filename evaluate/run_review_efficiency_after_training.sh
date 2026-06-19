#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data_splits/od_minute_review_3way}"
STUDY_DIR="${STUDY_DIR:-$ROOT_DIR/review_runs/review_3way_50ep_20260512}"
SEEDS="${SEEDS:-0,1,2}"
WAIT_PID="${WAIT_PID:-}"

mkdir -p "$STUDY_DIR/logs" "$STUDY_DIR/efficiency"

if [[ -n "$WAIT_PID" ]]; then
  echo "[Efficiency watcher] waiting for pid=$WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do
    sleep 60
  done
fi

echo "[Efficiency watcher] running efficiency benchmark"
python evaluate/evaluate_review_efficiency.py \
  --study_dir "$STUDY_DIR" \
  --data_root "$DATA_ROOT" \
  --train_subdir train \
  --test_subdir test \
  --seeds "$SEEDS" \
  --batch_size 1 \
  --num_workers 0 \
  --warmup_batches 2 \
  --max_batches 20 \
  --output_dir "$STUDY_DIR/efficiency"

echo "[Efficiency watcher] complete."
