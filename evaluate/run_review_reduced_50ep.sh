#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data_splits/od_minute_review_3way}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/review_runs}"
STUDY_NAME="${STUDY_NAME:-review_3way_50ep_20260512}"
SEEDS="${SEEDS:-0,1,2}"
NUM_WORKERS="${NUM_WORKERS:-2}"
MAX_EPOCHS="${MAX_EPOCHS:-50}"
RUN_STAT_BASELINES="${RUN_STAT_BASELINES:-1}"
ARIMA_MAX_PAIRS="${ARIMA_MAX_PAIRS:-0}"
ARIMA_PAIR_CHUNK_SIZE="${ARIMA_PAIR_CHUNK_SIZE:-0}"
SKIP_ARIMA="${SKIP_ARIMA:-0}"
MPGCN_RESUME_LAST="${MPGCN_RESUME_LAST:-0}"
MPGCN_EARLY_STOP_PATIENCE="${MPGCN_EARLY_STOP_PATIENCE:-0}"
MPGCN_EARLY_STOP_MIN_DELTA="${MPGCN_EARLY_STOP_MIN_DELTA:-0.0}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONUNBUFFERED=1

STUDY_DIR="$RUN_ROOT/$STUDY_NAME"
mkdir -p "$STUDY_DIR/logs" "$STUDY_DIR/artifacts" "$STUDY_DIR/statistical_baselines"

echo "[Reduced review rerun] root       = $ROOT_DIR"
echo "[Reduced review rerun] data_root  = $DATA_ROOT"
echo "[Reduced review rerun] run_root   = $RUN_ROOT"
echo "[Reduced review rerun] study_name = $STUDY_NAME"
echo "[Reduced review rerun] seeds      = $SEEDS"
echo "[Reduced review rerun] max_epochs = $MAX_EPOCHS"
echo "[Reduced review rerun] stat_base  = $RUN_STAT_BASELINES"
echo "[Reduced review rerun] arima_pairs= $ARIMA_MAX_PAIRS"
echo "[Reduced review rerun] arima_chunk= $ARIMA_PAIR_CHUNK_SIZE"

STAT_PID=""
STAT_OUTPUT="$STUDY_DIR/statistical_baselines/ha_arima_test_metrics.json"
if [[ "$RUN_STAT_BASELINES" == "1" ]]; then
  STAT_COMPLETE="0"
  if [[ -f "$STAT_OUTPUT" ]]; then
    STAT_COMPLETE=$(python - "$STAT_OUTPUT" "$SKIP_ARIMA" <<'PY'
import json
import sys

path, skip_arima = sys.argv[1], sys.argv[2]
try:
    with open(path, "r", encoding="utf-8") as fp:
        payload = json.load(fp)
except Exception:
    print("0")
    raise SystemExit

ha_done = payload.get("status", {}).get("HA") == "complete" or "HA" in payload
arima_done = payload.get("status", {}).get("ARIMA") == "complete" or "ARIMA" in payload
print("1" if ha_done and (skip_arima == "1" or arima_done) else "0")
PY
)
  fi
  if [[ "$STAT_COMPLETE" == "1" ]]; then
    echo "[0/6] Statistical baselines already complete: $STAT_OUTPUT"
  else
    echo "[0/6] Statistical baselines: HA + ARIMA (CPU background)"
    STAT_ARGS=(
      --data_root "$DATA_ROOT"
      --train_subdir train
      --test_subdir test
      --window_size 60
      --pred_size 30
      --hop_size 10
      --time_resolution 1
      --arima_max_pairs "$ARIMA_MAX_PAIRS"
      --arima_pair_chunk_size "$ARIMA_PAIR_CHUNK_SIZE"
      --output_json "$STAT_OUTPUT"
    )
    if [[ "$SKIP_ARIMA" == "1" ]]; then
      STAT_ARGS+=(--skip_arima)
    fi
    python evaluate/statistical_baselines_review.py "${STAT_ARGS[@]}" \
      > "$STUDY_DIR/logs/statistical_baselines.log" 2>&1 &
    STAT_PID=$!
    echo "$STAT_PID" > "$STUDY_DIR/statistical_baselines/statistical_baselines.pid"
  fi
else
  echo "[0/6] Statistical baselines skipped for this resume run."
fi

echo "[1/6] Proposed S8_gate"
python train/train_abligation.py \
  --data_root "$DATA_ROOT" \
  --train_subdir train \
  --val_subdir val \
  --experiment_plan progressive_core \
  --stages S8_gate \
  --seeds "$SEEDS" \
  --study_name "$STUDY_NAME" \
  --output_root "$RUN_ROOT" \
  --batch_size 2 \
  --num_workers "$NUM_WORKERS" \
  --max_epochs "$MAX_EPOCHS" \
  --disable_wandb \
  --resume_if_complete

for SEED in ${SEEDS//,/ }; do
  echo "[2/6] ODFormer seed=$SEED"
  python train/train_odformer.py \
    --data_root "$DATA_ROOT" \
    --train_subdir train \
    --val_subdir val \
    --seed "$SEED" \
    --output_root "$RUN_ROOT" \
    --study_name "$STUDY_NAME" \
    --batch_size 8 \
    --num_workers "$NUM_WORKERS" \
    --max_epochs "$MAX_EPOCHS" \
    --disable_wandb \
    --resume_if_complete
done

for SEED in ${SEEDS//,/ }; do
  echo "[3/6] GCN-LSTM seed=$SEED"
  python train/train_gcn_lstm.py \
    --data_root "$DATA_ROOT" \
    --train_subdir train \
    --val_subdir val \
    --seed "$SEED" \
    --output_root "$RUN_ROOT" \
    --study_name "$STUDY_NAME" \
    --batch_size 16 \
    --num_workers "$NUM_WORKERS" \
    --max_epochs "$MAX_EPOCHS" \
    --disable_wandb \
    --resume_if_complete
done

for SEED in ${SEEDS//,/ }; do
  echo "[4/6] Autoformer seed=$SEED"
  python train/train_autoformer.py \
    --data_root "$DATA_ROOT" \
    --train_subdir train \
    --val_subdir val \
    --seed "$SEED" \
    --output_root "$RUN_ROOT" \
    --study_name "$STUDY_NAME" \
    --batch_size 16 \
    --num_workers "$NUM_WORKERS" \
    --max_epochs "$MAX_EPOCHS" \
    --disable_wandb \
    --resume_if_complete
done

for SEED in ${SEEDS//,/ }; do
  echo "[5/6] MPGCN seed=$SEED"
  MPGCN_EXTRA_ARGS=()
  if [[ "$MPGCN_RESUME_LAST" == "1" ]]; then
    CKPT_DIR="$STUDY_DIR/MPGCN/seed_$SEED/checkpoints"
    LAST_CKPT="$(find "$CKPT_DIR" -maxdepth 1 -name 'last*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2- || true)"
    if [[ -n "$LAST_CKPT" ]]; then
      echo "[Resume] MPGCN seed=$SEED from $LAST_CKPT"
      MPGCN_EXTRA_ARGS+=(--resume_from_checkpoint "$LAST_CKPT")
    fi
  fi
  if [[ "$MPGCN_EARLY_STOP_PATIENCE" -gt 0 ]]; then
    echo "[EarlyStopping] MPGCN seed=$SEED patience=$MPGCN_EARLY_STOP_PATIENCE min_delta=$MPGCN_EARLY_STOP_MIN_DELTA"
    MPGCN_EXTRA_ARGS+=(--early_stop_patience "$MPGCN_EARLY_STOP_PATIENCE")
    MPGCN_EXTRA_ARGS+=(--early_stop_min_delta "$MPGCN_EARLY_STOP_MIN_DELTA")
  fi
  python train/train_mpgcn.py \
    --data_root "$DATA_ROOT" \
    --train_subdir train \
    --val_subdir val \
    --seed "$SEED" \
    --output_root "$RUN_ROOT" \
    --study_name "$STUDY_NAME" \
    --dynamic_graph_cache "$STUDY_DIR/artifacts/mpgcn_dyn_60m.pt" \
    --batch_size 1 \
    --num_workers "$NUM_WORKERS" \
    --max_epochs "$MAX_EPOCHS" \
    --disable_wandb \
    --resume_if_complete \
    "${MPGCN_EXTRA_ARGS[@]}"
done

echo "[6/6] Test-set evaluation for neural checkpoints"
python evaluate/evaluate_review_checkpoints.py \
  --study_dir "$STUDY_DIR" \
  --data_root "$DATA_ROOT" \
  --train_subdir train \
  --test_subdir test \
  --seeds "$SEEDS" \
  --batch_size 1 \
  --num_workers 0 \
  --output_dir "$STUDY_DIR/evaluation"

echo "[7/7] Computational efficiency benchmark"
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

if [[ -n "$STAT_PID" ]]; then
  echo "[Reduced review rerun] waiting for HA/ARIMA process pid=$STAT_PID"
  wait "$STAT_PID"
fi

echo "[Reduced review rerun] complete."
