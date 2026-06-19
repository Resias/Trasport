#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/data_splits/od_minute_review_3way}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/review_runs}"
STUDY_NAME="${STUDY_NAME:-review_3way_20260512}"
SEEDS="${SEEDS:-0,1,2}"
NUM_WORKERS="${NUM_WORKERS:-2}"

export WANDB_MODE="${WANDB_MODE:-disabled}"
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_ROOT/$STUDY_NAME/logs" "$RUN_ROOT/$STUDY_NAME/artifacts"

echo "[Review rerun] root       = $ROOT_DIR"
echo "[Review rerun] data_root  = $DATA_ROOT"
echo "[Review rerun] run_root   = $RUN_ROOT"
echo "[Review rerun] study_name = $STUDY_NAME"
echo "[Review rerun] seeds      = $SEEDS"

echo "[1/5] Proposed S8_gate"
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
  --max_epochs 300 \
  --disable_wandb \
  --resume_if_complete

for SEED in ${SEEDS//,/ }; do
  echo "[2/5] ODFormer seed=$SEED"
  python train/train_odformer.py \
    --data_root "$DATA_ROOT" \
    --train_subdir train \
    --val_subdir val \
    --seed "$SEED" \
    --output_root "$RUN_ROOT" \
    --study_name "$STUDY_NAME" \
    --batch_size 8 \
    --num_workers "$NUM_WORKERS" \
    --max_epochs 100 \
    --disable_wandb \
    --resume_if_complete
done

for SEED in ${SEEDS//,/ }; do
  echo "[3/5] MPGCN seed=$SEED"
  python train/train_mpgcn.py \
    --data_root "$DATA_ROOT" \
    --train_subdir train \
    --val_subdir val \
    --seed "$SEED" \
    --output_root "$RUN_ROOT" \
    --study_name "$STUDY_NAME" \
    --dynamic_graph_cache "$RUN_ROOT/$STUDY_NAME/artifacts/mpgcn_dyn_60m.pt" \
    --batch_size 1 \
    --num_workers "$NUM_WORKERS" \
    --max_epochs 200 \
    --disable_wandb \
    --resume_if_complete
done

for SEED in ${SEEDS//,/ }; do
  echo "[4/5] GCN-LSTM seed=$SEED"
  python train/train_gcn_lstm.py \
    --data_root "$DATA_ROOT" \
    --train_subdir train \
    --val_subdir val \
    --seed "$SEED" \
    --output_root "$RUN_ROOT" \
    --study_name "$STUDY_NAME" \
    --batch_size 16 \
    --num_workers "$NUM_WORKERS" \
    --max_epochs 200 \
    --disable_wandb \
    --resume_if_complete
done

for SEED in ${SEEDS//,/ }; do
  echo "[5/5] Autoformer seed=$SEED"
  python train/train_autoformer.py \
    --data_root "$DATA_ROOT" \
    --train_subdir train \
    --val_subdir val \
    --seed "$SEED" \
    --output_root "$RUN_ROOT" \
    --study_name "$STUDY_NAME" \
    --batch_size 16 \
    --num_workers "$NUM_WORKERS" \
    --max_epochs 200 \
    --disable_wandb \
    --resume_if_complete
done

echo "[Review rerun] training phase complete."
