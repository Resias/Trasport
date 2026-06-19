#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

DATA_ROOT="${DATA_ROOT:-$ROOT/data_splits/od_minute_review_3way}"
TRAIN_SUBDIR="${TRAIN_SUBDIR:-train}"
VAL_SUBDIR="${VAL_SUBDIR:-val}"
OD_CSV="${OD_CSV:-$ROOT/AD_matrix_trimmed_common.csv}"
STATION_LATLON_CSV="${STATION_LATLON_CSV:-$ROOT/ad_station_latlon.csv}"

python "$ROOT/train/train_graph.py" \
  --data_root "$DATA_ROOT" \
  --train_subdir "$TRAIN_SUBDIR" \
  --val_subdir "$VAL_SUBDIR" \
  --od_csv "$OD_CSV" \
  --station_latlon_csv "$STATION_LATLON_CSV" \
  --time_resolution "${TIME_RESOLUTION:-1}" \
  --window_size "${WINDOW_SIZE:-60}" \
  --pred_size "${PRED_SIZE:-30}" \
  --hop_size "${HOP_SIZE:-10}" \
  --batch_size "${METRO_GATF_BATCH_SIZE:-2}" \
  --gat_heads "${METRO_GATF_GAT_HEADS:-6}" \
  --node_feat_dim "${METRO_GATF_NODE_FEAT_DIM:-16}" \
  --gat_hidden "${METRO_GATF_GAT_HIDDEN:-64}" \
  --decode_num_layers "${METRO_GATF_DECODE_LAYERS:-2}" \
  --lr "${METRO_GATF_LR:-1e-4}" \
  --max_epochs "${METRO_GATF_MAX_EPOCHS:-200}" \
  --lambda_gate "${METRO_GATF_LAMBDA_GATE:-1.0}" \
  --gate_tau "${METRO_GATF_GATE_TAU:-0.9}" \
  --pos_weight_clip "${METRO_GATF_POS_WEIGHT_CLIP:-10.0}" \
  --num_workers "${NUM_WORKERS:-2}" \
  --wandb_project "${WANDB_PROJECT:-metro-GATrasformer-od-week-latlon}"
