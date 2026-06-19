#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ -z "${GNN_CKPT:-}" && -z "${MPGCN_CKPT:-}" ]]; then
  echo "Set at least one of GNN_CKPT or MPGCN_CKPT." >&2
  exit 2
fi

python "$ROOT/evaluate/evaluate_full_network.py" \
  --gnn_ckpt "${GNN_CKPT:-}" \
  --mpgcn_ckpt "${MPGCN_CKPT:-}" \
  --data_root "${DATA_ROOT:-$ROOT/data_splits/od_minute_review_3way}" \
  --train_subdir "${TRAIN_SUBDIR:-train}" \
  --test_subdir "${TEST_SUBDIR:-test}" \
  --od_csv "${OD_CSV:-$ROOT/AD_matrix_trimmed_common.csv}" \
  --time_resolution "${TIME_RESOLUTION:-1}" \
  --window_size "${WINDOW_SIZE:-60}" \
  --pred_size "${PRED_SIZE:-30}" \
  --hop_size "${HOP_SIZE:-10}" \
  --graph_gate_tau "${METRO_GATF_GATE_TAU:-0.9}" \
  --mape_eps "${MAPE_EPS:-1e-3}" \
  --mpgcn_dynamic_graph_cache "${MPGCN_DYNAMIC_GRAPH_CACHE:-$ROOT/artifacts/mpgcn_dyn_60m.pt}" \
  --mpgcn_dynamic_bin_size "${MPGCN_DYNAMIC_BIN_SIZE:-60}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --num_workers "${NUM_WORKERS:-0}" \
  --device "${DEVICE:-}" \
  --output_json "${OUTPUT_JSON:-}"
