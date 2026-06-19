#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STUDY_DIR="${STUDY_DIR:-$ROOT_DIR/review_runs/review_3way_20260512}"

echo "[Review rerun status] study_dir = $STUDY_DIR"

if [[ -f "$STUDY_DIR/full_rerun.pid" ]]; then
  PID="$(cat "$STUDY_DIR/full_rerun.pid")"
  echo
  echo "[Process]"
  ps -p "$PID" -o pid,stat,etime,cmd || true
fi

echo
echo "[Active training processes]"
pgrep -af 'run_review_reexperiments|run_review_reduced_50ep|train_ablation|train_abligation|train_odformer|train_mpgcn|train_gcn_lstm|train_autoformer|statistical_baselines_review|evaluate_review_checkpoints' || true

echo
echo "[Latest log]"
if [[ -f "$STUDY_DIR/full_rerun.logpath" ]]; then
  LOG="$(cat "$STUDY_DIR/full_rerun.logpath")"
  echo "$LOG"
  tail -n "${TAIL_LINES:-60}" "$LOG" || true
elif [[ -f "$STUDY_DIR/reduced_50ep.logpath" ]]; then
  LOG="$(cat "$STUDY_DIR/reduced_50ep.logpath")"
  echo "$LOG"
  tail -n "${TAIL_LINES:-60}" "$LOG" || true
else
  echo "No logpath file found."
fi

echo
echo "[Completed run_result.json files]"
find "$STUDY_DIR" -name run_result.json -print | sort || true

echo
echo "[Statistical baseline output]"
find "$STUDY_DIR" -path '*statistical_baselines*' -type f -print | sort || true

echo
echo "[GPU]"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
