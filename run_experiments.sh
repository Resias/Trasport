#!/usr/bin/env bash

WANDB_PROJECT=${WANDB_PROJECT:-metro-gnn}

declare -a GNN_HIDDEN_LIST=(32 64 128)
declare -a RNN_HIDDEN_LIST=(32 64 128)
declare -a WEEKDAY_EMB_LIST=(8 16 32)
declare -a LR_LIST=(1e-3 5e-4)

for gnn_hidden in "${GNN_HIDDEN_LIST[@]}"; do
  for rnn_hidden in "${RNN_HIDDEN_LIST[@]}"; do
    for weekday_emb in "${WEEKDAY_EMB_LIST[@]}"; do
      for lr in "${LR_LIST[@]}"; do
        echo "Running: GNN=${gnn_hidden}, RNN=${rnn_hidden}, WeekdayEmb=${weekday_emb}, LR=${lr}"
        python train.py \
          --lr "$lr" \
          --gnn_hidden "$gnn_hidden" \
          --rnn_hidden "$rnn_hidden" \
          --weekday_emb_dim "$weekday_emb" \
          --wandb_project "$WANDB_PROJECT"
      done
    done
  done
done
