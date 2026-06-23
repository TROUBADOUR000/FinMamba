#!/usr/bin/env bash
set -euo pipefail

# Always run relative to this project directory.
cd "$(dirname "$0")"

# Paths can be overridden without editing the command, e.g.
# DEVICE=cuda:1 OUTPUT_DIR=outputs/nasdaq bash run_finmamba.sh
STOCK="${STOCK:-nasdaq}"
DEVICE="${DEVICE:-auto}"
DATA_DIR="${DATA_DIR:-data}"
RELATION_DIR="${RELATION_DIR:-${STOCK}_stock_relation}"
OUTPUT_DIR="${OUTPUT_DIR:-.}"

python train_finmamba.py \
  --stock "$STOCK" \
  --data-dir "$DATA_DIR" \
  --relation-dir "$RELATION_DIR" \
  --relation-pattern 'day{index}.pkl' \
  --train-start 2018-01-01 \
  --train-end 2021-12-31 \
  --valid-start 2022-01-01 \
  --valid-end 2022-12-31 \
  --test-start 2023-01-01 \
  --test-end 2023-12-31 \
  --seq-len 20 \
  --market-kernel-sizes 4 10 20 \
  --market-init-sparsity 0.2 \
  --gat-hidden-channels 32 \
  --gat-out-channels auto \
  --gat-layers 2 \
  --gat-heads 2 \
  --mamba-hidden-sizes 64 64 \
  --mamba-num-heads 2 \
  --mamba-output-size 16 \
  --mamba-d-state 128 \
  --mamba-d-conv 2 \
  --mamba-expand 1 \
  --dropout 0.1 \
  --epochs 5 \
  --batch-size 16 \
  --learning-rate 0.005 \
  --weight-decay 1e-7 \
  --hinge-weight 3.0 \
  --mse-weight 1.0 \
  --patience 10 \
  --log-interval 10 \
  --seed 2024 \
  --device "$DEVICE" \
  --output-dir "$OUTPUT_DIR" \
  --checkpoint-name best_model.pth \
  --scores-name scores.csv \
  --prediction-name pred.csv \
  --prediction-layout legacy
