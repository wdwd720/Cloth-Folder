#!/usr/bin/env bash
#
# 09_train_act.sh -- Prepare/launch ACT policy training on an exported
# LeRobot dataset via `terafold train-act`.
#
# By default this only PREPARES and prints the exact training command
# (no GPU/lerobot needed). Pass --run to actually launch training, which
# requires the lerobot extra and typically a GPU:
#
#     pip install -e '.[lerobot]'
#
# Usage:
#   scripts/09_train_act.sh [DATASET_DIR] [OUT_DIR] [--run]
#
set -euo pipefail

DATASET="${1:-data/lerobot/fold_towel_half}"
OUT="${2:-runs/act}"
RUN_FLAG=""
if [ "${3:-}" = "--run" ]; then
    RUN_FLAG="--run"
fi

echo "==> ACT training wrapper (terafold train-act)"
echo "    dataset : ${DATASET}"
echo "    out     : ${OUT}"
if [ -n "${RUN_FLAG}" ]; then
    echo "    mode    : RUN (requires lerobot + GPU; pip install -e '.[lerobot]')"
else
    echo "    mode    : DRY (prints the command only; pass --run to launch)"
fi

set -x
terafold train-act --dataset "${DATASET}" --out "${OUT}" ${RUN_FLAG}
