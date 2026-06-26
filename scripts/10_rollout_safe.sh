#!/usr/bin/env bash
#
# 10_rollout_safe.sh -- SAFE fold rollout on a robot+camera.
#
# SAFETY MODEL (read this):
#   * This script is DRY-RUN by default: it plans a fold and executes it on the
#     MOCK robot with no physical motion (`terafold dry-run-fold`).
#   * Physical hardware NEVER moves unless you explicitly opt in with BOTH:
#         --enable-motion
#         --i-understand-this-moves-hardware
#     A real-motion rollout also honors the STOP file (STOP_TERAFOLD) and the
#     workspace/safety bounds; planning is validated before any motion.
#   * Real motion is performed via `terafold record-demo` (which enforces the
#     two flags above). This script will refuse to pass those flags unless you
#     ALSO set TERAFOLD_REAL_ROBOT=1, as a third guard against accidents.
#
# Usage:
#   scripts/10_rollout_safe.sh [--task FILE] [--robot mock|learm|so101] [--camera mock|opencv]
#   # Real motion (advanced, dangerous):
#   TERAFOLD_REAL_ROBOT=1 scripts/10_rollout_safe.sh --robot so101 --camera opencv \
#       --enable-motion --i-understand-this-moves-hardware
#
set -euo pipefail

TASK="configs/task_fold_towel_half.yaml"
ROBOT="mock"
CAMERA="mock"
ENABLE_MOTION=0
I_UNDERSTAND=0

while [ $# -gt 0 ]; do
    case "$1" in
        --task) TASK="$2"; shift 2 ;;
        --robot) ROBOT="$2"; shift 2 ;;
        --camera) CAMERA="$2"; shift 2 ;;
        --enable-motion) ENABLE_MOTION=1; shift ;;
        --i-understand-this-moves-hardware) I_UNDERSTAND=1; shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ "${ENABLE_MOTION}" -eq 1 ] && [ "${I_UNDERSTAND}" -eq 1 ]; then
    if [ "${TERAFOLD_REAL_ROBOT:-0}" != "1" ]; then
        echo "REFUSING real motion: set TERAFOLD_REAL_ROBOT=1 to confirm." >&2
        exit 3
    fi
    echo "==> REAL-MOTION rollout requested. Hardware WILL move."
    echo "    Ensure the STOP file mechanism (STOP_TERAFOLD) and e-stop are reachable."
    set -x
    terafold record-demo --task "${TASK}" --robot "${ROBOT}" --camera "${CAMERA}" \
        --enable-motion --i-understand-this-moves-hardware
else
    echo "==> SAFE DRY-RUN rollout (no physical motion)."
    echo "    task=${TASK} robot=${ROBOT} camera=${CAMERA}"
    echo "    To move real hardware, pass --enable-motion AND"
    echo "    --i-understand-this-moves-hardware AND set TERAFOLD_REAL_ROBOT=1."
    set -x
    terafold dry-run-fold --task "${TASK}" --robot mock --camera "${CAMERA}"
fi
