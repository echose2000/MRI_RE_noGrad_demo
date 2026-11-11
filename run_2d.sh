#!/usr/bin/env bash
# Run the 2D reconstruction experiment from the repository root.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Override these variables when launching, for example:
#   DEVICE=cpu STEPS=100 RECURSIVE_STEPS=2 ./run_2d.sh
CONFIG="${CONFIG:-results/config/2d_img.yaml}"
DATA="${DATA:-}"
DEVICE="${DEVICE:-cuda:6}"
MODEL="${MODEL:-}"
STEPS="${STEPS:-5000}"
EVAL_EVERY="${EVAL_EVERY:-100}"
# Total model calls per reconstruction.
RECURSIVE_STEPS="${RECURSIVE_STEPS:-50}"
LR="${LR:-0.001}"
RESULT_DIR="${RESULT_DIR:-results/2d_3}"
SAVE_RESULTS="${SAVE_RESULTS:-1}"
NO_SHOW="${NO_SHOW:-0}"

ARGS=(--config "$CONFIG" --eval-every "$EVAL_EVERY"
      --recursive-steps "$RECURSIVE_STEPS" --result-dir "$RESULT_DIR")

[[ -n "$DATA" ]] && ARGS+=(--data "$DATA")
[[ -n "$DEVICE" ]] && ARGS+=(--device "$DEVICE")
[[ -n "$MODEL" ]] && ARGS+=(--model "$MODEL")
[[ -n "$STEPS" ]] && ARGS+=(--steps "$STEPS")
[[ -n "$LR" ]] && ARGS+=(--lr "$LR")

if [[ "$SAVE_RESULTS" == "1" || "$SAVE_RESULTS" == "true" ]]; then
    ARGS+=(--save-results)
else
    ARGS+=(--no-save-results)
fi

[[ "$NO_SHOW" == "1" || "$NO_SHOW" == "true" ]] && ARGS+=(--no-show)

if command -v uv >/dev/null 2>&1; then
    exec uv run python test_2D.py "${ARGS[@]}"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 test_2D.py "${ARGS[@]}"
else
    echo "Error: neither 'uv' nor 'python3' was found in PATH." >&2
    exit 1
fi
