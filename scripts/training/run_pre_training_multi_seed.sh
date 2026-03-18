#!/usr/bin/env bash

# Launch pre-training runs for multiple seeds so LoRA adapters are available for fine-tuning.
set -euo pipefail

SEEDS=(
    2842
    5923
    8347
    1290
    4751
    7634
    9821
    1234
    4567
    8901
    1357
    2468
    3579
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -a pids=()

for seed in "${SEEDS[@]}"; do
    echo ">>> Queuing pre-training with seed ${seed}"
    SEED="${seed}" bash "${SCRIPT_DIR}/train_pre-training.sh" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=$?
    fi
done

exit "${status}"
