#!/usr/bin/env bash

# Launch fine-tuning runs for multiple seeds using the tuned hyperparameters.
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
    echo ">>> Queuing fine-tuning with seed ${seed}"
    SEED="${seed}" bash "${SCRIPT_DIR}/train_fine_tuning.sh" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=$?
    fi
done

exit "${status}"
