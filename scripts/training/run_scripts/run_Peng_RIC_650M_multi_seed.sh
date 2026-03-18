#!/usr/bin/env bash

# Launch Peng pre-training + fine-tuning runs for multiple seeds.
# This wraps run_Peng_RIC_650M.sh (which runs both stages in one Slurm job)
# and queues one job per seed.
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
    echo ">>> Queuing Peng pre-train+finetune with seed ${seed}"
    SEED="${seed}" bash "${SCRIPT_DIR}/run_Peng_RIC_650M.sh" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=$?
    fi
done

exit "${status}"

