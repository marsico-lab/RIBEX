#!/usr/bin/env bash

# Sequentially launch all fine-tuning and pre-training sweep scripts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -a JOB_SCRIPTS=(
    "tune_finetune_lora_alpha.sh"
    "tune_finetune_lora_dropout.sh"
    "tune_finetune_lora_learning_rate.sh"
    "tune_finetune_lora_rank.sh"
    #"tune_finetune_lora_target_modules.sh"
    "tune_finetune_lora_weight_decay.sh"
    "tune_finetune_pe_dim.sh"
    # "tune_lora_alpha.sh"
    # "tune_lora_dropout.sh"
    # "tune_lora_learning_rate.sh"
    # "tune_lora_rank.sh"
    # "tune_lora_target_modules.sh"
    # "tune_lora_weight_decay.sh"
)

pids=()

for script in "${JOB_SCRIPTS[@]}"; do
    echo ">>> Running ${script}"
    bash "${SCRIPT_DIR}/${script}" &
    pids+=("$!")
done

status=0
set +e
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        status=$?
    fi
done
set -e

exit "${status}"
