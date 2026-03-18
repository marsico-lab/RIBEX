#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STORAGE_ROOT="${REPOSITORY:-/path/to/RBP_IG_storage}"

: "${DATASET:=RIC}"
: "${SEED:=2023}"
: "${LM_NAME:=esm2_t33_650M_UR50D}"
: "${EMB:=RIC}"
: "${NUM_TRIALS:=20}"
: "${DEVICES:=0}"
: "${GLOBAL_BS_MIN:=64}"
: "${GLOBAL_BS_MAX:=512}"
: "${MICRO_BS_CHOICES:=1 2 4}"
: "${FT_EPOCHS_MIN:=5}"
: "${FT_EPOCHS_MAX:=15}"
: "${PE_DIM_MIN:=128}"
: "${PE_DIM_MAX:=1024}"
: "${LORA_R_MIN:=2}"
: "${LORA_R_MAX:=12}"
: "${LORA_ALPHA_MIN:=0.25}"
: "${LORA_ALPHA_MAX:=4.0}"
: "${LORA_DROPOUT_MIN:=0.0}"
: "${LORA_DROPOUT_MAX:=0.5}"
: "${LR_LOG10_MIN:=-4.5}"
: "${LR_LOG10_MAX:=-3.0}"
: "${WD_LOG10_MIN:=-6.0}"
: "${WD_LOG10_MAX:=-2.0}"

case "${LM_NAME}" in
  esm1b_t33_650M_UR50S) LM_SHORT="ESM1b_650M" ;;
  esm2_t6_8M_UR50D) LM_SHORT="ESM2_8M" ;;
  esm2_t12_35M_UR50D) LM_SHORT="ESM2_35M" ;;
  esm2_t30_150M_UR50D) LM_SHORT="ESM2_150M" ;;
  esm2_t33_650M_UR50D) LM_SHORT="ESM2_650M" ;;
  esm2_t36_3B_UR50D) LM_SHORT="ESM2_3B" ;;
  esm2_t48_15B_UR50D) LM_SHORT="ESM2_15B" ;;
  protT5_xl_uniref50) LM_SHORT="ProtT5_XL" ;;
  *) LM_SHORT="MODEL" ;;
esac

sample_log_uniform() {
  python3 -c "import random; print(f'{10**random.uniform(${1}, ${2}):.8f}')"
}

sample_float_uniform() {
  python3 -c "import random; print(f'{random.uniform(${1}, ${2}):.6f}')"
}

sample_int() {
  python3 -c "import random; print(random.randint(${1}, ${2}))"
}

sample_choice() {
  python3 - "$@" <<'PY'
import random
import sys
choices = sys.argv[1:]
print(random.choice(choices))
PY
}

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SEARCH_TAG="lora_rs_${DATASET}_${LM_SHORT}_seed${SEED}_${TIMESTAMP}"
RESULT_DIR="${REPO_ROOT}/results/random_search/${SEARCH_TAG}"
MANIFEST="${RESULT_DIR}/manifest.tsv"
mkdir -p "${RESULT_DIR}"

cat <<EOF
================================================================================
LoRA nested random search
================================================================================
Dataset: ${DATASET}_human_fine-tuning.pkl
Seed: ${SEED}
LM: ${LM_NAME} (${LM_SHORT})
Trials: ${NUM_TRIALS}
Search tag: ${SEARCH_TAG}
Results: ${RESULT_DIR}
================================================================================
EOF

printf "trial\trun_tag\tsearch_tag\tmodel\tlm_name\tdataset\tseed\tglobal_batch_size\tmicro_batch_size\tnum_train_epochs\tlora_learning_rate\tpe_dim\tlora_r\tlora_alpha\tlora_dropout\tlora_weight_decay\n" > "${MANIFEST}"

SPLIT_JOB_ID="$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH -J split_${DATASET}_${SEED}
#SBATCH --output=${REPO_ROOT}/scripts/sbatch_logs/split_${DATASET}_${SEED}_%j.txt
#SBATCH --error=${REPO_ROOT}/scripts/sbatch_logs/split_${DATASET}_${SEED}_%j.txt
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000

source /path/to/home/.bashrc
source /path/to/miniconda3/bin/activate rbp_ig_lustre
cd ${REPO_ROOT}
bash scripts/data_util/generate_shared_splits_any.sh ${DATASET} ${SEED}
EOF
)"

echo "Split generation job: ${SPLIT_JOB_ID}"

TRIAL_JOB_IDS=()
for trial in $(seq 1 "${NUM_TRIALS}"); do
  GLOBAL_BS="$(sample_int "${GLOBAL_BS_MIN}" "${GLOBAL_BS_MAX}")"
  MICRO_BS="$(sample_choice ${MICRO_BS_CHOICES})"
  FT_EPOCHS="$(sample_int "${FT_EPOCHS_MIN}" "${FT_EPOCHS_MAX}")"
  LORA_LR="$(sample_log_uniform "${LR_LOG10_MIN}" "${LR_LOG10_MAX}")"
  PE_DIM="$(sample_int "${PE_DIM_MIN}" "${PE_DIM_MAX}")"
  LORA_R="$(sample_int "${LORA_R_MIN}" "${LORA_R_MAX}")"
  LORA_ALPHA="$(sample_float_uniform "${LORA_ALPHA_MIN}" "${LORA_ALPHA_MAX}")"
  LORA_DROPOUT="$(sample_float_uniform "${LORA_DROPOUT_MIN}" "${LORA_DROPOUT_MAX}")"
  LORA_WD="$(sample_log_uniform "${WD_LOG10_MIN}" "${WD_LOG10_MAX}")"
  RUN_TAG="${SEARCH_TAG}_trial$(printf '%02d' "${trial}")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${trial}" "${RUN_TAG}" "${SEARCH_TAG}" "Lora" "${LM_NAME}" "${DATASET}" "${SEED}" \
    "${GLOBAL_BS}" "${MICRO_BS}" "${FT_EPOCHS}" "${LORA_LR}" "${PE_DIM}" "${LORA_R}" \
    "${LORA_ALPHA}" "${LORA_DROPOUT}" "${LORA_WD}" >> "${MANIFEST}"

  JOB_ID="$(sbatch --parsable --dependency=afterok:${SPLIT_JOB_ID} <<EOF
#!/bin/bash
#SBATCH -J LoRA_${LM_SHORT}_rs_${trial}
#SBATCH --output=${REPO_ROOT}/scripts/sbatch_logs/LoRA_${SEARCH_TAG}_trial${trial}_%j.txt
#SBATCH --error=${REPO_ROOT}/scripts/sbatch_logs/LoRA_${SEARCH_TAG}_trial${trial}_%j.txt
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --constraint=a100_80gb
#SBATCH --nice=10000

source /path/to/home/.bashrc
source /path/to/miniconda3/bin/activate rbp_ig_lustre
cd ${REPO_ROOT}

python3 scripts/training/train.py \
  -M Lora \
  -D ${DEVICES} \
  -DS ${DATASET}_human_fine-tuning.pkl \
  -lm ${LM_NAME} \
  -ef ${EMB} \
  -S ${SEED} \
  -bs ${GLOBAL_BS} \
  --lora_per_device_bs ${MICRO_BS} \
  --lora_learning_rate ${LORA_LR} \
  --lora_num_train_epochs ${FT_EPOCHS} \
  --pe_dim ${PE_DIM} \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --lora_dropout ${LORA_DROPOUT} \
  --lora_weight_decay ${LORA_WD} \
  --patience 10 \
  --run_tag ${RUN_TAG}
EOF
)"
  TRIAL_JOB_IDS+=("${JOB_ID}")
  echo "Trial ${trial}/${NUM_TRIALS}: job ${JOB_ID} (${RUN_TAG})"
done

TRIAL_DEPENDENCY="$(IFS=:; echo "${TRIAL_JOB_IDS[*]}")"
SPLIT_FILE="${STORAGE_ROOT}/data/splits/${DATASET}_human_fine-tuning_lora_seed_${SEED}_${LM_NAME}__ft_val.tsv"
DATASET_PATH="${STORAGE_ROOT}/data/data_sets/${DATASET}_human_fine-tuning.pkl"

EVAL_JOB_ID="$(sbatch --parsable --dependency=afterany:${TRIAL_DEPENDENCY} <<EOF
#!/bin/bash
#SBATCH -J eval_${SEARCH_TAG}
#SBATCH --output=${REPO_ROOT}/scripts/sbatch_logs/eval_${SEARCH_TAG}_%j.txt
#SBATCH --error=${REPO_ROOT}/scripts/sbatch_logs/eval_${SEARCH_TAG}_%j.txt
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -p cpu_p
#SBATCH --qos=cpu_normal
#SBATCH --nice=10000

source /path/to/home/.bashrc
source /path/to/miniconda3/bin/activate rbp_ig_lustre
cd ${REPO_ROOT}

python3 scripts/training/evaluate_random_search_nested_holdout.py \
  --search-root ${REPO_ROOT} \
  --search-tag ${SEARCH_TAG} \
  --split-file ${SPLIT_FILE} \
  --dataset-path ${DATASET_PATH} \
  --manifest ${MANIFEST} \
  --output-dir ${RESULT_DIR}
EOF
)"

cat <<EOF
================================================================================
Submitted LoRA random search
Split job: ${SPLIT_JOB_ID}
Evaluation job: ${EVAL_JOB_ID}
Manifest: ${MANIFEST}
Leaderboard target: ${RESULT_DIR}/random_search_leaderboard.tsv
================================================================================
EOF
