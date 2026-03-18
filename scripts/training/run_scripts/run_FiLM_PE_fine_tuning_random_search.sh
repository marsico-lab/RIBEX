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
: "${BS_MIN:=64}"
: "${BS_MAX:=512}"
: "${FT_EPOCHS_MIN:=10}"
: "${FT_EPOCHS_MAX:=40}"
: "${PE_DIM_MIN:=64}"
: "${PE_DIM_MAX:=1536}"
: "${LR_LOG10_MIN:=-5.5}"
: "${LR_LOG10_MAX:=-3.0}"

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

sample_int() {
  python3 -c "import random; print(random.randint(${1}, ${2}))"
}

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SEARCH_TAG="film_rs_${DATASET}_${LM_SHORT}_seed${SEED}_${TIMESTAMP}"
RESULT_DIR="${REPO_ROOT}/results/random_search/${SEARCH_TAG}"
MANIFEST="${RESULT_DIR}/manifest.tsv"
FILM_ROOT="${STORAGE_ROOT}/data/models/FiLM_PE/lightning_logs"
mkdir -p "${RESULT_DIR}"

cat <<EOF
================================================================================
FiLM PE nested random search
================================================================================
Dataset: ${DATASET}_human_fine-tuning.pkl
Seed: ${SEED}
LM: ${LM_NAME} (${LM_SHORT})
Trials: ${NUM_TRIALS}
Search tag: ${SEARCH_TAG}
Results: ${RESULT_DIR}
================================================================================
EOF

printf "trial\trun_tag\tsearch_tag\tmodel\tlm_name\tdataset\tseed\tbatch_size\tnum_train_epochs\tlearning_rate\tpe_dim\n" > "${MANIFEST}"

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
  BS="$(sample_int "${BS_MIN}" "${BS_MAX}")"
  FT_EPOCHS="$(sample_int "${FT_EPOCHS_MIN}" "${FT_EPOCHS_MAX}")"
  LR="$(sample_log_uniform "${LR_LOG10_MIN}" "${LR_LOG10_MAX}")"
  PE_DIM="$(sample_int "${PE_DIM_MIN}" "${PE_DIM_MAX}")"
  RUN_TAG="${SEARCH_TAG}_trial$(printf '%02d' "${trial}")"

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${trial}" "${RUN_TAG}" "${SEARCH_TAG}" "FiLM_PE" "${LM_NAME}" "${DATASET}" "${SEED}" \
    "${BS}" "${FT_EPOCHS}" "${LR}" "${PE_DIM}" >> "${MANIFEST}"

  JOB_ID="$(sbatch --parsable --dependency=afterok:${SPLIT_JOB_ID} <<EOF
#!/bin/bash
#SBATCH -J FiLM_${LM_SHORT}_rs_${trial}
#SBATCH --output=${REPO_ROOT}/scripts/sbatch_logs/FiLM_${SEARCH_TAG}_trial${trial}_%j.txt
#SBATCH --error=${REPO_ROOT}/scripts/sbatch_logs/FiLM_${SEARCH_TAG}_trial${trial}_%j.txt
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --nice=10000

source /path/to/home/.bashrc
source /path/to/miniconda3/bin/activate rbp_ig_lustre
cd ${REPO_ROOT}

python3 scripts/training/train.py \
  -M FiLM_PE \
  -D ${DEVICES} \
  -DS ${DATASET}_human_fine-tuning.pkl \
  -lm ${LM_NAME} \
  -ef ${EMB} \
  -S ${SEED} \
  -e ${FT_EPOCHS} \
  -bs ${BS} \
  -lr ${LR} \
  --pe_dim ${PE_DIM} \
  --patience 5 \
  --keep_small_val_batches \
  --run_tag ${RUN_TAG}
EOF
)"
  TRIAL_JOB_IDS+=("${JOB_ID}")
  echo "Trial ${trial}/${NUM_TRIALS}: job ${JOB_ID} (${RUN_TAG})"
done

TRIAL_DEPENDENCY="$(IFS=:; echo "${TRIAL_JOB_IDS[*]}")"
SPLIT_FILE="${STORAGE_ROOT}/data/splits/${DATASET}_human_fine-tuning_film_pe_seed_${SEED}_${LM_NAME}__ft_val.tsv"
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
  --search-root ${FILM_ROOT} \
  --search-tag ${SEARCH_TAG} \
  --split-file ${SPLIT_FILE} \
  --dataset-path ${DATASET_PATH} \
  --manifest ${MANIFEST} \
  --output-dir ${RESULT_DIR}
EOF
)"

cat <<EOF
================================================================================
Submitted FiLM PE random search
Split job: ${SPLIT_JOB_ID}
Evaluation job: ${EVAL_JOB_ID}
Manifest: ${MANIFEST}
Leaderboard target: ${RESULT_DIR}/random_search_leaderboard.tsv
================================================================================
EOF
