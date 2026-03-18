#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Reproduction Script for Sequence and PE Explainability
# =============================================================================
# This script first regenerates the base predictions, alanine scans, and PE scans
# from a LoRA checkpoint, then runs the downstream alanine-scan summaries and the
# full PE scan clustering / enrichment workflow on the newly generated scan files.
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Configuration ------------------------------------------------------------
: "${CHECKPOINT:=/path/to/RBP_IG_storage/data/models/Lora/final_checkpoints/<checkpoint_dir>}"
: "${OUTPUT_DIR:=${REPO_ROOT}/pe_scan_analysis_IDR_val_repro}"
: "${SPLIT:=val}"
: "${BATCH_SIZE:=8}"
: "${PRECISION:=fp16}"
: "${PDF_NAME:=pe_scan_tsne_repro}"
: "${FINAL_PDF_NAME:=pe_tsne_clusters_enrichment_repro_v2}"
: "${TOP_TERMS:=5}"
: "${TOP_N_CLUSTERS:=10}"
: "${ALANINE_WINDOW_SIZE:=20}"
: "${ALANINE_MAX_LENGTH:=1200}"
: "${ALANINE_MAX_SAMPLES:=}"
: "${PE_SCAN_TARGET:=0.0}"
: "${INFER_DEVICE:=}"
: "${OVERRIDE_DATASET:=}"
: "${GENE_IDS_FILE:=}"
: "${LIMIT:=}"
: "${PLDDT_PICKLE:=}"

# Derived paths
BASE_PREDICTIONS_FILE="${OUTPUT_DIR}/base_predictions.tsv"
ALANINE_SCAN_FILE="${OUTPUT_DIR}/alanine_scan.tsv"
PE_SCAN_FILE="${OUTPUT_DIR}/pe_scan.tsv"
ALANINE_ANALYSIS_DIR="${OUTPUT_DIR}/alanine_scan_analysis"
NODE_SENSITIVITY="${OUTPUT_DIR}/node_pe_sensitivity.tsv"
TSNE_COORDS="${OUTPUT_DIR}/tsne_coordinates.npy"
CLUSTERED_NODES="${OUTPUT_DIR}/clustered_nodes.tsv"
CLUSTER_SUMMARY="${OUTPUT_DIR}/cluster_summary.tsv"
ENRICHMENT_SUMMARY="${OUTPUT_DIR}/enrichment/enrichment_summary_top${TOP_TERMS}.tsv"

echo "Creating output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

if [ ! -d "${CHECKPOINT}" ]; then
    echo "Error: checkpoint directory not found: ${CHECKPOINT}"
    exit 1
fi

echo ""
echo ">>> Step 1: Running LoRA inference to generate predictions, alanine scans, and PE scans..."
INFERENCE_CMD=(
    python3 "${REPO_ROOT}/scripts/inference/run_lora_inference.py"
    --checkpoint "${CHECKPOINT}"
    --split "${SPLIT}"
    --batch-size "${BATCH_SIZE}"
    --precision "${PRECISION}"
    --output "${BASE_PREDICTIONS_FILE}"
    --alanine-scan
    --alanine-output "${ALANINE_SCAN_FILE}"
    --alanine-window-size "${ALANINE_WINDOW_SIZE}"
    --alanine-max-length "${ALANINE_MAX_LENGTH}"
    --pe-scan
    --pe-scan-output "${PE_SCAN_FILE}"
    --pe-scan-target "${PE_SCAN_TARGET}"
)

if [ -n "${INFER_DEVICE}" ]; then
    INFERENCE_CMD+=(--device "${INFER_DEVICE}")
fi
if [ -n "${OVERRIDE_DATASET}" ]; then
    INFERENCE_CMD+=(--override-dataset "${OVERRIDE_DATASET}")
fi
if [ -n "${GENE_IDS_FILE}" ]; then
    INFERENCE_CMD+=(--gene-ids-file "${GENE_IDS_FILE}")
fi
if [ -n "${LIMIT}" ]; then
    INFERENCE_CMD+=(--limit "${LIMIT}")
fi
if [ -n "${ALANINE_MAX_SAMPLES}" ]; then
    INFERENCE_CMD+=(--alanine-max-samples "${ALANINE_MAX_SAMPLES}")
fi

"${INFERENCE_CMD[@]}"

if [ ! -f "${PE_SCAN_FILE}" ]; then
    echo "Error: PE scan TSV not created at ${PE_SCAN_FILE}"
    exit 1
fi

if [ ! -f "${ALANINE_SCAN_FILE}" ]; then
    echo "Error: alanine scan TSV not created at ${ALANINE_SCAN_FILE}"
    exit 1
fi

echo ""
echo ">>> Step 2: Running analyze_alanine_scan.py..."
ALANINE_CMD=(
    python3 "${REPO_ROOT}/scripts/training/analyze_alanine_scan.py"
    --alanine-scan "${ALANINE_SCAN_FILE}"
    --checkpoint "${CHECKPOINT}"
    --output-dir "${ALANINE_ANALYSIS_DIR}"
)

if [ -n "${OVERRIDE_DATASET}" ]; then
    ALANINE_CMD+=(--dataset-path "${OVERRIDE_DATASET}")
fi
if [ -n "${PLDDT_PICKLE}" ]; then
    ALANINE_CMD+=(--plddt-pickle "${PLDDT_PICKLE}")
fi

"${ALANINE_CMD[@]}"

echo ""
echo ">>> Step 3: Running analyze_pe_scan_effect.py..."
python3 "${REPO_ROOT}/scripts/training/analyze_pe_scan_effect.py" \
    --checkpoint "${CHECKPOINT}" \
    --pe-scan "${PE_SCAN_FILE}" \
    --output-dir "${OUTPUT_DIR}" \
    --pdf-name "${PDF_NAME}" \
    --tsne-save-path "${TSNE_COORDS}" \
    --overwrite

if [ ! -f "${TSNE_COORDS}" ]; then
    echo "Error: t-SNE coordinates file not created at ${TSNE_COORDS}"
    exit 1
fi

echo ""
echo ">>> Step 4: Running cluster_pe_scan_nodes.py..."
python3 "${REPO_ROOT}/scripts/training/cluster_pe_scan_nodes.py" \
    --checkpoint "${CHECKPOINT}" \
    --node-sensitivity "${NODE_SENSITIVITY}" \
    --output-dir "${OUTPUT_DIR}" \
    --tsne-coords "${TSNE_COORDS}" \
    --min-frequency 1 \
    --dbscan-eps 3.0 \
    --dbscan-min-samples 3

if [ ! -f "${CLUSTER_SUMMARY}" ]; then
    echo "Warning: no clusters found or cluster summary not generated."
    echo "Skipping enrichment and final PE cluster plot."
    exit 0
fi

echo ""
echo ">>> Step 5: Running enrichment_pe_clusters.py..."
python3 "${REPO_ROOT}/scripts/training/enrichment_pe_clusters.py" \
    --cluster-summary "${CLUSTER_SUMMARY}" \
    --output-dir "${OUTPUT_DIR}/enrichment" \
    --min-avg-frequency 1.2 \
    --min-nodes 6 \
    --top-terms "${TOP_TERMS}"

echo ""
echo ">>> Step 6: Running plot_pe_clusters_enrichment_labeled.py..."
python3 "${REPO_ROOT}/scripts/training/plot_pe_clusters_enrichment_labeled.py" \
    --checkpoint "${CHECKPOINT}" \
    --node-sensitivity "${NODE_SENSITIVITY}" \
    --clustered-nodes "${CLUSTERED_NODES}" \
    --cluster-summary "${CLUSTER_SUMMARY}" \
    --enrichment-summary "${ENRICHMENT_SUMMARY}" \
    --output-dir "${OUTPUT_DIR}" \
    --pdf-name "${FINAL_PDF_NAME}" \
    --tsne-coords "${TSNE_COORDS}" \
    --top-n-clusters "${TOP_N_CLUSTERS}"

echo ""
echo ">>> Pipeline complete!"
echo "Outputs are in: ${OUTPUT_DIR}"
echo "Base predictions: ${BASE_PREDICTIONS_FILE}"
echo "Alanine scan TSV: ${ALANINE_SCAN_FILE}"
echo "PE scan TSV: ${PE_SCAN_FILE}"
echo "Alanine analysis: ${ALANINE_ANALYSIS_DIR}"
echo "Final PE plot: ${OUTPUT_DIR}/${FINAL_PDF_NAME}.pdf"
