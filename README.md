# RIBEX / RBP_IG

RIBEX combines protein language model embeddings with graph-derived positional encodings from the human STRING protein-protein interaction network for RNA-binding protein prediction. The repository contains the raw-data builders, embedding generation, dataset assembly, FiLM-PE training, LoRA fine-tuning, and the explainability pipeline used for PE-scan clustering and enrichment.

<img src="./docs/images/ribex_architecture.svg" alt="RIBEX architecture" width="800" />

## Environment

Create the shared conda environment on Lustre:

```bash
conda env create -f environment.yaml
conda activate rbp_ig_lustre
```

Set the storage root used by the scripts. The code defaults to `/path/to/RBP_IG_storage`, but setting it explicitly is safer:

```bash
export REPOSITORY=/path/to/RBP_IG_storage
```

Several legacy HPC launchers also contain explicit placeholders such as `/path/to/RBP_IG`, `/path/to/RBP_IG_storage`, and `/path/to/miniconda3/bin/activate`. Replace those with your local checkout path, storage path, and conda installation before submitting jobs.

Optional if you do not want W&B runs uploaded:

```bash
export WANDB_MODE=offline
```

## Storage layout

The scripts read and write under `${REPOSITORY}/data`:

```text
${REPOSITORY}/data/
├── data_original/
│   ├── bressin19/
│   ├── InterPro/
│   └── RIC/
├── data_raw/
├── data_sets/
├── embeddings/
├── figures/
├── logs/
├── models/
└── splits/
```

The Git checkout itself is used for code, helper scripts, random-search launchers, and local run folders such as LoRA trial directories.

## Generating the data

The command inventory is in [pipeline.sh](pipeline.sh). The effective order is:

1. Put the original source files into `${REPOSITORY}/data/data_original/bressin19`, `${REPOSITORY}/data/data_original/InterPro`, and `${REPOSITORY}/data/data_original/RIC`.
2. Build the harmonised raw tables:

```bash
python3 scripts/data_raw/generate_Bressin19.py
python3 scripts/data_raw/generate_InterPro.py
python3 scripts/data_raw/generate_RIC.py
python3 scripts/data_raw/analyze.py
```

3. Run the sequence clustering step before dataset generation. This appends `cluster_number` to the raw TSVs and writes the MMseqs2 clustering files used later for leakage-aware splits:

```bash
python3 scripts/data_raw/cluster_tsv_data.py
```

4. Generate embeddings. The full set of model-specific commands is in [pipeline.sh](pipeline.sh); a common example is:

```bash
python3 scripts/embeddings/generate.py --device cuda:0 --languageModel esm2_t33_650M_UR50D --precision f16 --maxSeqLen 2000
```

5. Build the downstream datasets:

```bash
python3 scripts/data_sets/generate.py
python3 scripts/data_sets/analyze.py
```

This creates files such as `${REPOSITORY}/data/data_sets/RIC_human_fine-tuning.pkl` and `${REPOSITORY}/data/data_sets/bressin19_human_fine-tuning.pkl`.

## STRING PPI network and positional encodings

RIBEX expects the precomputed positional-encoding assets:

```text
${REPOSITORY}/data/data_sets/ranks_personalized_page_rank_0.5_v12_all.npy
${REPOSITORY}/data/data_sets/gene_names_0.5_v12_all.npy
```

If you need to regenerate them from STRING:

1. Confirm the latest official STRING release on the version-history page: `https://string-db.org/cgi/access`. At the time of writing, the current STRING release is `12.0`.
2. Go to the official download page: `https://string-db.org/cgi/download.pl`.
3. Restrict the download to `Homo sapiens` / taxon `9606` and download the filtered v12 full interaction table named `9606.protein.links.full.v12.0.txt.gz`.
4. Place that file at `${REPOSITORY}/data/data_original/string_db/9606.protein.links.full.v12.0.txt.gz`.
5. Generate the global PPI positional-encoding assets with:

```bash
mkdir -p ${REPOSITORY}/data/data_original/string_db
python3 scripts/data_sets/positional_encoding.py \
  --string-links ${REPOSITORY}/data/data_original/string_db/9606.protein.links.full.v12.0.txt.gz
```

6. This writes:

```text
${REPOSITORY}/data/data_sets/ranks_personalized_page_rank_0.5_v12_all.npy
${REPOSITORY}/data/data_sets/gene_names_0.5_v12_all.npy
```

At training and inference time, gene IDs are mapped to STRING IDs through the STRING `get_string_ids` API in [positional_encoding_processing.py](scripts/data_sets/positional_encoding_processing.py).

## Fine-tuning workflows

### Shared splits

Generate fair train/held-out splits that are consistent across FiLM PE and LoRA:

```bash
bash scripts/data_util/generate_shared_splits_any.sh RIC 2023
```

This writes split files under `${REPOSITORY}/data/splits/`.

### LoRA random search with nested holdout evaluation

```bash
bash scripts/training/run_scripts/run_LoRA_fine_tuning_random_search.sh
```

Useful overrides:

```bash
SEED=2024 LM_NAME=protT5_xl_uniref50 NUM_TRIALS=30 bash scripts/training/run_scripts/run_LoRA_fine_tuning_random_search.sh
```

### FiLM PE random search with nested holdout evaluation

```bash
bash scripts/training/run_scripts/run_FiLM_PE_fine_tuning_random_search.sh
```

Useful overrides:

```bash
SEED=2024 LM_NAME=esm2_t36_3B_UR50D NUM_TRIALS=30 bash scripts/training/run_scripts/run_FiLM_PE_fine_tuning_random_search.sh
```

### Nested random-search protocol

Both launchers now enforce the same protocol on the held-out split:

1. Train each trial on the shared training split.
2. Use the saved held-out predictions per epoch.
3. Split that held-out set into:
   - 1/3 nested validation for best-epoch selection
   - 2/3 nested test for reporting the hyperparameter combination
4. Rank hyperparameter combinations by nested-validation AUPRC only.
5. Report the corresponding nested-test metrics separately.

The post-search evaluator is [evaluate_random_search_nested_holdout.py](scripts/training/evaluate_random_search_nested_holdout.py).

Each search writes:

```text
results/random_search/<search_tag>/
├── manifest.tsv
├── nested_validation_split.tsv
├── nested_test_split.tsv
├── random_search_per_epoch.tsv
├── random_search_leaderboard.tsv
└── best_trial.json
```

If you want a single final model after the search, rerun `scripts/training/train.py` once with the selected hyperparameters from `best_trial.json`.

## Explainability and PE-scan clustering

The reproducible PE-scan clustering workflow is:

```bash
bash repro_pe_scan_pipeline.sh
```

That runs:

1. [analyze_pe_scan_effect.py](scripts/training/analyze_pe_scan_effect.py)
2. [cluster_pe_scan_nodes.py](scripts/training/cluster_pe_scan_nodes.py)
3. [enrichment_pe_clusters.py](scripts/training/enrichment_pe_clusters.py)
4. [plot_pe_clusters_enrichment_labeled.py](scripts/training/plot_pe_clusters_enrichment_labeled.py)

## Notes

- [pipeline.sh](pipeline.sh) is the quick reference for the main commands.
- The repository has many historical run folders; the new random-search launchers isolate trials by `run_tag` so evaluation only picks up the intended batch.
- The raw-data builders use online InterPro / MobiDB / STRING services, so network access is required when regenerating those assets from scratch.
