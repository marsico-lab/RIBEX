

## 1. **data_raw**: get protein data from different databases (files, web API, etc:9) and transoforms them into a semi-standard form
python3 scripts/data_raw/generate_Bressin19.py
    #IN: files in bressin19/ 
    #OUT: bressin19.tsv
    #LOG: logs/data_raw/generate_Bressin19.py
python3 scripts/data_raw/generate_InterPro.py
    #IN: files in InterPro/
    #OUT: InterPro.tsv
    #LOG: logs/data_raw/generate_InterPro.py
python3 scripts/data_raw/generate_RIC.py
    #IN: files in RIC/
    #OUT: RIC.tsv
    #LOG: logs/data_raw/generate_RIC.py
python3 scripts/data_raw/analyze.py
    #IN: bressin19.tsv, InterPro.tsv, RIC.tsv
    #OUT: [LOGS]
    #LOG: $LOGS/data_raw/analyze.py
python3 scripts/data_raw/cluster_tsv_data.py
    #IN: bressin19.tsv, InterPro.tsv, RIC.tsv
    #OUT: cluster_number column appended to each TSV and MMseqs2 cluster files in $DATA_RAW/clust/
    #LOG: stdout/stderr from mmseqs2 and script execution


## 2. **embedding**: takes the protein sequences  and creates embedings
python3 scripts/embeddings/generate.py --device cuda:0 --languageModel esm1b_t33_650M_UR50S --precision f32
python3 scripts/embeddings/generate.py --device cuda:1 --languageModel esm2_t33_650M_UR50D --precision f16 --maxSeqLen 2000 #70GB
#python3 scripts/embeddings/generate.py --device cuda --languageModel esm2_t33_650M_UR50D --precision f16
python3 scripts/embeddings/generate.py --device cuda:2 --languageModel esm2_t36_3B_UR50D --precision f16 --maxSeqLen 1600 # 64GB
#python3 scripts/embeddings/generate.py --device cuda:2 --languageModel esm2_t36_3B_UR50D --precision f16
python3 scripts/embeddings/generate.py --device cuda --languageModel esm2_t48_15B_UR50D --precision f16  --maxSeqLen 850 #70GB
#python3 scripts/embeddings/generate.py --device cuda --languageModel esm2_t48_15B_UR50D --precision f16
    #IN: bressin19.tsv, InterPro.tsv, RIC.tsv
    #OUT: files in $EMBEDDINGS/<LM_name>/<datsetName>/
    #LOG: $LOGS/embeddings/generate.py


## 3. **data_sets"'**: Agggates embeddings and all other relevantformation into different datasest that can be used for training.
#   Datasets may overlap and are created/filtered on different cirteria (e.g. a dataset for pre-training and one for fine-tuning)
python3 scripts/data_sets/generate.py
    #IN: bressin19.tsv, InterPro.tsv, RIC.tsv, [EMBEDDINGS in the respective embedding folder]
    #OUT: datasts in $DATA_SETS\<dataset-name>.pkl
    #LOG: $LOGS/data_sets/generate.py
python3 scripts/data_sets/analyze.py
    #IN: datasts in $DATA_SETS\<dataset-name>.pkl
    #OUT: [LOGS] and figures in $FIGURES/data_sets
    #LOG: $LOGS/data_sets/generate.py
mkdir -p ${REPOSITORY}/data/data_original/string_db
    #STRING current version page: https://string-db.org/cgi/access
    #STRING download page: https://string-db.org/cgi/download.pl
    #STRING API docs: https://string-db.org/help/api/
    #For Homo sapiens on STRING v12, download the filtered file named:
    #9606.protein.links.full.v12.0.txt.gz
python3 scripts/data_sets/positional_encoding.py --string-links ${REPOSITORY}/data/data_original/string_db/9606.protein.links.full.v12.0.txt.gz
    #IN: STRING human PPI network (filtered download from STRING v12)
    #OUT: $DATA_SETS/ranks_personalized_page_rank_0.5_v12_all.npy and $DATA_SETS/gene_names_0.5_v12_all.npy
    #LOG: $LOGS/data_sets/positional_encoding.py


## 4. **training**: traines models based on datasets

## Bressin

#Random model (sklearn)
python3 scripts/training/train.py -D 2 -M Random -DS bressin19_human_pre-training.pkl -lm esm1b_t33_650M_UR50S -ef "bressin19" -S 2023

#Linear model (sklearn)
python3 scripts/training/train.py -D 2 -M Linear -DS bressin19_human_pre-training.pkl -lm esm1b_t33_650M_UR50S -ef "bressin19" -S 2023
python3 scripts/training/train.py -D 2 -M Linear -DS bressin19_human_pre-training.pkl -lm esm2_t33_650M_UR50D -ef "bressin19" -S 2023
python3 scripts/training/train.py -D 2 -M Linear -DS bressin19_human_pre-training.pkl -lm esm2_t36_3B_UR50D -ef "bressin19" -S 2023
python3 scripts/training/train.py -D 2 -M Linear -DS bressin19_human_pre-training.pkl -lm esm2_t48_15B_UR50D -ef "bressin19" -S 2023

#Random Forest (sklearn)
python3 scripts/training/train.py -D 2 -M RandomForest -DS bressin19_human_pre-training.pkl -lm esm1b_t33_650M_UR50S -ef "bressin19" -S 2023
python3 scripts/training/train.py -D 2 -M RandomForest -DS bressin19_human_pre-training.pkl -lm esm2_t33_650M_UR50D -ef "bressin19" -S 2023
python3 scripts/training/train.py -D 2 -M RandomForest -DS bressin19_human_pre-training.pkl -lm esm2_t36_3B_UR50D -ef "bressin19" -S 2023
python3 scripts/training/train.py -D 2 -M RandomForest -DS bressin19_human_pre-training.pkl -lm esm2_t48_15B_UR50D -ef "bressin19" -S 2023

#Gradient Boosting
python3 scripts/training/train.py -D 2 -M XGBoost -DS bressin19_human_pre-training.pkl -lm esm1b_t33_650M_UR50S -ef "bressin19" -S 2023 -lr 0.5
python3 scripts/training/train.py -D 2 -M XGBoost -DS bressin19_human_pre-training.pkl -lm esm2_t33_650M_UR50D -ef "bressin19" -S 2023 -lr 0.5
python3 scripts/training/train.py -D 2 -M XGBoost -DS bressin19_human_pre-training.pkl -lm esm2_t36_3B_UR50D -ef "bressin19" -S 2023 -lr 0.5
python3 scripts/training/train.py -D 2 -M XGBoost -DS bressin19_human_pre-training.pkl -lm esm2_t48_15B_UR50D -ef "bressin19" -S 2023 -lr 0.5

#Peng
python3 scripts/training/train.py -D 1 -M Peng -DS bressin19_human_pre-training.pkl -lm esm1b_t33_650M_UR50S -ef "bressin19" -S 2023 -e 30 -bs 1024 --patience 5 -lr 0.0005
python3 scripts/training/train.py -D 1 -M Peng -DS bressin19_human_pre-training.pkl -lm esm1b_t33_650M_UR50S -ef "bressin19" -S 2023 -e 50 -bs 1024 --patience 20 -lr 0.0005 --checkpoint-folder "Peng_ESM1b_650M-E=bressin19-S=2023-E=30-BS=1024-LR=0.000500"


## RIC

#Linear model (sklearn)
python3 scripts/training/train.py -D 2 -M Linear -DS RIC_human_pre-training.pkl -lm esm1b_t33_650M_UR50S -ef "RIC" -S 2023

#Peng
python3 scripts/training/train.py -D 1 -M Peng -DS RIC_human_pre-training.pkl -lm esm1b_t33_650M_UR50S -ef "RIC" -S 2023 -e 30 -bs 1024 --patience 10

#Shared splits for fair model comparison (generate once per dataset)
bash scripts/data_util/generate_shared_splits_any.sh RIC 2023
bash scripts/data_util/generate_shared_splits_any.sh bressin19 2023

#Peng fine-tuning on shared splits (RIC)
python3 scripts/training/train.py -D 1 -M Peng -DS RIC_human_fine-tuning.pkl -lm esm2_t33_650M_UR50D -ef "RIC" -S 2023 -e 50 -bs 256 --patience 5 -lr 0.005
python3 scripts/training/train.py -D 1 -M Peng -DS RIC_human_fine-tuning.pkl -lm protT5_xl_uniref50 -ef "RIC" -S 2023 -e 50 -bs 256 --patience 5 -lr 0.005

#Peng fine-tuning on shared splits (bressin19)
python3 scripts/training/train.py -D 1 -M Peng -DS bressin19_human_fine-tuning.pkl -lm esm2_t33_650M_UR50D -ef "bressin19" -S 2023 -e 100 -bs 512 --patience 20 -lr 0.0005
python3 scripts/training/train.py -D 1 -M Peng -DS bressin19_human_fine-tuning.pkl -lm protT5_xl_uniref50 -ef "bressin19" -S 2023 -e 100 -bs 512 --patience 20 -lr 0.0005

#Nested-holdout random search for fine-tuning
bash scripts/training/run_scripts/run_LoRA_fine_tuning_random_search.sh
bash scripts/training/run_scripts/run_FiLM_PE_fine_tuning_random_search.sh



#legacy:
python3 scripts/training/train.py -D 1 -M Peng -DS bressin19_human_pre-training.pkl -lm esm1b_t33_650M_UR50S -ef "bressin19" -S 2023 -e 30 -bs 1024 --patience 5
python3 scripts/training/train.py -D 1 -M Peng -DS bressin19_human_fine-tuning.pkl -lm esm1b_t33_650M_UR50S -ef "bressin19" --checkpoint-folder "LM=esm1b_t33_650M_UR50S-E=bressin19-S=2023-E=100-BS=1024" -S 2024 -e 50 -bs 1024 --patience 20
python3 scripts/training/train.py -D 1 -M Peng -DS InterPro_human_pre-training.pkl -lm esm1b_t33_650M_UR50S -ef "InterPro" -S 2024 -e 20 -bs 1024 --patience 5
python3 scripts/training/train.py -D 1 -M Peng -DS InterPro_human_fine-tuning.pkl -lm esm1b_t33_650M_UR50S -ef "InterPro" --checkpoint-folder "LM=esm1b_t33_650M_UR50S-E=InterPro-S=2023-E=30-BS=1024" -S 2023 -e 10 -bs 1024 --patience 3
python3 scripts/training/train.py -D 1 -M Peng -DS RIC_human_pre-training.pkl -lm esm1b_t33_650M_UR50S -ef "RIC" -S 2023 -e 50 -bs 1024 --patience 10
python3 scripts/training/train.py -D 1 -M Peng -DS RIC_human_fine-tuning.pkl -lm esm1b_t33_650M_UR50S -ef "RIC" --checkpoint-folder "LM=esm1b_t33_650M_UR50S-E=RIC-S=2023-E=50-BS=1024" -S 2023 -e 25 -bs 1024 --patience 5
