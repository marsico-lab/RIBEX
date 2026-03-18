import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *
import subprocess
initialize(__file__)

for dataset_name in ["bressin19", "InterPro", "RIC"]:
    print(f"Processing dataset: {dataset_name}")
    tsv_file = DATA_RAW.joinpath(dataset_name + ".tsv")
    fasta_file = DATA_RAW.joinpath(dataset_name + ".fasta")

    df = pd.read_csv(tsv_file, sep='\t')  # Assuming a tab-separated values file

    #print(df.head())
    if not fasta_file.exists():
        with open(fasta_file, 'w') as fasta:
            for index, row in tqdm(df.iterrows(), total=df.shape[0]):   
                if pd.isna(row['sequence']) or row['sequence'].strip() == '':
                    print(f"Skipping {row['Gene_ID']} because sequence is not there.")
                    continue 

                #check if sequence length is between 40 and 2000
                if len(row['sequence']) < 40 or len(row['sequence']) > 6000:
                    print(f"Skipping {row['Gene_ID']} because sequence length is not between 40 and 2000.")
                    continue
                header = f">{row['Gene_ID']}|{row['taxon_ID']}\n"
                sequence = f"{row['sequence']}\n"
                fasta.write(header)
                fasta.write(sequence)
        print(f"Conversion to FASTA format completed for {dataset_name}.")

    # convert fasta file to db using mmseqs2 command line tool
    db_file = DATA_CLUST.joinpath(dataset_name, dataset_name + "_DB")
    if not db_file.parent.exists():
        db_file.parent.mkdir(parents=True, exist_ok=True)
    
    # run mmseqs createdb
    if fasta_file.exists() and not db_file.exists():
        print(f"Creating db for {dataset_name} using mmseqs2.")
        subprocess.run(["mmseqs", "createdb", fasta_file, db_file])
        print(f"DB creation completed for {dataset_name}.")

    clust_file = DATA_CLUST.joinpath(dataset_name, dataset_name + "_clust")
    if not clust_file.parent.exists():
        clust_file.parent.mkdir(parents=True, exist_ok=True)
    clust_file_db = Path(str(clust_file) + ".dbtype")

    # run mmseqs cluster
    if db_file.exists() and not clust_file_db.exists():
        print(f"Creating clust for {dataset_name} using mmseqs2.")
        subprocess.run(["mmseqs", "cluster", db_file, clust_file, "/tmp", "-c", "0.5", "--min-seq-id", "0.2", "--cov-mode", "0"])
        print(f"Clust creation completed for {dataset_name}.")

    tsv_clust_file = DATA_CLUST.joinpath(dataset_name, dataset_name + "_clust.tsv")
    if not tsv_clust_file.parent.exists():
        tsv_clust_file.parent.mkdir(parents=True, exist_ok=True)

    # run mmseqs createtsv
    if clust_file_db.exists() and not tsv_clust_file.exists():
        print(f"Creating tsv for {dataset_name} using mmseqs2.")
        subprocess.run(["mmseqs", "createtsv", db_file, db_file, clust_file, tsv_clust_file])
        print(f"tsv creation completed for {dataset_name}.")        

    # add the cluster number to the original tsv file
    if tsv_clust_file.exists() and tsv_file.exists():
        df_clust = pd.read_csv(tsv_clust_file, sep='\t', header=None, names=["cluster", "protein_names"])
        # for each repeated cluster member in cluster column of tsv_clust_file, add the cluster number to the original tsv file
        # create a lookup dictionary that connects protein names to cluster numbers
        cluster_lookup = {}
        for cluster_index, cluster_hub in enumerate(df_clust['cluster'].unique()):
            protein_names = df_clust[df_clust['cluster'] == cluster_hub]['protein_names'].values
            for protein_name in protein_names:
                protein_name = protein_name.split("|")[0]
                cluster_lookup[protein_name] = cluster_index
        # add the cluster number to the original tsv file as integers
        df['cluster_number'] = df['Gene_ID'].map(cluster_lookup)
        # overwrite the original tsv file with the new column
        print(f"Adding cluster numbers to {dataset_name} tsv file.")
        df.to_csv(DATA_RAW.joinpath(dataset_name + ".tsv"), sep='\t', index=False)
        
        

        
# # %%
# df.dropna(subset=['cluster_number'], inplace=True)

# # %%
# df.reset_index(inplace=True, drop=True)

# # %%
# df_dict = df.groupby('cluster_number').apply(lambda x: x.Gene_ID.tolist()).to_dict()
# # %%
