import argparse
import pickle
from pathlib import Path
import sys
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.initialize import *

def main():
    parser = argparse.ArgumentParser(description="Create a dataset containing only genes present in all specified LM embeddings.")
    parser.add_argument(
        "--lms",
        nargs="+",
        required=True,
        help="List of Language Model names (e.g., protT5_xl_uniref50 esm2_t33_650M_UR50D)"
    )
    parser.add_argument(
        "--emb",
        required=True,
        help="Embedding subfolder name (e.g., bressin19)"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Input dataset filename (e.g., bressin19_human_fine-tuning.pkl)"
    )
    
    args = parser.parse_args()
    
    # Initialize paths
    initialize(__file__)
    
    dataset_path = DATA_SETS.joinpath(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'rb') as f:
        df = pickle.load(f)
    
    initial_count = len(df)
    print(f"Initial dataset size: {initial_count}")
    
    # Find intersection of genes
    common_genes = None
    
    for lm_name in args.lms:
        emb_path = EMBEDDINGS.joinpath(lm_name).joinpath(args.emb)
        if not emb_path.exists():
            raise FileNotFoundError(f"Embedding folder not found: {emb_path}")
            
        print(f"Scanning embeddings for {lm_name} in {emb_path}...")
        # Get all filenames in the directory (assuming filenames are Gene_IDs)
        # Note: Embeddings are typically saved as torch files, but we just need the stem (Gene_ID)
        # However, checking file existence for every gene in the DF might be slow if we iterate DF.
        # Faster to listdir.
        available_genes = set(p.name for p in emb_path.iterdir())
        
        print(f"  Found {len(available_genes)} embeddings.")
        
        if common_genes is None:
            common_genes = available_genes
        else:
            common_genes = common_genes.intersection(available_genes)
            
        print(f"  Intersection size so far: {len(common_genes)}")
        
    # Filter dataset
    print(f"Filtering dataset to keep only {len(common_genes)} common genes...")
    
    # Filter rows where Gene_ID is in common_genes
    # Note: dataset.py says: existing = set([ p.name for p in self.embeddingFolder.iterdir()])
    # So p.name includes extension? Usually embeddings don't have extension in this repo based on dataset.py line 50:
    # embeddingPath = self.embeddingFolder.joinpath(d['Gene_ID'])
    # So Gene_ID should match filename exactly.
    
    df_filtered = df[df['Gene_ID'].isin(common_genes)].reset_index(drop=True)
    
    final_count = len(df_filtered)
    print(f"Final dataset size: {final_count}")
    print(f"Removed {initial_count - final_count} entries ({100 * (initial_count - final_count) / initial_count:.2f}%)")
    
    # Save new dataset
    new_filename = f"{dataset_path.stem}_intersection{dataset_path.suffix}"
    output_path = DATA_SETS.joinpath(new_filename)
    
    print(f"Saving new dataset to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(df_filtered, f)
        
    print("Done!")

if __name__ == "__main__":
    main()
