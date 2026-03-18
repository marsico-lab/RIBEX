import argparse
from dataclasses import dataclass
import logging
import sys
from pathlib import Path
from typing import Dict, Optional
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *  # noqa: F401, E402
import pickle
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from functools import lru_cache

initialize(__file__)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

RANKS_FILE = DATA_SETS.joinpath("ranks_personalized_page_rank_0.5_v12_all.npy")
GENE_NAMES_FILE = DATA_SETS.joinpath("gene_names_0.5_v12_all.npy")

def load_ranks_and_genes(ranks_file=RANKS_FILE, gene_names_file=GENE_NAMES_FILE):
    ranks = np.load(ranks_file, mmap_mode="r")
    ranks = np.asarray(ranks).copy(order="C")
    d = ranks.diagonal().copy()
    np.fill_diagonal(ranks, np.maximum(0.0, d - 1.0))
    gene_names = np.load(gene_names_file, allow_pickle=True).astype(str).tolist()
    return ranks, gene_names

def map_genes_to_string_ids(gene_list):
    # Define the STRING API endpoint for mapping
    string_api_url = "https://string-db.org/api/json/get_string_ids"

    # Prepare your parameters
    params = {
        'identifiers': '\r'.join(gene_list),  # Join gene names with carriage return
        'species': 9606,  # Taxonomy ID for human
        #'caller_identity': 'your_email@example.com'  # Replace with your email or identifier
    }

    # Send a request to the STRING API
    response = requests.post(string_api_url, data=params)

    # Check if the request was successful
    if response.status_code == 200:
        mappings = response.json()
        return {mapping['queryItem']: mapping['stringId'] for mapping in mappings}
    else:
        print("Error:", response.status_code, response.text)
        
def load_datafile_and_map_genes(datafile, ranks_file=RANKS_FILE, gene_names_file=GENE_NAMES_FILE): 
    
    with open(datafile, 'rb') as f:
        df = pickle.load(f)
    
    gene_names = df['Gene_ID'].values.tolist()
    # Map gene names to STRING IDs
    gene_id_map = map_genes_to_string_ids(gene_names)

    print(f"Percentage of unmapped genes: {len([gene for gene in gene_names if gene not in gene_id_map]) / len(gene_names) * 100:.2f}%")
    
    ranks, string_ids = load_ranks_and_genes(ranks_file, gene_names_file)
    print(f"Percentage of mapped genes present in the PPI network: {len([gene for gene in gene_id_map.values() if gene in string_ids])/len(gene_id_map)*100}%")
    # map string IDs to their indices in the ranks matrix
    string_ids_to_ranks_indices = {string_ids: index for index, string_ids in enumerate(string_ids)}
        
    return ranks, gene_id_map, string_ids_to_ranks_indices

# --- Fix field order + types so dataclass is valid and lightweight ----------
@dataclass
class PositionalEncodingData:
    gene_names_file: Path
    genes_ids_to_string_ids: Dict[str, str]
    string_ids_to_ranks_indices: Dict[str, int]
    ranks: Optional[np.ndarray] = None
    ranks_reduced: Optional[np.ndarray] = None
    pca_mean: Optional[np.ndarray] = None
    pca_components: Optional[np.ndarray] = None
    pca_n_components: Optional[int] = None
    scaler_mean: Optional[np.ndarray] = None
    scaler_scale: Optional[np.ndarray] = None
    

def create_positional_encoding_data(datafile, ranks_file=RANKS_FILE, gene_names_file=GENE_NAMES_FILE, pca_n_components: Optional[int] = None) -> PositionalEncodingData:
    ranks, gene_id_map, string_ids_to_ranks_indices = load_datafile_and_map_genes(datafile, ranks_file, gene_names_file)

    scaler = StandardScaler()
    ranks = scaler.fit_transform(ranks)
    scaler_mean = scaler.mean_
    scaler_scale = scaler.scale_
    pca = PCA(n_components=pca_n_components)
    ranks_reduced = pca.fit_transform(ranks)
    # Save the PCA components and mean
    pca_mean = pca.mean_
    pca_components = pca.components_

    return PositionalEncodingData(
        gene_names_file=gene_names_file,
        genes_ids_to_string_ids=gene_id_map,
        string_ids_to_ranks_indices=string_ids_to_ranks_indices,
        ranks=ranks,
        ranks_reduced=ranks_reduced,
        pca_mean=pca_mean,
        pca_components=pca_components,
        pca_n_components=pca_n_components,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
    )

@lru_cache(maxsize=8)
def get_posenc_pkg(datafile: str, pca_n_components: Optional[int]) -> PositionalEncodingData:
    """LRU-cached creator; args must be hashable (use strings/ints)."""
    return create_positional_encoding_data(
        datafile=Path(datafile),
        ranks_file=RANKS_FILE,
        gene_names_file=GENE_NAMES_FILE,
        pca_n_components=pca_n_components,
    )

def build_pe_matrix_for_dataset(dataset, pkg: PositionalEncodingData, use_pca: bool = True) -> np.ndarray:
    """Align PE table to ds.dataSet_df['Gene_ID'] order with mean fallback for unmapped genes."""
    table = pkg.ranks_reduced if (use_pca and pkg.ranks_reduced is not None) else pkg.ranks
    assert table is not None, "PositionalEncodingData has no ranks table."
    genes = dataset.dataSet_df["Gene_ID"].astype(str).values
    idxs = np.array([pkg.string_ids_to_ranks_indices.get(pkg.genes_ids_to_string_ids.get(g, ""), -1) for g in genes], dtype=int)
    valid = idxs >= 0

    X = np.empty((len(genes), table.shape[1]), dtype=np.float32)
    mean_vec = (table[idxs[valid]].mean(axis=0) if valid.any() else table.mean(axis=0)).astype(np.float32)
    X[:] = mean_vec
    if valid.any():
        X[valid] = table[idxs[valid]].astype(np.float32)
    return X





