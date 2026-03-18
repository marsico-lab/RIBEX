#!/usr/bin/env python3
"""
Generate global STRING-based positional-encoding assets for RIBEX.

This script consumes the human STRING interaction table
`9606.protein.links.full.v12.0.txt.gz` and produces the two assets used by
`positional_encoding_processing.py`:

- `${REPOSITORY}/data/data_sets/ranks_personalized_page_rank_0.5_v12_all.npy`
- `${REPOSITORY}/data/data_sets/gene_names_0.5_v12_all.npy`

Official STRING references:
- Version history: https://string-db.org/cgi/access
- Downloads: https://string-db.org/cgi/download.pl
- API docs: https://string-db.org/help/api/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch_scatter
import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import spmm

sys.path.append(str(Path(".").absolute()))
from scripts.initialize import DATA_ORIGINAL, DATA_SETS, initialize  # noqa: E402


initialize(__file__)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


STRING_VERSION_URL = "https://string-db.org/cgi/access"
STRING_DOWNLOAD_URL = "https://string-db.org/cgi/download.pl"
STRING_API_DOCS_URL = "https://string-db.org/help/api/"
DEFAULT_STRING_FILE = DATA_ORIGINAL / "string_db" / "9606.protein.links.full.v12.0.txt.gz"
DEFAULT_RANKS_FILE = DATA_SETS / "ranks_personalized_page_rank_0.5_v12_all.npy"
DEFAULT_GENES_FILE = DATA_SETS / "gene_names_0.5_v12_all.npy"
DEFAULT_SCORE_COLUMNS = (
    "neighborhood",
    "neighborhood_transferred",
    "fusion",
    "cooccurence",
    "coexpression",
    "coexpression_transferred",
    "experiments",
    "experiments_transferred",
    "database",
    "database_transferred",
    "textmining",
    "textmining_transferred",
)
DEFAULT_WEIGHTS = (1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate STRING-based global Personalized PageRank assets for RIBEX."
    )
    parser.add_argument(
        "--string-links",
        type=Path,
        default=DEFAULT_STRING_FILE,
        help=(
            "Human STRING interaction table. Expected latest filename for STRING v12: "
            "9606.protein.links.full.v12.0.txt.gz"
        ),
    )
    parser.add_argument(
        "--output-ranks",
        type=Path,
        default=DEFAULT_RANKS_FILE,
        help="Output .npy file for the Personalized PageRank matrix.",
    )
    parser.add_argument(
        "--output-genes",
        type=Path,
        default=DEFAULT_GENES_FILE,
        help="Output .npy file for STRING node identifiers in matrix order.",
    )
    parser.add_argument(
        "--connection-threshold",
        type=float,
        default=0.5,
        help="Threshold on recomputed STRING combined score after prior handling (default: 0.5).",
    )
    parser.add_argument(
        "--damping-factor",
        type=float,
        default=0.85,
        help="Personalized PageRank damping factor (default: 0.85).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10000,
        help="Maximum PageRank iterations per seed node (default: 10000).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-8,
        help="Convergence tolerance for PageRank (default: 1e-8).",
    )
    parser.add_argument(
        "--range-start",
        type=int,
        default=0,
        help="Start index in STRING node order for partial generation (default: 0).",
    )
    parser.add_argument(
        "--range-end",
        type=int,
        help="End index (exclusive) in STRING node order for partial generation. Defaults to all nodes.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for PageRank computation (default: auto).",
    )
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def page_rank(
    data: Data,
    teleport_probs: torch.Tensor,
    damping_factor: float,
    max_iterations: int,
    tolerance: float,
) -> torch.Tensor:
    device = data.edge_index.device
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    edge_weight = data.edge_weight

    row_sums = torch_scatter.scatter(edge_weight, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")
    edge_weight = edge_weight / row_sums[edge_index[1]]
    dangling_nodes = torch.where(row_sums == 0)[0]

    normalized_row_sums = torch_scatter.scatter(edge_weight, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")
    normalized_row_sums[dangling_nodes] = 1
    if not torch.allclose(normalized_row_sums, torch.ones(num_nodes, device=device), atol=1e-3):
        raise RuntimeError("Edge weights did not normalize correctly for PageRank.")

    adj_matrix = torch.sparse_coo_tensor(edge_index, edge_weight, torch.Size([num_nodes, num_nodes]), device=device)
    adj_matrix = adj_matrix.to_sparse_csr()

    ranks = torch.full((num_nodes,), 1.0 / num_nodes, device=device)

    for iteration in range(max_iterations):
        prev_ranks = ranks.clone()
        dangling_mass = torch.sum(prev_ranks[dangling_nodes]) * damping_factor / max(int(dangling_nodes.numel()), 1)
        ranks = (
            (1.0 - damping_factor) * teleport_probs
            + damping_factor * spmm(adj_matrix, ranks.unsqueeze(-1)).squeeze(-1)
            + dangling_mass
        )
        if torch.allclose(prev_ranks, ranks, atol=tolerance):
            logger.info("PageRank converged after %d iterations.", iteration)
            break
    else:
        logger.warning("PageRank did not converge within %d iterations.", max_iterations)

    return ranks


def aggregate_scores(
    df_ppi: pd.DataFrame,
    weights: Sequence[int],
    score_columns: Sequence[str] = DEFAULT_SCORE_COLUMNS,
) -> pd.DataFrame:
    if len(weights) != len(score_columns):
        raise ValueError("weights and score_columns must have the same length.")

    prior = 0.041
    df_ppi = df_ppi.copy()
    df_ppi["homology"] = df_ppi["homology"] / 1000.0

    for column, weight in zip(score_columns, weights):
        df_ppi[column] = df_ppi[column] / 1000.0
        no_prior_col = f"{column}_no_prior"
        df_ppi[no_prior_col] = (df_ppi[column] - prior) / (1.0 - prior)
        df_ppi.loc[df_ppi[no_prior_col] < 0.0, no_prior_col] = 0.0
        if column in {"cooccurence", "textmining", "textmining_transferred"}:
            df_ppi[no_prior_col] = df_ppi[no_prior_col] * (1.0 - df_ppi["homology"])
        df_ppi[no_prior_col] = df_ppi[no_prior_col] * weight

    combined = np.ones(len(df_ppi), dtype=np.float64)
    for column in score_columns:
        combined *= (1.0 - df_ppi[f"{column}_no_prior"].to_numpy(dtype=np.float64))
    df_ppi["combined_score_no_prior"] = 1.0 - combined
    df_ppi["combined_score_recomputed"] = df_ppi["combined_score_no_prior"] * (1.0 - prior) + prior
    return df_ppi


def load_string_links(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "STRING interaction table not found at "
            f"{path}. Download STRING v12 human links from {STRING_DOWNLOAD_URL} "
            f"(expected filtered filename: 9606.protein.links.full.v12.0.txt.gz)."
        )

    logger.info("Reading STRING links from %s", path)
    df = pd.read_csv(path, sep=r"\s+")

    required_columns = {"protein1", "protein2", "combined_score", "homology", *DEFAULT_SCORE_COLUMNS}
    missing = required_columns.difference(df.columns)
    if missing:
        raise KeyError(f"{path} is missing required columns: {sorted(missing)}")

    logger.info("Loaded %d STRING interaction rows.", len(df))
    return df


def build_graph(
    df_ppi: pd.DataFrame,
    connection_threshold: float,
    weights: Sequence[int],
    device: torch.device,
) -> Tuple[Data, np.ndarray]:
    logger.info("Aggregating STRING evidence scores.")
    df_ppi = aggregate_scores(df_ppi, weights=weights)
    logger.info("Filtering interactions at combined_score_recomputed > %.3f", connection_threshold)
    df_ppi = df_ppi.loc[df_ppi["combined_score_recomputed"] > connection_threshold].reset_index(drop=True)
    if df_ppi.empty:
        raise RuntimeError("No STRING interactions remain after thresholding.")

    logger.info("Retained %d STRING interaction rows after filtering.", len(df_ppi))

    node_names = pd.Index(df_ppi["protein1"].astype(str)).union(pd.Index(df_ppi["protein2"].astype(str))).to_numpy()
    node_to_index = {name: idx for idx, name in enumerate(node_names)}

    edge_index_0: List[int] = []
    edge_index_1: List[int] = []
    edge_weights: List[float] = []

    for protein1, protein2, score in tqdm.tqdm(
        zip(
            df_ppi["protein1"].astype(str).to_numpy(),
            df_ppi["protein2"].astype(str).to_numpy(),
            df_ppi["combined_score_recomputed"].to_numpy(dtype=np.float32),
        ),
        total=len(df_ppi),
        desc="Building graph",
    ):
        edge_index_0.append(node_to_index[protein1])
        edge_index_1.append(node_to_index[protein2])
        edge_weights.append(float(score))

    if not edge_index_0:
        raise RuntimeError("Graph construction produced no usable edges.")

    data = Data()
    data.edge_index = torch.tensor([edge_index_0, edge_index_1], dtype=torch.long, device=device)
    data.edge_weight = torch.tensor(edge_weights, dtype=torch.float32, device=device)
    data.num_nodes = len(node_names)
    data.node_names = node_names.tolist()

    logger.info("Constructed graph with %d nodes and %d directed edges.", data.num_nodes, len(edge_weights))
    return data, node_names


def compute_positional_encoding(
    data: Data,
    node_names: np.ndarray,
    range_start: int,
    range_end: int,
    damping_factor: float,
    max_iterations: int,
    tolerance: float,
) -> Tuple[torch.Tensor, np.ndarray]:
    if not 0 <= range_start < len(node_names):
        raise ValueError(f"--range-start must be in [0, {len(node_names) - 1}] for this graph.")
    if not range_start < range_end <= len(node_names):
        raise ValueError(f"--range-end must be in ({range_start}, {len(node_names)}] for this graph.")

    selected_names = node_names[range_start:range_end]
    logger.info(
        "Computing Personalized PageRank columns for STRING nodes [%d:%d) (%d seeds).",
        range_start,
        range_end,
        len(selected_names),
    )

    rank_columns: List[torch.Tensor] = []
    for local_idx in tqdm.tqdm(range(range_start, range_end), desc="PageRank seeds"):
        teleport_probs = torch.zeros(data.num_nodes, device=data.edge_index.device)
        teleport_probs[local_idx] = 1.0
        ranks = page_rank(
            data=data,
            teleport_probs=teleport_probs,
            damping_factor=damping_factor,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        max_rank = torch.max(ranks)
        if max_rank > 0:
            ranks = ranks / max_rank
        rank_columns.append(ranks.detach().cpu())

    ranks_matrix = torch.stack(rank_columns, dim=1)
    return ranks_matrix, selected_names


def main() -> None:
    args = parse_args()

    args.output_ranks.parent.mkdir(parents=True, exist_ok=True)
    args.output_genes.parent.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    logger.info("Using device: %s", device)
    logger.info("Official STRING version page: %s", STRING_VERSION_URL)
    logger.info("Official STRING download page: %s", STRING_DOWNLOAD_URL)
    logger.info("Official STRING API docs: %s", STRING_API_DOCS_URL)

    df_ppi = load_string_links(args.string_links)
    data, node_names = build_graph(
        df_ppi=df_ppi,
        connection_threshold=args.connection_threshold,
        weights=DEFAULT_WEIGHTS,
        device=device,
    )

    range_end = args.range_end if args.range_end is not None else len(node_names)
    ranks_matrix, selected_names = compute_positional_encoding(
        data=data,
        node_names=node_names,
        range_start=args.range_start,
        range_end=range_end,
        damping_factor=args.damping_factor,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
    )

    np.save(args.output_ranks, ranks_matrix.numpy().astype(np.float32))
    np.save(args.output_genes, selected_names.astype(object))

    logger.info("Saved Personalized PageRank matrix with shape %s to %s", tuple(ranks_matrix.shape), args.output_ranks)
    logger.info("Saved %d STRING node identifiers to %s", len(selected_names), args.output_genes)
    if args.range_start != 0 or range_end != len(node_names):
        logger.warning(
            "A partial STRING range was generated. Do not use these outputs as the global production PE assets unless that was intentional."
        )


if __name__ == "__main__":
    main()
