#!/usr/bin/env python3
"""
Analyze positional encoding (PE) scan results by reconstructing the full PE vector
via inverse PCA and visualizing the impact of critical dimensions.

The script expects the same checkpoint directory that was used for inference
and the TSV file saved by `run_lora_inference.py --pe-scan`.

Example
-------
python scripts/training/analyze_pe_scan_effect.py \
    --checkpoint /path/to/exported/checkpoint \
    --pe-scan /path/to/val_pe_scan.tsv
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

try:
    from adjustText import adjust_text
except ImportError:  # pragma: no cover - optional dependency for nicer plots
    adjust_text = None
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from scripts.inference.load_lora_checkpoint import load_metadata
from scripts.data_sets.positional_encoding_processing import get_posenc_pkg


logger = logging.getLogger("analyze_pe_scan")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze positional encoding scan results.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint directory used for inference.")
    parser.add_argument("--pe-scan", type=Path, required=True, help="TSV file produced by run_lora_inference.py --pe-scan-output.")
    parser.add_argument("--output-dir", type=Path,
                        help="Directory for plots and tables (default: alongside PE scan file).")
    parser.add_argument("--delta-threshold", type=float, default=-0.01,
                        help="Delta cutoff to mark PE dimensions as critical (default: -0.01).")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0, help="Perplexity for TSNE embedding.")
    parser.add_argument("--tsne-random-state", type=int, default=42, help="Random seed for TSNE.")
    parser.add_argument("--label-critical", type=int, default=15,
                        help="Number of high-impact nodes to annotate on the plot.")
    parser.add_argument("--label-scan", type=int, default=10,
                        help="Number of nodes from the PE scan to annotate separately.")
    parser.add_argument("--top-node-count", type=int, default=40,
                        help="Number of top-difference nodes per gene to track for frequency counts.")
    parser.add_argument("--include-inversion", action="store_true",
                        help="Also evaluate sign-inverted PE dimensions (matches non-default inference setups).")
    parser.add_argument("--original-space", action="store_true",
                        help="Measure PE differences in the original (unscaled) ranks space instead of the "
                             "standardised space provided to PCA.")
    parser.add_argument("--pdf-name", type=str, default=None,
                        help="Custom name for the output PDF file (without extension). "
                             "Default: 'pe_scan_tsne' or 'pe_scan_tsne_original_space' if --original-space is used.")
    parser.add_argument("--tsne-save-path", type=Path, default=None,
                        help="Path to save the computed t-SNE coordinates as a .npy file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs in the target directory.")
    return parser.parse_args()


def ensure_output_dir(path: Path, overwrite: bool) -> Path:
    if path.exists():
        if not overwrite and any(path.iterdir()):
            raise FileExistsError(f"Output directory '{path}' already exists and is not empty. "
                                  "Use --overwrite to reuse it.")
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path


def load_string_ids(pkg) -> np.ndarray:
    string_ids = np.load(pkg.gene_names_file, allow_pickle=True).astype(str)
    if string_ids.ndim != 1:
        raise ValueError("Loaded STRING identifier list has unexpected shape.")
    return string_ids


def inverse_pca(
    vector: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    scaler_mean: Optional[np.ndarray] = None,
    scaler_scale: Optional[np.ndarray] = None,
    original_space: bool = False,
) -> np.ndarray:
    """
    Invert a PCA vector back into feature space.

    Parameters
    ----------
    vector : np.ndarray
        PCA-reduced vector.
    components : np.ndarray
        PCA component matrix.
    mean : np.ndarray
        Mean of the scaled data used in PCA.
    scaler_mean : Optional[np.ndarray]
        Mean of the StandardScaler that produced the scaled data.
    scaler_scale : Optional[np.ndarray]
        Scale (std) of the StandardScaler.
    original_space : bool
        When True, return data in the original (unscaled) ranks space.
    """
    recon_scaled = np.dot(vector, components) + mean
    if original_space:
        if scaler_mean is None or scaler_scale is None:
            raise ValueError("Scaler statistics required to reconstruct in original space.")
        recon_original = recon_scaled * scaler_scale + scaler_mean
        return recon_original
    return recon_scaled


def compute_gene_statistics(
    pe_scan_df: pd.DataFrame,
    pe_matrix: np.ndarray,
    components: np.ndarray,
    mean: np.ndarray,
    scaler_mean: Optional[np.ndarray],
    scaler_scale: Optional[np.ndarray],
    string_ids: Sequence[str],
    gene_to_string: Dict[str, str],
    string_to_index: Dict[str, int],
    delta_threshold: float,
    top_node_count: int,
    include_inversion: bool,
    original_space: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    if components is None or mean is None:
        raise ValueError("PCA components and mean must be provided.")

    n_nodes = len(string_ids)

    # aggregate containers
    node_zero_sum = np.zeros(n_nodes, dtype=np.float64)
    node_invert_sum = np.zeros(n_nodes, dtype=np.float64)
    node_zero_counts = np.zeros(n_nodes, dtype=np.int64)
    node_invert_counts = np.zeros(n_nodes, dtype=np.int64)

    critical_records: List[Dict] = []
    missing_genes: List[str] = []

    filtered_dims = pe_scan_df[pe_scan_df["delta"] <= delta_threshold].copy()
    dimension_records: List[Dict] = []
    if not filtered_dims.empty:
        for dim, grp in filtered_dims.groupby("dimension"):
            dimension_records.append(
                {
                    "dimension": int(dim),
                    "count": len(grp),
                    "mean_delta": float(grp["delta"].mean()),
                    "min_delta": float(grp["delta"].min()),
                    "max_delta": float(grp["delta"].max()),
                    "mean_original_value": float(grp["original_value"].mean()),
                }
            )
    for gene_id, group in pe_scan_df.groupby("Gene_ID"):
        str_gene = str(gene_id)
        string_id = gene_to_string.get(str_gene)
        if string_id is None:
            missing_genes.append(str_gene)
            continue

        critical_dims = group.loc[group["delta"] <= delta_threshold, "dimension"].astype(int).to_numpy()
        if critical_dims.size == 0:
            continue

        idx = string_to_index.get(string_id)
        if idx is None or idx >= pe_matrix.shape[0]:
            missing_genes.append(str_gene)
            continue

        base_vec = pe_matrix[idx].astype(np.float32, copy=True)
        if base_vec.shape[0] != components.shape[0]:
            raise ValueError(
                f"PE vector length {base_vec.shape[0]} for gene {str_gene} does not match PCA component count {components.shape[0]}."
            )

        base_recon = inverse_pca(
            base_vec,
            components,
            mean,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            original_space=original_space,
        )
        zero_vec = base_vec.copy()
        zero_vec[critical_dims] = 0.0
        zero_recon = inverse_pca(
            zero_vec,
            components,
            mean,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            original_space=original_space,
        )

        zero_diff = zero_recon - base_recon
        abs_zero = np.abs(zero_diff)

        node_zero_sum += abs_zero

        invert_diff = None
        abs_invert = None
        if include_inversion:
            invert_vec = base_vec.copy()
            invert_vec[critical_dims] *= -1.0
            invert_recon = inverse_pca(
                invert_vec,
                components,
                mean,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                original_space=original_space,
            )
            invert_diff = invert_recon - base_recon
            abs_invert = np.abs(invert_diff)
            node_invert_sum += abs_invert

        if top_node_count > 0:
            top_zero_idx = np.argsort(abs_zero)[::-1][: min(len(abs_zero), top_node_count)]
            if include_inversion and abs_invert is not None:
                top_invert_idx = np.argsort(abs_invert)[::-1][: min(len(abs_invert), top_node_count)]
        else:
            top_zero_idx = np.arange(len(abs_zero), dtype=int)
            if include_inversion and abs_invert is not None:
                top_invert_idx = np.arange(len(abs_invert), dtype=int)
        node_zero_counts[top_zero_idx] += 1
        if include_inversion and abs_invert is not None:
            node_invert_counts[top_invert_idx] += 1

        record = {
            "Gene_ID": str_gene,
            "string_id": string_id,
            "base_probability": float(group["base_probability"].iloc[0]),
            "min_delta": float(group["delta"].min()),
            "mean_delta": float(group["delta"].mean()),
            "sum_delta": float(group["delta"].sum()),
            "n_dimensions_scanned": int(len(group)),
            "n_critical_dimensions": int(len(critical_dims)),
            "critical_dimensions": critical_dims.tolist(),
            "zero_l2_diff": float(np.linalg.norm(zero_diff)),
            "zero_mean_abs_diff": float(np.mean(np.abs(zero_diff))),
        }
        if include_inversion and invert_diff is not None:
            record.update(
                {
                    "invert_l2_diff": float(np.linalg.norm(invert_diff)),
                    "invert_mean_abs_diff": float(np.mean(np.abs(invert_diff))),
                }
            )
        critical_records.append(record)

    gene_stats = pd.DataFrame(critical_records)
    dimension_stats = pd.DataFrame(dimension_records)
    node_data = {
        "string_id": string_ids,
        "zero_total_abs_change": node_zero_sum,
        "zero_top_frequency": node_zero_counts,
    }
    if include_inversion:
        node_data.update(
            {
                "invert_total_abs_change": node_invert_sum,
                "invert_top_frequency": node_invert_counts,
            }
        )
    node_stats = pd.DataFrame(node_data)
    combined_abs = node_zero_sum + (node_invert_sum if include_inversion else 0.0)
    combined_freq = node_zero_counts + (node_invert_counts if include_inversion else 0)
    node_stats["combined_abs_change"] = combined_abs
    node_stats["combined_top_frequency"] = combined_freq

    return gene_stats, dimension_stats, node_stats, missing_genes


def run_tsne(pe_matrix: np.ndarray, perplexity: float, random_state: int) -> np.ndarray:
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="random")
    return tsne.fit_transform(pe_matrix)


def annotate_points(ax, indices: Iterable[int], tsne_results: np.ndarray, labels: List[str],
                    color: str, n_label: int) -> None:
    if n_label <= 0:
        return
    texts = []
    for idx in list(indices)[:n_label]:
        x, y = tsne_results[idx]
        texts.append(ax.text(x, y, labels[idx], fontsize=10, color=color, fontweight="bold"))
    if texts and adjust_text is not None:
        adjust_text(texts, ax=ax, arrowprops={'arrowstyle': '-', 'color': 'gray', 'lw': 0.5, 'alpha': 0.8})


def plot_results(
    tsne_results: np.ndarray,
    string_ids: Sequence[str],
    node_stats: pd.DataFrame,
    gene_stats: pd.DataFrame,
    pe_scan_nodes: List[int],
    string_to_gene_alias: Dict[str, str],
    output_path: Path,
    label_critical: int,
    label_scan: int,
) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(14, 12))
    cb_palette = sns.color_palette("colorblind")
    #import ipdb; ipdb.set_trace()
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], label="All Nodes", color="lightgray", alpha=0.6, s=40)

    significant_nodes = node_stats[node_stats["combined_top_frequency"] > 0]
    if not significant_nodes.empty:
        indices = significant_nodes.index.to_numpy()
        counts = significant_nodes["combined_top_frequency"].to_numpy(dtype=float)
        scatter_neighbors = ax.scatter(
            tsne_results[indices, 0],
            tsne_results[indices, 1],
            c=counts,
            cmap="Blues",
            label="Frequently Perturbed Nodes",
            alpha=0.9,
            s=70,
            edgecolor="black",
            linewidth=0.5,
        )
        cbar = fig.colorbar(scatter_neighbors, ax=ax, shrink=0.6)
        cbar.set_label("Top-frequency count", fontsize=12)

    if pe_scan_nodes:
        ax.scatter(
            tsne_results[pe_scan_nodes, 0],
            tsne_results[pe_scan_nodes, 1],
            label="PE Scan Nodes",
            color=cb_palette[1],
            marker="D",
            alpha=0.8,
            s=45,
            edgecolor="black",
            linewidth=0.6,
        )

    labels = [string_to_gene_alias.get(sid, sid) for sid in string_ids]
    if not significant_nodes.empty and label_critical > 0:
        top_nodes = significant_nodes.sort_values("combined_top_frequency", ascending=False).index.to_numpy(dtype=int)
        annotate_points(
            ax,
            top_nodes,
            tsne_results,
            labels,
            color=cb_palette[0],
            n_label=label_critical,
        )

    if pe_scan_nodes and label_scan > 0:
        annotate_points(
            ax,
            pe_scan_nodes,
            tsne_results,
            labels,
            color=cb_palette[1],
            n_label=label_scan,
        )

    legend_handles = [
        Line2D([], [], marker="o", linestyle="None", color="lightgray", alpha=0.8, label="All Nodes"),
        Line2D([], [], marker="o", linestyle="None", color=cb_palette[0], markerfacecolor=cb_palette[0],
               markeredgecolor="black", label="Frequent Perturbations"),
        Line2D([], [], marker="D", linestyle="None", color=cb_palette[1], markerfacecolor=cb_palette[1],
               markeredgecolor="black", label="PE Scan Nodes"),
    ]
    ax.legend(handles=legend_handles, title="Node Categories", fontsize=12, title_fontsize=13, loc="lower left",
              frameon=True, facecolor="white")

    ax.set_title("t-SNE of positional encodings with PE scan highlights", fontsize=18, pad=20)
    ax.set_xlabel("t-SNE Component 1", fontsize=14)
    ax.set_ylabel("t-SNE Component 2", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint}")
    metadata = load_metadata(args.checkpoint)
    if not metadata.get("uses_positional_encoding", False):
        raise RuntimeError("Checkpoint metadata indicates positional encodings were not used.")

    pe_dim = metadata.get("pe_dim")
    if pe_dim is None:
        raise KeyError("Checkpoint metadata lacks 'pe_dim'; cannot reconstruct PE vectors.")
    dataset_path = Path(metadata["dataset_paths"]["fine_tuning"])
    pkg = get_posenc_pkg(datafile=str(dataset_path), pca_n_components=int(pe_dim))
    if pkg.pca_components is None or pkg.pca_mean is None:
        raise RuntimeError("Positional encoding package does not include PCA components/mean.")
    if args.original_space and (pkg.scaler_mean is None or pkg.scaler_scale is None):
        raise RuntimeError(
            "Scaler statistics missing in positional encoding package; cannot reconstruct in original space."
        )
    if not args.pe_scan.exists():
        raise FileNotFoundError(f"PE scan TSV not found: {args.pe_scan}")
    pe_scan_df = pd.read_csv(args.pe_scan, sep="\t")
    if "Gene_ID" not in pe_scan_df.columns or "delta" not in pe_scan_df.columns:
        raise KeyError("PE scan TSV must contain 'Gene_ID' and 'delta' columns.")
    pe_scan_df["Gene_ID"] = pe_scan_df["Gene_ID"].astype(str)

    output_dir = args.output_dir or args.pe_scan.parent.joinpath("pe_scan_analysis")
    output_dir = ensure_output_dir(output_dir, overwrite=args.overwrite)

    string_ids = load_string_ids(pkg)
    gene_to_string = {str(k): v for k, v in pkg.genes_ids_to_string_ids.items()}
    string_to_index = {k: int(v) for k, v in pkg.string_ids_to_ranks_indices.items()}
    string_to_gene_alias: Dict[str, str] = {}
    for string in string_ids:
        aliases = sorted([gene for gene, sid in gene_to_string.items() if sid == string])
        if not aliases:
            string_to_gene_alias[string] = string
        elif len(aliases) == 1:
            string_to_gene_alias[string] = aliases[0]
        else:
            string_to_gene_alias[string] = ";".join(aliases[:3])

    pe_base_matrix = pkg.ranks_reduced
    if pe_base_matrix is None:
        raise RuntimeError("Positional encoding package lacks PCA-reduced data required for analysis.")

    gene_stats, dimension_stats, node_stats, missing_genes = compute_gene_statistics(
        pe_scan_df=pe_scan_df,
        pe_matrix=pe_base_matrix,
        components=pkg.pca_components,
        mean=pkg.pca_mean,
        scaler_mean=pkg.scaler_mean,
        scaler_scale=pkg.scaler_scale,
        string_ids=string_ids,
        gene_to_string=gene_to_string,
        string_to_index=string_to_index,
        delta_threshold=args.delta_threshold,
        top_node_count=args.top_node_count,
        include_inversion=args.include_inversion,
        original_space=args.original_space,
    )
    # change file name if original-space
    summary_path = output_dir / "gene_pe_effects_original_space.tsv" if args.original_space else output_dir / "gene_pe_effects.tsv"
    gene_stats.to_csv(summary_path, sep="\t", index=False)
    logger.info("Saved gene-level PE analysis to %s", summary_path)

    if not dimension_stats.empty:
        dimension_path = output_dir / "dimension_summary_original_space.tsv" if args.original_space else output_dir / "dimension_summary.tsv"
        dimension_stats.to_csv(dimension_path, sep="\t", index=False)
        logger.info("Saved dimension summary to %s", dimension_path)

    node_summary_path = output_dir / "node_pe_sensitivity_original_space.tsv" if args.original_space else output_dir / "node_pe_sensitivity.tsv"
    node_stats.to_csv(node_summary_path, sep="\t", index=False)
    logger.info("Saved node-level aggregation to %s", node_summary_path)

    if missing_genes:
        missing_path = output_dir / "missing_genes.txt"
        with open(missing_path, "w") as fh:
            for gene in sorted(set(missing_genes)):
                fh.write(f"{gene}\n")
        logger.warning("Some genes from the PE scan were not found in the dataset (%d). See %s.",
                       len(set(missing_genes)), missing_path)

    tsne_results = run_tsne(pe_base_matrix, perplexity=args.tsne_perplexity, random_state=args.tsne_random_state)

    scan_nodes: List[int] = []
    for gene in sorted(pe_scan_df["Gene_ID"].astype(str).unique()):
        string_id = gene_to_string.get(gene)
        if string_id:
            idx = string_to_index.get(string_id)
            if idx is not None:
                scan_nodes.append(idx)
    scan_nodes = sorted(set(scan_nodes))

    # Save t-SNE coordinates if requested or by default alongside results
    tsne_save_path = args.tsne_save_path or output_dir / "tsne_coordinates.npy"
    np.save(tsne_save_path, tsne_results)
    logger.info("Saved t-SNE coordinates to %s", tsne_save_path)

    if args.pdf_name:
        pdf_filename = f"{args.pdf_name}.pdf"
    else:
        pdf_filename = "pe_scan_tsne_original_space.pdf" if args.original_space else "pe_scan_tsne.pdf"
    plot_path = output_dir / pdf_filename
    plot_results(
        tsne_results=tsne_results,
        string_ids=string_ids,
        node_stats=node_stats,
        gene_stats=gene_stats,
        pe_scan_nodes=scan_nodes,
        string_to_gene_alias=string_to_gene_alias,
        output_path=plot_path,
        label_critical=args.label_critical,
        label_scan=args.label_scan,
    )
    logger.info("Saved t-SNE visualization to %s", plot_path)

    logger.info("Analysis complete. Processed %d genes with critical PE dimensions.",
                int((gene_stats["n_critical_dimensions"] > 0).sum()) if not gene_stats.empty else 0)


if __name__ == "__main__":
    main()
