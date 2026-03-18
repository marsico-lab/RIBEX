#!/usr/bin/env python3
"""
Summarize and plot alanine-scan outputs produced by run_lora_inference.py.

The script reproduces the core notebook analysis in a CLI form:
- per-gene alanine summary tables
- top-dropping windows
- per-gene PDF plots with annotation overlays
- optional AlphaFold pLDDT overlays if the matching pickle is available
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


logger = logging.getLogger("analyze_alanine_scan")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and visualize alanine scan outputs.")
    parser.add_argument("--alanine-scan", type=Path, required=True, help="Alanine scan TSV from run_lora_inference.py.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plots and summary tables.")
    parser.add_argument("--checkpoint", type=Path, help="Optional checkpoint directory used to infer dataset metadata.")
    parser.add_argument("--dataset-path", type=Path, help="Optional dataset pickle overriding checkpoint metadata.")
    parser.add_argument("--plddt-pickle", type=Path, help="Optional pLDDT pickle for AlphaFold confidence overlays.")
    parser.add_argument("--plddt-window", type=int, default=10, help="Window size for pLDDT smoothing (default: 10).")
    parser.add_argument("--top-windows", type=int, default=25, help="Number of strongest negative windows to save.")
    parser.add_argument("--gene-id", action="append", default=[], help="Restrict analysis to specific Gene_ID values.")
    return parser.parse_args()


def resolve_dataset_path(args: argparse.Namespace) -> Optional[Path]:
    if args.dataset_path is not None:
        return args.dataset_path

    if args.checkpoint is None:
        return None

    metadata_path = args.checkpoint / "metadata.json"
    if not metadata_path.exists():
        logger.warning("Checkpoint metadata not found: %s", metadata_path)
        return None

    with open(metadata_path, "r") as fh:
        metadata = json.load(fh)

    dataset_path = metadata.get("dataset_paths", {}).get("fine_tuning")
    if not dataset_path:
        logger.warning("No fine-tuning dataset path in checkpoint metadata: %s", metadata_path)
        return None
    return Path(dataset_path)


def guess_plddt_pickle(dataset_path: Optional[Path]) -> Optional[Path]:
    if dataset_path is None:
        return None
    try:
        data_root = dataset_path.parents[1]
    except IndexError:
        return None
    return data_root / "data_raw" / "pLDDT_scores" / f"{dataset_path.stem}_pLDDT_dict.pkl"


def load_alanine_scan(path: Path, requested_genes: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Alanine scan TSV not found: {path}")

    df = pd.read_csv(path, sep="\t")
    required = {"Gene_ID", "start", "end", "delta"}
    missing = required.difference(df.columns)
    if missing:
        raise KeyError(f"{path} is missing required columns: {sorted(missing)}")

    df["Gene_ID"] = df["Gene_ID"].astype(str)
    df["start"] = pd.to_numeric(df["start"], errors="raise").astype(int)
    df["end"] = pd.to_numeric(df["end"], errors="raise").astype(int)
    df["delta"] = pd.to_numeric(df["delta"], errors="raise").astype(float)

    if requested_genes:
        requested = set(requested_genes)
        df = df[df["Gene_ID"].isin(requested)].copy()

    if df.empty:
        raise ValueError("No alanine scan records remain after filtering.")

    return df.sort_values(["Gene_ID", "start", "end"]).reset_index(drop=True)


def load_dataset(dataset_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if dataset_path is None:
        return None
    if not dataset_path.exists():
        logger.warning("Dataset pickle not found: %s", dataset_path)
        return None

    with open(dataset_path, "rb") as fh:
        dataset_df = pickle.load(fh)
    dataset_df = dataset_df.reset_index(drop=True)
    if "Gene_ID" in dataset_df.columns:
        dataset_df["Gene_ID"] = dataset_df["Gene_ID"].astype(str)
    return dataset_df


def parse_annotations(value: object) -> List[dict]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []

    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            logger.warning("Could not parse annotation string: %s", value)
            return []
    elif isinstance(value, (list, tuple)):
        parsed = value
    else:
        return []

    rows: List[dict] = []
    for ann in parsed:
        if not isinstance(ann, (list, tuple)) or len(ann) < 4:
            continue
        try:
            ann_start = int(ann[0])
            ann_end = int(ann[1])
        except (TypeError, ValueError):
            continue
        ann_type = str(ann[3])
        ann_name = str(ann[4]) if len(ann) > 4 else ann_type
        rows.append(
            {
                "ann_start": ann_start,
                "ann_end": ann_end,
                "ann_type": ann_type,
                "ann_name": ann_name,
            }
        )
    return rows


def build_annotation_table(dataset_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if dataset_df is None or "annotations" not in dataset_df.columns:
        return pd.DataFrame(columns=["Gene_ID", "ann_start", "ann_end", "ann_type", "ann_name"])

    rows: List[dict] = []
    for row in dataset_df[["Gene_ID", "annotations"]].itertuples(index=False):
        gene_id = str(row.Gene_ID)
        for ann in parse_annotations(row.annotations):
            ann["Gene_ID"] = gene_id
            rows.append(ann)
    return pd.DataFrame(rows)


def load_plddt_dict(path: Optional[Path]) -> Dict[str, object]:
    if path is None:
        return {}
    if not path.exists():
        logger.warning("pLDDT pickle not found: %s", path)
        return {}

    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, dict):
        logger.warning("Unexpected pLDDT pickle payload type: %s", type(obj).__name__)
        return {}
    logger.info("Loaded pLDDT scores for %d genes from %s", len(obj), path)
    return obj


def extract_plddt_scores(plddt_dict: Dict[str, object], gene_id: str) -> Optional[List[float]]:
    if gene_id not in plddt_dict:
        return None

    value = plddt_dict[gene_id]
    candidates: List[object] = []
    if isinstance(value, dict):
        if 1 in value:
            candidates.append(value[1])
        candidates.extend(value.values())
    else:
        candidates.append(value)

    for candidate in candidates:
        if isinstance(candidate, (list, tuple)):
            if len(candidate) >= 2 and isinstance(candidate[1], (list, tuple)):
                try:
                    return [float(x) for x in candidate[1]]
                except (TypeError, ValueError):
                    pass
            if candidate and all(isinstance(x, (int, float, np.integer, np.floating)) for x in candidate):
                return [float(x) for x in candidate]

    return None


def smooth_scores(scores: Sequence[float], window: int) -> np.ndarray:
    if len(scores) == 0:
        return np.array([], dtype=float)
    window = max(1, min(int(window), len(scores)))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(np.asarray(scores, dtype=float), kernel, mode="valid") / 100.0


def build_gene_summary(
    alanine_df: pd.DataFrame,
    dataset_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    sequence_lengths: Dict[str, int] = {}
    gene_names: Dict[str, str] = {}
    if dataset_df is not None:
        if "sequence" in dataset_df.columns:
            sequence_lengths = (
                dataset_df[["Gene_ID", "sequence"]]
                .drop_duplicates(subset=["Gene_ID"], keep="first")
                .assign(sequence_length=lambda df: df["sequence"].astype(str).str.len())
                .set_index("Gene_ID")["sequence_length"]
                .astype(int)
                .to_dict()
            )
        if "Gene_Name" in dataset_df.columns:
            gene_names = (
                dataset_df[["Gene_ID", "Gene_Name"]]
                .drop_duplicates(subset=["Gene_ID"], keep="first")
                .set_index("Gene_ID")["Gene_Name"]
                .astype(str)
                .to_dict()
            )

    rows: List[dict] = []
    for gene_id, gene_df in alanine_df.groupby("Gene_ID", sort=True):
        gene_df = gene_df.sort_values(["delta", "start", "end"]).reset_index(drop=True)
        worst = gene_df.iloc[0]
        best = gene_df.iloc[-1]
        row = {
            "Gene_ID": gene_id,
            "Gene_Name": gene_names.get(gene_id, ""),
            "sequence_length": sequence_lengths.get(gene_id, int(gene_df["end"].max())),
            "base_probability": float(gene_df["base_probability"].iloc[0]) if "base_probability" in gene_df.columns else np.nan,
            "n_windows": int(len(gene_df)),
            "min_delta": float(worst["delta"]),
            "min_delta_start": int(worst["start"]),
            "min_delta_end": int(worst["end"]),
            "min_delta_window": str(worst["original_window"]) if "original_window" in gene_df.columns else "",
            "max_delta": float(best["delta"]),
            "max_delta_start": int(best["start"]),
            "max_delta_end": int(best["end"]),
            "mean_delta": float(gene_df["delta"].mean()),
            "mean_abs_delta": float(gene_df["delta"].abs().mean()),
            "n_negative_windows": int((gene_df["delta"] < 0).sum()),
            "n_windows_delta_le_m0p01": int((gene_df["delta"] <= -0.01).sum()),
            "fraction_windows_delta_le_m0p01": float((gene_df["delta"] <= -0.01).mean()),
        }
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["min_delta", "Gene_ID"]).reset_index(drop=True)


def build_plddt_export(plddt_dict: Dict[str, object], gene_ids: Iterable[str]) -> pd.DataFrame:
    rows: List[dict] = []
    for gene_id in gene_ids:
        scores = extract_plddt_scores(plddt_dict, gene_id)
        if not scores:
            continue
        for pos, score in enumerate(scores, start=1):
            rows.append({"Gene_ID": gene_id, "position": pos, "plddt": float(score)})
    return pd.DataFrame(rows)


def sanitize_filename(text: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_", ".") else "_" for c in text)


def plot_gene(
    gene_id: str,
    gene_name: str,
    gene_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    plddt_scores: Optional[Sequence[float]],
    plddt_window: int,
    output_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(16, 4.5))
    ax1.plot(
        gene_df["start"].to_numpy(),
        gene_df["delta"].to_numpy(),
        color="blue",
        marker=".",
        linewidth=2.0,
        label="Alanine scan delta probability",
    )
    ax1.axhline(0.0, color="red", linestyle="--", linewidth=2.0, label="Zero")

    ax2 = ax1.twinx()
    line2 = None
    if plddt_scores:
        smoothed = smooth_scores(plddt_scores, plddt_window)
        if len(smoothed) > 0:
            x = np.arange(1, len(smoothed) + 1)
            (line2,) = ax2.plot(x, smoothed, color="green", linewidth=2.0, label="AlphaFold confidence score")
            ax2.set_ylim(0.0, 1.0)
            ax2.set_ylabel("AlphaFold confidence score", fontsize=14, color="green")
            ax2.tick_params(axis="y", labelcolor="green", labelsize=12)
    else:
        ax2.set_yticks([])

    ymin, ymax = ax1.get_ylim()

    legend_patches: List[mpatches.Patch] = []
    if not ann_df.empty:
        unique_types = sorted(ann_df["ann_type"].astype(str).unique().tolist())
        non_idr_types = [ann_type for ann_type in unique_types if ann_type != "IDR"]
        palette = plt.get_cmap("tab10")
        color_map = {ann_type: palette(idx % 10) for idx, ann_type in enumerate(non_idr_types)}
        color_map["IDR"] = "orange"

        for ann_type, group in ann_df.groupby("ann_type", sort=True):
            color = color_map.get(str(ann_type), "gray")
            alpha = 0.25 if str(ann_type) == "IDR" else 0.30
            for ann in group.itertuples(index=False):
                width = max(int(ann.ann_end) - int(ann.ann_start), 1)
                rect = plt.Rectangle(
                    (int(ann.ann_start) - 1, ymin),
                    width,
                    ymax - ymin,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor=None,
                    zorder=0,
                )
                ax1.add_patch(rect)

        if "IDR" in unique_types:
            legend_patches.append(
                mpatches.Patch(color="orange", alpha=0.25, label="Intrinsic Disordered Region (IDR)")
            )
        for ann_type in non_idr_types:
            legend_patches.append(mpatches.Patch(color=color_map[ann_type], alpha=0.30, label=ann_type))

    handles = [ax1.lines[0], ax1.lines[1]]
    if line2 is not None:
        handles.append(line2)
    handles.extend(legend_patches)
    ax1.legend(handles=handles, fontsize=11, loc="best", framealpha=0.4)

    title = gene_id if not gene_name else f"{gene_id} ({gene_name})"
    ax1.set_title(f"Alanine Scan Results for {title}", fontsize=16)
    ax1.set_xlabel("Position", fontsize=14)
    ax1.set_ylabel("Alanine scan delta probability", fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.tick_params(axis="both", labelsize=12)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    plots_dir = output_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    alanine_df = load_alanine_scan(args.alanine_scan, requested_genes=args.gene_id)
    dataset_path = resolve_dataset_path(args)
    dataset_df = load_dataset(dataset_path)
    annotation_df = build_annotation_table(dataset_df)

    plddt_pickle = args.plddt_pickle or guess_plddt_pickle(dataset_path)
    plddt_dict = load_plddt_dict(plddt_pickle)

    summary_df = build_gene_summary(alanine_df, dataset_df)
    top_windows_df = alanine_df.sort_values(["delta", "Gene_ID", "start", "end"]).reset_index(drop=True)
    top_windows_df = top_windows_df.head(max(1, int(args.top_windows)))

    summary_path = output_dir / "alanine_scan_summary.tsv"
    top_windows_path = output_dir / "alanine_scan_top_drops.tsv"
    annotation_path = output_dir / "annotations.tsv"
    plddt_path = output_dir / "plddt_scores.tsv"
    missing_ann_path = output_dir / "genes_missing_annotations.txt"
    missing_plddt_path = output_dir / "genes_missing_plddt.txt"

    summary_df.to_csv(summary_path, sep="\t", index=False)
    top_windows_df.to_csv(top_windows_path, sep="\t", index=False)
    if not annotation_df.empty:
        annotation_df.to_csv(annotation_path, sep="\t", index=False)

    plddt_export_df = build_plddt_export(plddt_dict, summary_df["Gene_ID"].tolist())
    if not plddt_export_df.empty:
        plddt_export_df.to_csv(plddt_path, sep="\t", index=False)

    gene_name_map: Dict[str, str] = {}
    if dataset_df is not None and "Gene_Name" in dataset_df.columns:
        gene_name_map = (
            dataset_df[["Gene_ID", "Gene_Name"]]
            .drop_duplicates(subset=["Gene_ID"], keep="first")
            .set_index("Gene_ID")["Gene_Name"]
            .astype(str)
            .to_dict()
        )

    missing_annotations: List[str] = []
    missing_plddt: List[str] = []
    for gene_id, gene_df in alanine_df.groupby("Gene_ID", sort=True):
        gene_annotations = annotation_df[annotation_df["Gene_ID"] == gene_id].copy()
        if gene_annotations.empty:
            missing_annotations.append(gene_id)

        scores = extract_plddt_scores(plddt_dict, gene_id)
        if not scores:
            missing_plddt.append(gene_id)

        output_path = plots_dir / f"{sanitize_filename(gene_id)}.pdf"
        plot_gene(
            gene_id=gene_id,
            gene_name=gene_name_map.get(gene_id, ""),
            gene_df=gene_df.sort_values("start").reset_index(drop=True),
            ann_df=gene_annotations,
            plddt_scores=scores,
            plddt_window=args.plddt_window,
            output_path=output_path,
        )

    with open(missing_ann_path, "w") as fh:
        for gene_id in sorted(set(missing_annotations)):
            fh.write(f"{gene_id}\n")

    with open(missing_plddt_path, "w") as fh:
        for gene_id in sorted(set(missing_plddt)):
            fh.write(f"{gene_id}\n")

    logger.info("Saved alanine summary to %s", summary_path)
    logger.info("Saved top-dropping windows to %s", top_windows_path)
    logger.info("Saved %d per-gene plots to %s", summary_df.shape[0], plots_dir)
    if not annotation_df.empty:
        logger.info("Saved annotation table to %s", annotation_path)
    if not plddt_export_df.empty:
        logger.info("Saved pLDDT export to %s", plddt_path)


if __name__ == "__main__":
    main()
