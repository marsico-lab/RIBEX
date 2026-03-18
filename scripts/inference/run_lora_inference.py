import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import PreTrainedTokenizerBase
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from scripts.initialize import initialize, log
from scripts.inference.load_lora_checkpoint import load_lora_checkpoint, load_finetune_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned LoRA checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to exported checkpoint directory.")
    parser.add_argument("--split", choices=["train", "val", "all"], default="val", help="Subset to score.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--output", help="Optional TSV file to write base predictions.")
    parser.add_argument("--limit", type=int, help="Limit number of proteins processed.")
    parser.add_argument("--gene-id", action="append", default=[], help="Filter to specific Gene_ID (repeatable).")
    parser.add_argument("--gene-ids-file", help="Path to file with one Gene_ID per line.")
    parser.add_argument("--alanine-scan", action="store_true", help="Enable alanine scanning for selected proteins.")
    parser.add_argument("--alanine-output", help="Optional TSV file for alanine scan results.")
    parser.add_argument("--alanine-max-length", type=int, default=1200, help="Skip alanine scan for sequences longer than this.")
    parser.add_argument("--alanine-max-samples", type=int, help="Limit number of proteins to scan.")
    parser.add_argument("--alanine-window-size", type=int, default=10, help="Sliding window size for alanine substitutions.")
    parser.add_argument("--pe-scan", action="store_true", help="Enable positional encoding dimension ablation.")
    parser.add_argument("--pe-scan-output", help="Optional TSV file for positional encoding scan results.")
    parser.add_argument("--pe-scan-target", type=float, default=0.0, help="Value to assign to a dimension during PE scan.")
    parser.add_argument("--override-dataset", help="Path to an alternative dataset to use (ignoring checkpoint metadata).")
    return parser.parse_args()


def select_dtype(name: str) -> Optional[torch.dtype]:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }.get(name, torch.float32)


def load_gene_ids(args: argparse.Namespace) -> Optional[set]:
    gene_ids: List[str] = []
    gene_ids.extend(args.gene_id)

    if args.gene_ids_file:
        path = Path(args.gene_ids_file)
        if not path.exists():
            raise FileNotFoundError(f"Gene IDs file not found: {path}")
        with open(path, "r") as fh:
            for line in fh:
                gene_id = line.strip()
                if gene_id:
                    gene_ids.append(gene_id)

    if not gene_ids:
        return None
    return set(gene_ids)


def chunked(iterable: Sequence[int], size: int) -> Iterable[List[int]]:
    for start in range(0, len(iterable), size):
        yield list(iterable[start:start + size])


def prepare_batch(
    batch_items: List[Dict],
    device: torch.device,
    target_dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([item["input_ids"] for item in batch_items]).to(device)
    attention_mask = torch.stack([item["attention_mask"] for item in batch_items]).to(device)
    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if "positional_encoding" in batch_items[0]:
        pe_arrays = []
        for item in batch_items:
            value = item["positional_encoding"]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            pe_arrays.append(value)
        pe = np.stack(pe_arrays).astype(np.float32)
        batch["positional_encoding"] = torch.from_numpy(pe).to(device=device, dtype=target_dtype)
    return batch


def alanine_scan(
    model: torch.nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    sequence: str,
    positional_encoding: Optional[np.ndarray],
    model_dtype: torch.dtype,
    window_size: int,
    max_length: int,
    batch_size: int,
    base_probability: float,
    max_length_allowed: int,
) -> List[Dict]:
    if len(sequence) > max_length_allowed:
        return []

    if len(sequence) == 0:
        return []

    window_size = max(1, window_size)
    window_size = min(window_size, len(sequence))

    mutants: List[Tuple[int, int, str, str]] = []
    for start in range(0, len(sequence) - window_size + 1):
        original_window = sequence[start:start + window_size]
        mutated_window = "A" * window_size
        if original_window == mutated_window:
            continue
        mutated = sequence[:start] + mutated_window + sequence[start + window_size :]
        mutants.append((start, window_size, original_window, mutated))

    if not mutants:
        return []

    positional_tensor = None
    if positional_encoding is not None:
        positional_tensor = torch.from_numpy(positional_encoding.astype(np.float32)).to(dtype=model_dtype)

    results: List[Dict] = []
    model_device = next(model.parameters()).device

    for chunk in chunked(mutants, batch_size):
        sequences = [entry[3] for entry in chunk]
        enc = tokenizer(
            sequences,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(model_device)
        attention_mask = enc["attention_mask"].to(model_device)
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if positional_tensor is not None:
            repeat_pe = positional_tensor.unsqueeze(0).repeat(len(chunk), 1).to(model_device)
            kwargs["positional_encoding"] = repeat_pe.to(dtype=model_dtype)

        with torch.no_grad():
            outputs = model(**kwargs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

        for (start, length, original_window, mutated_seq), prob in zip(chunk, probs):
            results.append(
                {
                    "start": start,
                    "end": start + length,
                    "window_size": length,
                    "original_window": original_window,
                    "mutated_window": "A" * length,
                    "probability": float(prob),
                    "delta": float(prob - base_probability),
                }
            )

    return results


def positional_encoding_scan(
    model: torch.nn.Module,
    base_inputs: Dict[str, torch.Tensor],
    positional_vector: torch.Tensor,
    model_dtype: torch.dtype,
    target_value: float,
    batch_limit: int,
) -> List[Dict]:
    if positional_vector is None:
        return []

    vector = positional_vector.to(dtype=model_dtype)
    if vector.ndim == 1:
        vector = vector.unsqueeze(0)
    base_vec = vector.squeeze(0)
    dim = base_vec.shape[-1]
    if dim == 0:
        return []

    device = base_vec.device
    chunk = max(1, batch_limit)
    results: List[Dict] = []

    base_inputs = {k: v.to(device) for k, v in base_inputs.items()}

    for start in range(0, dim, chunk):
        dims = list(range(start, min(start + chunk, dim)))
        batch_count = len(dims)
        mutated_inputs = {}
        for key, tensor in base_inputs.items():
            mutated_inputs[key] = tensor.repeat(batch_count, 1)

        mutated_pe = base_vec.repeat(batch_count, 1)
        for row_idx, dim_idx in enumerate(dims):
            mutated_pe[row_idx, dim_idx] = target_value
        mutated_inputs["positional_encoding"] = mutated_pe.to(device=device, dtype=model_dtype)

        with torch.no_grad():
            outputs = model(**mutated_inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()

        for dim_idx, prob in zip(dims, probs):
            results.append(
                {
                    "dimension": dim_idx,
                    "perturbed_value": target_value,
                    "probability": float(prob),
                }
            )

    return results
def main():
    args = parse_args()
    initialize(__file__)
    dtype = select_dtype(args.precision)
    device = torch.device(args.device)

    model, tokenizer, metadata = load_lora_checkpoint(args.checkpoint, device=device, dtype=dtype)
    model_dtype = next(model.parameters()).dtype
    
    # Pass override argument
    dataset, split_indices = load_finetune_dataset(
        metadata, 
        tokenizer, 
        dataset_path_override=args.override_dataset
    )
    dataset_df = dataset.dataSet_df

    indices = None
    if args.override_dataset:
        print("Dataset override active: Using entire dataset (ignoring splits) and filtering by arguments.")
        indices = torch.arange(len(dataset))
    elif args.split == "train":
        indices = split_indices.get("ft_train")
    elif args.split == "val":
        indices = split_indices.get("ft_val")
    else:
        indices = torch.arange(len(dataset))

    if indices is None:
        if args.override_dataset:
             # Should be covered above, but safety fallback
             indices = torch.arange(len(dataset))
        else:
             raise RuntimeError(f"Split '{args.split}' not available in checkpoint metadata.")

    # select samples with positive labels or high predicted probability
    positive_indices = []
    for idx in indices:
        row = dataset_df.iloc[idx.item()]
        # If specific gene requested, allow even if negative
        if args.gene_id:
             if row["Gene_ID"] in args.gene_id:
                 positive_indices.append(idx.item())
        # Otherwise filter for positives only (default behavior)
        elif row["positive"] == 1:
            positive_indices.append(idx.item())
    indices = torch.tensor(positive_indices, dtype=torch.long)
    indices_list = indices.cpu().tolist()
    if args.limit is not None:
        indices_list = indices_list[: args.limit] if len(indices_list) > args.limit else indices_list

    gene_filter = load_gene_ids(args)
    if gene_filter is not None:
        indices_list = [idx for idx in indices_list if str(dataset_df.iloc[idx]["Gene_ID"]) in gene_filter]

    if not indices_list:
        print("No samples selected for inference.")
        return

    batch_size = args.batch_size
    max_length = metadata.get("tokenizer_max_length", 1000)
    has_positional = bool(metadata.get("uses_positional_encoding", False)) and bool(getattr(dataset, "return_pe", False))
    pe_scan_enabled = args.pe_scan and has_positional
    if args.pe_scan and not has_positional:
        print("Positional encodings not available in this dataset; skipping positional scan.")

    base_records: List[Dict] = []
    alanine_records: List[Dict] = []
    alanine_processed = 0
    pe_scan_records: List[Dict] = []

    with torch.no_grad():
        for batch_indices in chunked(indices_list, batch_size):
            batch_items = [dataset[i] for i in batch_indices]
            batch = prepare_batch(batch_items, device, model_dtype)

            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            logits_pos = outputs.logits[:, 1].detach().cpu().numpy()

            for local_idx, global_idx in enumerate(batch_indices):
                row = dataset_df.iloc[global_idx]
                gene_id = str(row["Gene_ID"])
                label = int(row["positive"])
                sequence = dataset.sequences[global_idx]
                base_prob = float(probs[local_idx])
                base_records.append(
                    {
                        "index": global_idx,
                        "Gene_ID": gene_id,
                        "label": label,
                        "probability": base_prob,
                        "logit": float(logits_pos[local_idx]),
                        "sequence_length": len(sequence),
                    }
                )

                if args.alanine_scan:
                    if args.alanine_max_samples is not None and alanine_processed >= args.alanine_max_samples:
                        continue
                    positional = batch.get("positional_encoding")
                    positional_np = None
                    if positional is not None:
                        positional_np = positional[local_idx].to(dtype=model_dtype).cpu().numpy()

                    scan_results = alanine_scan(
                        model=model,
                        tokenizer=tokenizer,
                        sequence=sequence,
                        positional_encoding=positional_np,
                        model_dtype=model_dtype,
                        window_size=args.alanine_window_size,
                        max_length=max_length,
                        batch_size=batch_size,
                        base_probability=base_prob,
                        max_length_allowed=args.alanine_max_length,
                    )
                    for entry in scan_results:
                        entry.update(
                            {
                                "Gene_ID": gene_id,
                                "index": global_idx,
                                "base_probability": base_prob,
                            }
                        )
                        alanine_records.append(entry)
                    alanine_processed += 1
                    print(f"Processed alanine scan for {alanine_processed} proteins.", end="\r")

                if pe_scan_enabled:
                    #import ipdb; ipdb.set_trace()
                    positional = batch.get("positional_encoding")
                    if positional is not None:
                        pe_vector = positional[local_idx : local_idx + 1].detach()
                        base_inputs = {
                            "input_ids": batch["input_ids"][local_idx : local_idx + 1],
                            "attention_mask": batch["attention_mask"][local_idx : local_idx + 1],
                        }
                        scan_results = positional_encoding_scan(
                            model=model,
                            base_inputs=base_inputs,
                            positional_vector=pe_vector,
                            model_dtype=model_dtype,
                            target_value=args.pe_scan_target,
                            batch_limit=batch_size,
                        )
                        for entry in scan_results:
                            dim_idx = entry["dimension"]
                            entry.update(
                                {
                                    "Gene_ID": gene_id,
                                    "index": global_idx,
                                    "base_probability": base_prob,
                                    "original_value": float(pe_vector[0, dim_idx].item()),
                                    "delta": entry["probability"] - base_prob,
                                }
                            )
                            pe_scan_records.append(entry)
# np.array([base_record['label'] for base_record in base_records])[np.where(np.array([base_record['probability'] for base_record in base_records]) > 0.6)[0]]
    if base_records:
        preview = base_records[: min(5, len(base_records))]
        print("Sample predictions:")
        print(json.dumps(preview, indent=2))

    if args.output:
        import pandas as pd

        df = pd.DataFrame(base_records)
        df.to_csv(args.output, sep="\t", index=False)
        print(f"Wrote predictions to {args.output}")

    if args.alanine_scan and args.alanine_output:
        import pandas as pd

        df = pd.DataFrame(alanine_records)
        df.to_csv(args.alanine_output, sep="\t", index=False)
        print(f"Wrote alanine scan results to {args.alanine_output}")

    if pe_scan_enabled and args.pe_scan_output:
        import pandas as pd

        df = pd.DataFrame(pe_scan_records)
        df.to_csv(args.pe_scan_output, sep="\t", index=False)
        print(f"Wrote positional encoding scan results to {args.pe_scan_output}")


if __name__ == "__main__":
    main()
