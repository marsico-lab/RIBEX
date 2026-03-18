import json
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase
from peft import PeftModel
import pandas as pd

from scripts.training.utils import EsmWithPE
from scripts.training.dataset import LoraDataset
from scripts.data_sets.positional_encoding_processing import get_posenc_pkg, build_pe_matrix_for_dataset


def _disable_dropout(module: nn.Module) -> None:
    module.eval()
    for child in module.modules():
        if isinstance(child, nn.Dropout):
            child.p = 0.0


def load_metadata(checkpoint_dir: Union[str, Path]) -> Dict:
    checkpoint_dir = Path(checkpoint_dir)
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {checkpoint_dir}")
    with open(metadata_path, "r") as fh:
        return json.load(fh)


def load_lora_checkpoint(
    checkpoint_dir: Union[str, Path],
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[nn.Module, PreTrainedTokenizerBase, Dict]:
    """
    Load a fine-tuned LoRA checkpoint exported by save_final_lora_checkpoint.
    Returns (model, tokenizer, metadata).
    """
    checkpoint_dir = Path(checkpoint_dir)
    metadata = load_metadata(checkpoint_dir)

    lm_repo = metadata.get("lm_repo")
    trust_remote_code = metadata.get("trust_remote_code", False)
    num_labels = metadata.get("num_labels", 2)

    tokenizer_subdir = metadata.get("tokenizer_subdir")
    if tokenizer_subdir:
        tokenizer_path = checkpoint_dir / tokenizer_subdir
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(lm_repo, trust_remote_code=trust_remote_code)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        lm_repo,
        trust_remote_code=trust_remote_code,
        num_labels=num_labels,
    )

    adapter_subdir = metadata.get("adapter_subdir", "adapter")
    adapter_path = checkpoint_dir / adapter_subdir
    model = PeftModel.from_pretrained(base_model, adapter_path)

    uses_positional = metadata.get("uses_positional_encoding", False)
    pe_dim = metadata.get("pe_dim")
    head_state_file = metadata.get("esm_head_state")
    if uses_positional or head_state_file:
        model = EsmWithPE(model, pe_dim or 2, num_labels=num_labels, p_pe_drop=0.0)
        if head_state_file:
            head_pkg = torch.load(checkpoint_dir / head_state_file, map_location="cpu")
            head_state = head_pkg.get("state_dict", {})
            missing, unexpected = model.load_state_dict(head_state, strict=False) # strict False not to overwrite LoRA weights and PE weights
            if missing: # Why checking all that is not lora and it's automatically missing? is this a problem?
                print(f"[WARN] Missing keys while loading head state: {missing}")
            if unexpected:
                print(f"[WARN] Unexpected keys while loading head state: {unexpected}")

    _disable_dropout(model)

    if device is not None:
        model.to(device)
    if dtype is not None:
        model.to(dtype=dtype)

    return model, tokenizer, metadata


def load_finetune_dataset(
    metadata: Dict,
    tokenizer: PreTrainedTokenizerBase,
    include_positional: bool = True,
    dataset_path_override: Optional[str] = None,
) -> Tuple[LoraDataset, Dict[str, torch.Tensor]]:
    """
    Recreate the fine-tuning dataset using metadata and the provided tokenizer.
    Returns the dataset and a dict of split name -> index tensor.
    """
    if dataset_path_override:
        dataset_path = Path(dataset_path_override)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Override dataset not found: {dataset_path}")
        print(f"Overriding dataset path: {dataset_path}")
    else:
        dataset_path = Path(metadata["dataset_paths"]["fine_tuning"])

    max_length = metadata.get("tokenizer_max_length", 1000)
    dataset = LoraDataset(dataset_path, tokenizer, max_length=max_length)

    if include_positional and metadata.get("uses_positional_encoding", False):
        pe_dim = metadata.get("pe_dim")
        # Ensure PE matrix building works for override dataset too if compatible
        # Note: Position encoding packages are cached by filename hash, so this should work if pre-computed
        try:
            pkg = get_posenc_pkg(datafile=str(dataset_path), pca_n_components=pe_dim)
            pe_matrix = build_pe_matrix_for_dataset(dataset, pkg, use_pca=True)
            dataset.set_positional_encodings(pe_matrix)
        except Exception as e:
            print(f"Warning: Could not load positional encodings for dataset {dataset_path}: {e}")

    split_indices: Dict[str, torch.Tensor] = {}
    
    # Only load splits if NOT overriding (splits are tied to specific dataset files/indices)
    if not dataset_path_override:
        for split_name, split_path in metadata.get("splits", {}).items():
            path = Path(split_path)
            if path.exists():
                df = pd.read_csv(path, sep="\t")
                split_indices[split_name] = torch.tensor(df["index"].to_numpy(), dtype=torch.long)
    else:
        print("Dataset override active: Excluding original split indices (they likely don't match).")

    return dataset, split_indices
