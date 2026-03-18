
import torch
from torch.utils.data import DistributedSampler
from tqdm import tqdm
import inspect
from natsort import natsorted
from pathlib import Path
import sys
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *
from safetensors.torch import load_file
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
import argparse
from peft import get_peft_model
import os
import json
import copy
import math
import torch.nn.functional as F
import re

LM_PROVIDER_ALIASES = {
    "synthyra": {
        "esm2_t6_8M_UR50D": "Synthyra/ESM2-8M",
        "esm2_t12_35M_UR50D": "Synthyra/ESM2-35M",
        "esm2_t30_150M_UR50D": "Synthyra/ESM2-150M",
        "esm2_t33_650M_UR50D": "Synthyra/ESM2-650M",
        "esm2_t36_3B_UR50D": "Synthyra/ESM2-3B",
    },
    "facebook": {
        "protT5_xl_uniref50": "Rostlab/prot_t5_xl_uniref50",
    },
}

DEFAULT_LM_PROVIDER = "facebook"


def _sanitize_for_path(value: str) -> str:
    text = str(value).strip()
    if not text:
        return "run"
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    text = text.strip("._")
    return text or "run"


def _ensure_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            pass
    try:
        return float(value)
    except Exception:
        return str(value)


def parseArguments():
    """Parse commandline arguments for training script and return them as "params" dictionary"""

    parser = argparse.ArgumentParser(description="Training script")
    
    parser.add_argument(
        "--local_rank", "--local-rank",
        type=int,
        default=0,
        help="Local process rank for distributed training"
    )

    parser.add_argument(
        "-M", "--modelName", dest="model_name", action="store", help="Model to be trained. See scripts/models for options", default="Linear"
    )
    parser.add_argument(
        "-DS",
        "--dataSet",
        dest="data_set_name",
        action="store",
        help="The dataset filename that should be used. See $DATA/data_sets for options",
        default="bressin19_human_pre-training.pkl",
    )
    parser.add_argument(
        "-lm",
        "--languageModel",
        dest="LM_name",
        action="store",
        help=(
            "Language Model. Options: esm1b_t33_650M_UR50S, "
            "esm2_t6_8M_UR50D, esm2_t12_35M_UR50D, esm2_t30_150M_UR50D, "
            "esm2_t33_650M_UR50D, esm2_t36_3B_UR50D, esm2_t48_15B_UR50D, "
            "protT5_xl_uniref50"
        ),
        default="esm1b_t33_650M_UR50S",
    )
    parser.add_argument(
        "--lm_repo",
        dest="lm_repo",
        action="store",
        help="Full Hugging Face repository id for the language model (e.g. facebook/esm2_t33_650M_UR50D). Defaults to facebook/<LM_name> when omitted.",
        default="",
    )
    parser.add_argument(
        "--lm_provider",
        dest="lm_provider",
        action="store",
        choices=["facebook", "synthyra"],
        help="Base model provider to resolve the Hugging Face repo when --lm_repo is not supplied.",
        default=DEFAULT_LM_PROVIDER,
    )
    parser.add_argument(
        "-ef",
        "--embeddingSubfolder",
        dest="embeddingSubfolder",
        action="store",
        help="Embedding subfolder. Options: bressin19, RIC, InterPro",
        default="bressin19",
    )
    parser.add_argument("-S", "--seed", dest="seed", type=int, default=2023, help="Training seed")
    parser.add_argument(
        "-D", "--devices", dest="devices", action="store", nargs="+", type=int, help='GPU IDs to be used. -1 for cpu usage', default="3" #"auto" uses all
    )
    parser.add_argument(
        "-p",
        "--precision",
        dest="precision",
        action="store",
        help="Model precision. Either f32, f16 or auto (for automatic mixed precision)",
        default="auto",
    )
    parser.add_argument(
        "-f",
        "--fsdp",
        dest="fsdp",
        action="store",
        help="Use Fully Sharded Data Parallel. Use this flag if your model is to big for your GPU and you want to offload also tp CPU",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "-e", "--epochs", dest="epochs", type=int, default=30, help="(Maximal) Epochs of training (if early stopping does not stop before)"
    )
    parser.add_argument("-lr", "--learning_rate", dest="lr", type=float, default=0.005, help="Learning rate") #Peng default was 0.00005
    parser.add_argument("-bs", "--batch_size", dest="bs", type=int, default=1024, help="Batch size")
    parser.add_argument(
        "-cpt",
        "--checkpoint-folder",
        dest="checkpoint_folder",
        action="store",
        help="Name of model instance folder where the latest checkpoint shall be used. Leave empty if a new instance shall be created.",
        default="" #e.g. 
    )
    parser.add_argument("-ds", "--split", dest="split", type=float, nargs="+", default=(0.8, 0.1, 0.1), help="Dataset split")
    parser.add_argument(
        "-pt", "--patience", dest="patience", type=int, default=40, help="Patience of early stopping. Set 0 for no early stopping"
    )
    parser.add_argument("-t", "--threads", dest="threads", type=int, default=16, help="Number of threads for parallel processing")
    parser.add_argument(
        "--pe_dim", "--pe-dim",
        dest="pe_dim",
        type=int,
        default=128,
        help="Dimensionality of positional encodings.",
    )
    # add boolean argument for hpo
    parser.add_argument("--is_hpo", dest="is_hpo", action="store_true", help="Enable hyperparameter optimization")

    # LoRA-specific overrides (used when model_name == 'Lora')
    parser.add_argument(
        "--lora_r", "--lora-r",
        dest="lora_r",
        type=int,
        default=3,
        help="LoRA rank (r). Overrides the default only for LoRA models.",
    )
    parser.add_argument(
        "--lora_alpha", "--lora-alpha",
        dest="lora_alpha",
        type=float,
        default=0.42,
        help="LoRA alpha scaling factor. Overrides the default only for LoRA models.",
    )
    parser.add_argument(
        "--lora_target_modules", "--lora-target-modules",
        dest="lora_target_modules",
        nargs="+",
        default=["key", "value"],
        help="List of target module names to apply LoRA adapters to. Defaults to ['key', 'value'] for usage with ESM models. Use ['q', 'v'] for ProtT5.",
    )
    parser.add_argument(
        "--lora_dropout", "--lora-dropout",
        dest="lora_dropout",
        type=float,
        default=0.45,
        help="LoRA dropout probability. Overrides the default only for LoRA models.",
    )
    parser.add_argument(
        "--lora_learning_rate", "--lora-learning-rate",
        dest="lora_learning_rate",
        type=float,
        default=3.5e-4,
        help="Learning rate for LoRA TrainingArguments.",
    )
    parser.add_argument(
        "--lora_weight_decay", "--lora-weight-decay",
        dest="lora_weight_decay",
        type=float,
        default=5.3e-4,
        help="Weight decay for LoRA TrainingArguments.",
    )
    parser.add_argument(
        "--lora_per_device_bs",
        dest="lora_per_device_bs",
        type=int,
        default=2,
        help="Per-device batch size for LoRA training. Keep low (e.g. 1 or 2) for large models to avoid OOM.",
    )
    parser.add_argument(
        "--lora_num_train_epochs",
        dest="lora_num_train_epochs",
        type=int,
        default=10,
        help="Number of epochs for LoRA training.",
    )
    # New argument to optionally disable 2-stage training if desired
    parser.add_argument(
        "--use_two_stage_training",
        dest="use_two_stage_training",
        action="store_true",
        help="Enable two-stage training (freeze then unfreeze)",
    )
    
    parser.add_argument(
        "--keep_small_val_batches",
        dest="keep_small_val_batches",
        action="store_true",
        help="If set, validation DataLoader will not drop the last batch and will not shuffle, useful for very small validation sets."
    )
    parser.add_argument(
        "--run_tag",
        dest="run_tag",
        action="store",
        default="",
        help="Optional tag appended to run names and output paths. Useful for grouping random-search trials.",
    )
    args = parser.parse_args()

    # write commandline arguments to params dict
    params = vars(args)
    params["lm_provider"] = params.get("lm_provider", DEFAULT_LM_PROVIDER).lower()

    # Auto-adjust lora_target_modules for ProtT5 if user didn't explicitly override (heuristic)
    # Note: argparse defaults are applied before this, so we check if the user *didn't* change it from default
    # But since default is ["key", "value"], we can check if it matches that and if model is T5.
    if params["LM_name"] == "protT5_xl_uniref50":
         # If user hasn't provided custom targets (still defaults), switch to q,v
        if params["lora_target_modules"] == ["key", "value"]:
            params["lora_target_modules"] = ["q", "v"]
            # Also ensure provider/repo logic works if needed, but alias map handles repo.
            
    return params


class EsmWithPE(nn.Module):
    def __init__(self, base_model, pe_dim, num_labels=2, p_pe_drop=0.2):
        super().__init__()
        H = base_model.config.hidden_size
        self.base_model = base_model
        self.num_labels = num_labels

        self.norm = nn.LayerNorm(H)

        # --- PE → FiLM ---
        self.pe_dropout = nn.Dropout(p_pe_drop)
        self.pe_to_gamma = nn.Sequential(nn.Linear(pe_dim, H), nn.Tanh())
        self.pe_to_beta  = nn.Sequential(nn.Linear(pe_dim, H), nn.Tanh())
        self.gamma_dropout = nn.Dropout(0.1)
        self.beta_dropout  = nn.Dropout(0.1)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # --- head on pooled (B,H)
        self.classifier = nn.Sequential(
            nn.LayerNorm(H), nn.Dropout(0.2), nn.Linear(H, num_labels)
        )

        self.pe_dimension = pe_dim
        print(f"EsmWithPE: Initialized with PE dim={pe_dim}, base hidden size={H}")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        positional_encoding=None,
        labels=None,                # <-- keep labels so Trainer doesn’t drop them
        **kw,
    ):
        out = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            labels=None,            # don’t let base compute its own loss
            **kw,
        )

        hidden_state = getattr(out, "last_hidden_state", None)
        if hidden_state is None:
            hs = getattr(out, "hidden_states", None)
            if hs is not None:
                hidden_state = hs[-1]
        if hidden_state is None and isinstance(out, (tuple, list)) and len(out) > 0:
            hidden_state = out[0]
        if hidden_state is None:
            raise AttributeError("Model output lacks hidden states required for pooling.")

        # masked mean pooling
        if attention_mask is None:
            h = hidden_state.mean(dim=1)
        else:
            m = attention_mask.unsqueeze(-1).to(hidden_state.dtype)
            denom = m.sum(dim=1).clamp_min(1e-6)
            h = (hidden_state * m).sum(dim=1) / denom
        h = self.norm(h)

        # FiLM from PE (safe against NaNs/Inf in PE)
        if positional_encoding is not None and self.pe_dimension > 2:
            pe = torch.nan_to_num(positional_encoding, nan=0.0, posinf=0.0, neginf=0.0)
            pe = self.pe_dropout(pe)
            gamma = self.gamma_dropout(self.pe_to_gamma(pe))
            beta  = self.beta_dropout(self.pe_to_beta(pe))
            h = h * (1 + self.alpha * gamma) + self.alpha * beta

        logits = self.classifier(h)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=out.hidden_states if self.training else None,
            attentions=out.attentions if self.training else None,
        )

    # passthrough helpers
    def print_trainable_parameters(self, *args, **kwargs):
        if hasattr(self.base_model, "print_trainable_parameters"):
            return self.base_model.print_trainable_parameters(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self.base_model.save_pretrained(*args, **kwargs)


def resolve_lm_repo(params):
    override = (params.get("lm_repo") or "").strip()
    if override:
        return override

    provider = params.get("lm_provider", DEFAULT_LM_PROVIDER).lower()
    lm_name = params.get("LM_name", "")

    alias_map = LM_PROVIDER_ALIASES.get(provider, {})
    if lm_name in alias_map:
        return alias_map[lm_name]

    if provider == "facebook":
        return f"facebook/{lm_name}"

    return f"{provider}/{lm_name}"


def should_trust_remote_code(params, lm_repo):
    provider = (params.get("lm_provider") or DEFAULT_LM_PROVIDER).lower()
    if provider != "facebook":
        return True

    explicit_repo = (params.get("lm_repo") or "").strip()
    if explicit_repo:
        owner = explicit_repo.split("/")[0].lower()
        if owner != "facebook":
            return True

    repo_owner = (lm_repo or "").split("/")[0].lower()
    return repo_owner != "facebook"

def setupFolders(params):
    # Model directory (output)
    modelFolder = MODELS.joinpath(params["model_name"])
    modelFolder.mkdir(exist_ok=True, parents=True)  # create

    match params['LM_name']:
        case "esm1b_t33_650M_UR50S":
            LM_name_short = "ESM1b_650M"
        case "esm2_t6_8M_UR50D":
            LM_name_short = "ESM2_8M"
        case "esm2_t12_35M_UR50D":
            LM_name_short = "ESM2_35M"
        case "esm2_t30_150M_UR50D":
            LM_name_short = "ESM2_150M"
        case "esm2_t33_650M_UR50D":
            LM_name_short = "ESM2_650M"
        case "esm2_t36_3B_UR50D":
            LM_name_short = "ESM2_3B"
        case "esm2_t48_15B_UR50D":
            LM_name_short = "ESM2_15B"
        case "protT5_xl_uniref50":
            LM_name_short = "ProtT5_XL"
        case _:
            LM_name_short = "NA"

    effective_epochs = params["epochs"]
    effective_lr = params["lr"]
    if params.get("model_name") == "Lora":
        effective_epochs = params.get("lora_num_train_epochs", params["epochs"])
        effective_lr = params.get("lora_learning_rate", params["lr"])

    params['model_file_name'] = (
        f"{params['model_name']}_{LM_name_short}-E={params['embeddingSubfolder']}"
        f"-S={params['seed']}-E={effective_epochs}-BS={params['bs']}-LR={effective_lr:.6f}"
    )

    if params.get('model_name') == "Lora":
        def _sanitize(value):
            if isinstance(value, float):
                txt = f"{value:g}"
            else:
                txt = str(value)
            return txt.replace('.', 'p').replace('-', 'm')

        lora_parts = [
            f"a{_sanitize(params.get('lora_alpha', 'NA'))}",
            f"d{_sanitize(params.get('lora_dropout', 'NA'))}",
            f"lr{_sanitize(params.get('lora_learning_rate', 'NA'))}",
            f"r{_sanitize(params.get('lora_r', 'NA'))}",
            f"wd{_sanitize(params.get('lora_weight_decay', 'NA'))}",
            f"pe{_sanitize(params.get('pe_dim', 'NA'))}",
        ]
        target_modules = params.get('lora_target_modules')
        if target_modules:
            modules_sanitized = "_".join(_sanitize(m) for m in target_modules)
            lora_parts.append(f"tm{modules_sanitized}")
        params['model_file_name'] += "-" + "-".join(lora_parts)

    if params.get('model_name') == "FiLM_PE":
        pe_dim = params.get('pe_dim', 0)
        if pe_dim > 2:
            params['model_file_name'] += f"-PE={pe_dim}"
        else:
            params['model_file_name'] += "-PE=No"

    run_tag = _sanitize_for_path(params.get("run_tag", ""))
    params["run_tag"] = "" if run_tag == "run" and not params.get("run_tag", "") else run_tag
    if params["run_tag"]:
        params["model_file_name"] += f"--{params['run_tag']}"

    # Input folders
    embeddingFolder = EMBEDDINGS.joinpath(params["LM_name"]).joinpath(params["embeddingSubfolder"])
    dataSetPath = DATA_SETS.joinpath(params["data_set_name"])

    return modelFolder, embeddingFolder, dataSetPath


def _dataset_tag_from_params(params):
    raw_name = params.get("data_set_name") or ""
    if raw_name:
        stem = Path(raw_name).stem
        if stem.startswith("train_dataset_"):
            stem = stem[len("train_dataset_"):]
        tokens = [tok for tok in re.split(r"[^\w]+", stem) if tok]
        if tokens:
            head = tokens[0]
            if head.isalpha() and head.upper() == head:
                return head
            return head
    return params.get("embeddingSubfolder", "dataset")


def _lm_size_from_name(params):
    lm_name = params.get("LM_name") or ""
    match = re.search(r"(\d+(?:\.\d+)?[MB])", lm_name)
    if match:
        return match.group(1)
    return lm_name or "model"


def _has_pretrained_adapter(params, model_folder):
    lm_repo = params.get("LM_path", "")
    lm_suffix = lm_repo.replace("/", "__") if lm_repo else params.get("LM_name", "")
    seed_suffix = f"_seed{params['seed']}" if params.get("seed") is not None else ""
    dataset_hint = (params.get("data_set_name") or "")[:3] or "ds"
    adapter_dir = model_folder.joinpath(f"{params['model_name']}_{dataset_hint}_{lm_suffix}{seed_suffix}_adapters")
    return adapter_dir.exists() and adapter_dir.joinpath("adapter_config.json").exists()


def build_wandb_run_name(params, model_folder):
    dataset_tag = _dataset_tag_from_params(params)
    lm_size = _lm_size_from_name(params)
    pe_dim = params.get("pe_dim")

    parts = [p for p in (dataset_tag, lm_size) if p]
    parts.append(f"PE{pe_dim}" if pe_dim is not None else "PE?")

    dataset_name = params.get("data_set_name", "")
    if "fine-tuning" in dataset_name:
        pretrained_found = _has_pretrained_adapter(params, model_folder)
        params["pretrained_adapter_found"] = pretrained_found
        if not pretrained_found:
            parts.append("No_PT")

    return "_".join(parts)

from scripts.training.dataset import DataSet, DataSet_Residual, LoraDataset, DataSet_PE
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd


def getDataset(params, dataSetPath, embeddingFolder, is_positional_encoding_none=False):
    """
    Returns a dict with different key-value pairs based on what the models need/expect:
    - Torch models  (Peng, Peng6, Linear_pytorch): Datasets (train, val) and DataLoaders (train, val)
    - For Huggingface models Lora: {train_dataset, val_dataset, dataset, tokenizer}
    - For sklearn models: X_train, X_val, Y_train, Y_val (dataset is created only to compute class imbalance)
    """
    log(f"Setting up Dataset from {dataSetPath}")

    # --- Setup paths/flags ----------------------------------------------------
    cluster_path = REPOSITORY.joinpath("data")
    cluster_path.mkdir(parents=True, exist_ok=True)
    split_dir = cluster_path.joinpath("splits")
    split_dir.mkdir(parents=True, exist_ok=True)

    # Heuristic used previously (kept for compatibility)
    #is_pre_training = params['data_set_name'][-16:-4] == "pre-training"
    is_pre_training = "pre-training" in params['data_set_name']
    

    # Helper to attach groups to a dataset
    def attach_groups(_dataset):
        raw_data_path = DATA_RAW.joinpath(params['embeddingSubfolder'] + '.tsv')
        raw_df = pd.read_csv(raw_data_path, sep='\t')
        raw_df['cluster_number'] = raw_df['cluster_number'].fillna(-1).astype(int)
        gmap = raw_df[['Gene_ID','cluster_number']].set_index('Gene_ID').to_dict()['cluster_number']

        df = _dataset.dataSet_df
        df['cluster_number'] = df['Gene_ID'].map(gmap)
        df['__groups__'] = np.where(
            df['cluster_number'] != -1,
            df['cluster_number'].astype(str),
            "NA_" + df['Gene_ID'].astype(str)
        )
        return df['__groups__'].values

    # Builders for datasets/tokenizer (so we can reuse in both phases)
    tokenizer = None
    def build_dataset_and_tokenizer(path):
        nonlocal tokenizer
        if params["model_name"] in ["Peng", "Peng6", "Linear_pytorch", "Linear", "RandomForest", "XGBoost", "Random_SK", "Random"]:
            return DataSet_Residual(path, embeddingFolder)
        elif params["model_name"] == "FiLM_PE":
            return DataSet_PE(path, embeddingFolder)
        elif params["model_name"] == "Lora":
            if tokenizer is None:
                from transformers import AutoTokenizer, T5Tokenizer
                lm_repo = resolve_lm_repo(params)
                params["LM_path"] = lm_repo
                params["lm_repo"] = lm_repo
                
                if "prot_t5" in lm_repo.lower() or "prott5" in params["LM_name"].lower():
                    tokenizer = T5Tokenizer.from_pretrained(lm_repo, do_lower_case=False)
                else:
                    tokenizer = AutoTokenizer.from_pretrained(lm_repo, trust_remote_code=False)
            return LoraDataset(path, tokenizer, embedding_folder=embeddingFolder, max_length=1500)
        else:
            raise NotImplementedError(f"Model \"{params['model_name']}\" not implemented for dataset setup")

    # -------------------------------------------------------------------------
    # PRE-TRAINING PHASE
    # -------------------------------------------------------------------------
    if is_pre_training:
        # Build pre-training dataset
        dataset = build_dataset_and_tokenizer(dataSetPath)

        # Build fine-tuning dataset path + dataset
        dataSetPath_ft = dataSetPath.parent.joinpath(
            dataSetPath.stem.replace("pre-training", "fine-tuning") + dataSetPath.suffix
        )
        dataset_ft = build_dataset_and_tokenizer(dataSetPath_ft)

        # 1) Pre-training split (group-based)
        groups = attach_groups(dataset)
        all_idx = np.arange(len(dataset.dataSet_df))
        gss = GroupShuffleSplit(n_splits=1, test_size=params["split"][1], random_state=params["seed"])
        train_idx, val_idx = next(gss.split(all_idx, groups=groups))

        log(f"Positive ratio full: {dataset.dataSet_df['positive'].mean():.4f} | "
            f"train: {dataset.dataSet_df.iloc[train_idx]['positive'].mean():.4f} | "
            f"val: {dataset.dataSet_df.iloc[val_idx]['positive'].mean():.4f}")

        # Sanity checks
        overlap = set(groups[train_idx]) & set(groups[val_idx])
        assert len(overlap) == 0, f"Pre-train group overlap train/val: {overlap}"
        log(f"N_train={len(train_idx)}, N_val={len(val_idx)}")

        train_dataset = torch.utils.data.Subset(dataset, indices=train_idx.tolist())
        val_dataset   = torch.utils.data.Subset(dataset, indices=val_idx.tolist())

        # 2) Fine-tuning split prepared now: FT test groups must exclude pre-train TRAIN groups
        ft_groups = attach_groups(dataset_ft)
        pretrain_train_groups = set(groups[train_idx])

        # Eligible FT test groups
        eligible_ft_test_groups = set(ft_groups) - pretrain_train_groups
        if len(eligible_ft_test_groups) == 0:
            raise RuntimeError("No eligible FT test groups after excluding pre-train TRAIN groups.")

        ft_all_idx = np.arange(len(dataset_ft.dataSet_df))
        eligible_mask = np.isin(ft_groups, list(eligible_ft_test_groups))
        ft_candidate_idx = ft_all_idx[eligible_mask]

        if len(ft_candidate_idx) == 0:
            raise RuntimeError("Eligible FT candidate indices empty; cannot form FT test split.")

        # If too few candidates to satisfy test fraction, reduce test_size gracefully
        requested_test_size = params["split"][1]
        # approximate by sample count; we split by groups but this guard helps avoid over-filtering surprises
        min_needed = max(1, int(np.ceil(requested_test_size * len(ft_candidate_idx))))
        if len(ft_candidate_idx) < min_needed:
            log(f"FT: reducing test fraction since only {len(ft_candidate_idx)} eligible samples "
                f"after excluding pre-train TRAIN groups.")
            # fall back to using all eligible as test, leaving the rest as train
            ft_val_idx = ft_candidate_idx
            ft_val_groups = set(ft_groups[ft_val_idx])
            ft_train_idx = ft_all_idx[~np.isin(ft_groups, list(ft_val_groups))]
        else:
            # Group split **within the eligible subset**
            ft_groups_subset = ft_groups[ft_candidate_idx]
            ft_gss = GroupShuffleSplit(n_splits=1, test_size=requested_test_size, random_state=params["seed"])
            ft_train_sub, ft_val_sub = next(ft_gss.split(np.arange(len(ft_candidate_idx)), groups=ft_groups_subset))
            # Map subset-local indices back to GLOBAL indices
            ft_val_idx = ft_candidate_idx[ft_val_sub]
            ft_val_groups = set(ft_groups[ft_val_idx])
            ft_train_idx = ft_all_idx[~np.isin(ft_groups, list(ft_val_groups))]

        # Final FT sanity
        assert len(set(ft_groups[ft_train_idx]) & set(ft_groups[ft_val_idx])) == 0, "FT train/val group overlap."
        assert len(set(ft_groups[ft_val_idx]) & pretrain_train_groups) == 0, "FT val/test intersects pre-train TRAIN groups."

        # Save FT indices for later loading
        ft_prefix = f"{params['data_set_name'].replace('pre-training','fine-tuning')[:-4]}_{params['model_name'].lower()}_seed_{params['seed']}_{params['LM_name']}"

        ft_train_file = split_dir.joinpath(f"{ft_prefix}__ft_train.tsv")
        ft_val_file   = split_dir.joinpath(f"{ft_prefix}__ft_val.tsv")

        def _save_indices(out_path, idx_array, ds):
            df = ds.dataSet_df.iloc[idx_array][['Gene_ID', '__groups__']].copy()
            df.insert(0, 'index', idx_array)
            df.to_csv(out_path, sep='\t', index=False)
        # if the file already exists, do not overwrite
        if ft_train_file.exists() and ft_val_file.exists():
            log(f"FT split files already exist, not overwriting:\n  {ft_train_file}\n  {ft_val_file}")
        else:
            _save_indices(ft_train_file, ft_train_idx, dataset_ft)
            _save_indices(ft_val_file,   ft_val_idx,   dataset_ft)
            log(f"Saved FT splits:\n  train -> {ft_train_file}\n  val   -> {ft_val_file}")

    # -------------------------------------------------------------------------
    # FINE-TUNING PHASE
    # -------------------------------------------------------------------------
    else:
        # Build FT dataset from the *fine-tuning* file (dataSetPath points to FT in this phase)

        dataset = build_dataset_and_tokenizer(dataSetPath)  # this is FT dataset in this branch
        _ = attach_groups(dataset)  # ensure '__groups__' present for consistency

        ft_prefix = f"{params['data_set_name'].replace('pre-training','fine-tuning')[:-4]}_{params['model_name'].lower()}_seed_{params['seed']}_{params['LM_name']}"

        ft_train_file = split_dir.joinpath(f"{ft_prefix}__ft_train.tsv")
        ft_val_file   = split_dir.joinpath(f"{ft_prefix}__ft_val.tsv")

        if not (ft_train_file.exists() and ft_val_file.exists()):
            raise RuntimeError(f"FT split files not found:\n  {ft_train_file}\n  {ft_val_file}\nRun pre-training to generate them.")
        log(f"Loading FT splits from:\n  {ft_train_file}\n  {ft_val_file}")
        log(f"FT dataset size: {len(dataset)} samples")
        log(f"Positive ratio FT dataset: {dataset.dataSet_df['positive'].mean():.4f}")
        ft_train_idx = pd.read_csv(ft_train_file, sep='\t')['index'].to_numpy()
        ft_val_idx   = pd.read_csv(ft_val_file,   sep='\t')['index'].to_numpy()
        
        # --- Attach positional encoding only for LoRA fine-tuning ---------------------
        
        if params["model_name"] in ["Lora", "FiLM_PE"]:
            from scripts.data_sets.positional_encoding_processing import get_posenc_pkg, build_pe_matrix_for_dataset
            pkg = get_posenc_pkg(datafile=str(dataSetPath), pca_n_components=params["pe_dim"]) # cached by (data set + dim)
            pe_full = build_pe_matrix_for_dataset(dataset, pkg, use_pca=True)
            if not is_positional_encoding_none:
                dataset.set_positional_encodings(pe_full)
            log(f"Positional encoding matrix shape: {pe_full.shape}")

        train_dataset = torch.utils.data.Subset(dataset, indices=ft_train_idx.tolist())
        val_dataset   = torch.utils.data.Subset(dataset, indices=ft_val_idx.tolist())

    # Class imbalance
    params["crit_weight"] = computeClassWeight(train_dataset) #compute class weight for later use in BCELoss  
    log(f"Class weights: {params['crit_weight']}")
    
    # Differnt output format based on model type
    if params["model_name"] in ["Linear", "RandomForest", "XGBoost", "Random_SK" ]: #sklearn models: will need all data in X and Y array.
        #Make training lists
        X_train = []
        Y_train = []
        for Y, X, idx in train_dataset:
            X_train.append(X)
            Y_train.append(Y)
        #Make validation lists
        X_val = []
        Y_val = []
        for Y, X, idx in val_dataset:
            X_val.append(X)
            Y_val.append(Y)

        return {"X_train":X_train, "X_val":X_val, "Y_train":Y_train, "Y_val":Y_val}
    
    elif params["model_name"] in ["Peng","Peng6","Linear_pytorch", "Random", "FiLM_PE"]: # Lora model: only #TODO: Random too
            
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=params["bs"],
            drop_last=True, #otherwise we might train on a batch with only 2 elements
            shuffle=True, #reshuffle training data each epoch
            num_workers=params["num_workers"],
            prefetch_factor=params["prefetch_factor"],
            persistent_workers=params["persistent_workers"],
            pin_memory=params["pin_memory"],
        )
        val_bs = max(15 , params["bs"] // 15) # smaller val batch size to drop less incomplete batches during eval
        
        # Check if we should keep small batches (for small validation sets)
        drop_last_val = True
        shuffle_val = True
        if params.get("keep_small_val_batches", False):
            drop_last_val = False
            shuffle_val = False
            
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_bs,
            drop_last=drop_last_val, #otherwise we might evaluate on a batch with only 2 elements
            shuffle=shuffle_val, #reshuffle validation data each epoch
            num_workers=params["num_workers"],
            prefetch_factor=params["prefetch_factor"],
            persistent_workers=params["persistent_workers"],
            #pin_memory=params["pin_memory"], #NOTE: causes an unidentified error on the cuda devise for some reason!
        )
        return {"train_dataset": train_dataset, "val_dataset": val_dataset, "train_loader": train_loader,   "val_loader": val_loader}
    
    elif params["model_name"] in ["Lora"]: # Lora model: needs tokenizer but not dataloaders 
        return {"train_dataset": train_dataset, "val_dataset": val_dataset, "dataset": dataset, "tokenizer": tokenizer}
    else:
        raise NotImplementedError(f"Model \"{params['model_name']}\" not implemented for dataset setup")

from scripts.embeddings.utils import pretrainedModelInfos #other module/folder

def getModelFromCkpt(params):
    """
    Load model from checkpoint file
    """
    assert "model_name" in params.keys(), "params dict does not contain key 'model_name'"
    assert "checkpoint_path" in params.keys(), "params dict does not contain key 'checkpoint_path'"
    assert params["checkpoint_path"].exists() == True, f"Checkpoint does not exist: \"{params['checkpoint_path']}\""
    modelName = params["model_name"]
    ckptPath = params["checkpoint_path"]

    #Model case destinction
    if modelName in ["Peng", "Peng6"]: # Peng style models
        from scripts.models.Peng import peng_parametrized
        model = peng_parametrized.load_from_checkpoint(ckptPath)
        params["device"] = torch.device(f"cuda:{params['devices'][0]}" if params['devices'][0] >=0 else "cpu")
        print(f"DEVICE: repr({params['device']})")
        model.to(params["device"])
    elif modelName == "FiLM_PE":
        from scripts.models.FiLM_PE import FiLM_PE
        model = FiLM_PE.load_from_checkpoint(ckptPath)
        params["device"] = torch.device(f"cuda:{params['devices'][0]}" if params['devices'][0] >=0 else "cpu")
        model.to(params["device"])
    elif modelName in ["Random"]:
        raise NotImplementedError(f"Loading model \"{modelName}\" needs refactoring/adaption!")
        # I simply did not bother yet...
    elif modelName in ["Linear_pytorch"]:
        raise NotImplementedError(f"Loading model \"{modelName}\" needs refactoring/adaption!")
        #TODO: implement proper saving and loading of linear pytorch models

        #OPTION 1: 
        #from scripts.models.Linear_pytorch import linearClassififer
        #model =  linearClassififer.load_from_checkpoint(ckptPath, map_location=lambda device, loc: device)
        # the above line does not work as it would require "params" and "embedding_dim" for initialization

        #OPTION 2: 
        # model = torch.load(ckptPath, map_location=lambda device, loc: device) #NOTE: this call will prodive a dict (not having params and embedding_dim inside)
    elif modelName in ["Linear", "RandomForest", "XGBoost", "Random_SK"]:
        import pickle
        with open(ckptPath, 'rb') as f:
            model = pickle.load(f)
    else:
        raise NotImplementedError(f"Model \"{modelName}\" not implemented for loading from checkpoint")
    #checkpoint = torch.load(ckptPath, map_location=lambda storage, loc: storage) #TODO: how is that different?
    return model

def getLatestModel(params, modelFolder):

    #Load pytorch checkpoint if possible
    if  params["checkpoint_folder"] != "":
        instanceFolder = modelFolder.joinpath("lightning_logs").joinpath(params["checkpoint_folder"])
        #params['model_file_name'] += f"-ckpt={ckptName}"
        params['model_file_name'] = f"{instanceFolder.name}__{params['model_file_name']}"

        log(f"Loading {instanceFolder}",indentation=1)
        checkpointFolder = instanceFolder.joinpath("checkpoints")
        if( not checkpointFolder.exists() == True):
            raise RuntimeError(f"No ckpt folder {checkpointFolder}")

        ckptName = natsorted( [path.name for path in checkpointFolder.iterdir()] )[-1] #get last checkpoint
        log(f"\tLatest checkpoint: {ckptName}",indentation=2)

        params['checkpoint_path'] = checkpointFolder.joinpath(ckptName) # Create NEW key value pair
        if params["checkpoint_path"].exists() == False:
            raise RuntimeError(f'Checkpoint does not exist: \"{params["checkpoint_path"]}\"')

        #Load model
        model = getModelFromCkpt(params)

        log(f"\tLoaded model from {params['checkpoint_path']}")
        log(f"\t\tLoaded Params:")
        for key in model.params.keys():
            log(f"\t\t\t{key}: {model.params[key]}")

        #Update Hyperparameters
        # - max_epoch
        params["epochs"] += model.params["epochs"] # e.g. 150 for pre-traing + 100 for fine tune = 250 max
        model.params = params #set model parameters

        return model

    else: # Create new instance

        if params["model_name"] in ["Peng", "Peng6"]: # Peng style models
            from scripts.models.Peng import peng_parametrized
            layers, million_params, used_dataset, embedding_dim = pretrainedModelInfos[params["LM_name"]] # get general parameters/infos
            # Adapt Peng model size based on the embedding size
            scalingFactor = embedding_dim / 1024.0  # size in comparison to original Peng approach
            hiddem_dim = int(scalingFactor * 320)
            embedding_dim = int(scalingFactor * 1024)
            num_GRU_layers = 6 if params["model_name"] == "Peng6" else int(scalingFactor * 6) #THIS IS THE ONLY DIFFERENCE
            log(f"Creating new peng_parametrized with scalingFactor {scalingFactor} ({embedding_dim}/1024):")
            log(f"\tembedding_dim={embedding_dim}")
            log(f"\thiddem_dim={hiddem_dim}")
            log(f"\tnum_GRU_layers={num_GRU_layers}")
            return peng_parametrized(params, hiddem_dim=hiddem_dim, embedding_dim=embedding_dim, num_GRU_layers=num_GRU_layers, L=1)
        
        elif params["model_name"] == "FiLM_PE":
            from scripts.models.FiLM_PE import FiLM_PE
            layers, million_params, used_dataset, embedding_dim = pretrainedModelInfos[params["LM_name"]]
            return FiLM_PE(params, embedding_dim=embedding_dim, pe_dim=params["pe_dim"])

        elif params["model_name"] == "Linear_pytorch": # Linear pytorch model (1 FC layer)
            layers, million_params, used_dataset, embedding_dim = pretrainedModelInfos[params["LM_name"]] # get general parameters/infos
            from scripts.models.Linear_pytorch import linearClassififer
            return linearClassififer(params, embedding_dim)
            
        elif params["model_name"] == "Linear": # Linear Model (Logistic regression sklearn)
            from sklearn.linear_model import LogisticRegression
            class_weight = {0: params["crit_weight"][0].item(), 1: params["crit_weight"][1].item()}
            return LogisticRegression(
                penalty="l2", #default: l2
                solver="lbfgs", #default: lbfgs
                C=1.0, #default: 1.0
                max_iter=1000, #default: 100
                class_weight=class_weight,
                random_state=params["seed"],
                n_jobs=params["threads"]
                )

        elif params["model_name"] == "RandomForest": # Ranom Forest
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(
                max_depth=20, # default: 2
                n_estimators=100, #default: 100 
                class_weight={0:params["crit_weight"][0], 1:params["crit_weight"][1]},
                random_state=params["seed"],
                n_jobs=params["threads"]
                )
            
            # current best: max depth=20 -> 0.662 0.351 0.983 0.931 
        
        elif params["model_name"] == "XGBoost": # XGBoost (gradient boosting)
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(
                n_estimators=100, #default: 100 -> makes no difference
                learning_rate=params["lr"], #default: 0.1
                max_depth=15, #default: 3
                random_state=params["seed"])
        
            #current best: max_depth=15, lr=0.5 -> 0.6801, 0.3679, 0.9788 , 0.9373

        elif params["model_name"] == "Random_SK": # Random 8SKlearn) Model (random predictions)
            from sklearn.dummy import DummyClassifier
            return DummyClassifier(
                #strategy="stratified", # respect class imbalance?
                strategy="uniform", # random predictions
                random_state=params["seed"]
                )

        elif params["model_name"] == "Random": # Random Model (random predictions)
            useOwn = True

            from scripts.models.Random import RandomClassifier
            return RandomClassifier(
                params=params #training dataset with GT labels
            )                

        elif params["model_name"] == "Lora": # Lora Model (having 1 FC)
            from transformers import AutoModelForSequenceClassification
            #from accelerate import Accelerator #TODO: maybe later if we have multi-GPU scenario

            # Define labels and model
            id2label = {0: "Not RNA binding", 1: "RNA binding"}
            label2id = {v: k for k, v in id2label.items()}
            lm_repo = resolve_lm_repo(params)
            trust_remote = should_trust_remote_code(params, lm_repo)
            params["LM_path"] = lm_repo
            params["lm_repo"] = lm_repo
            params["trust_remote_code"] = trust_remote

            
            if "prot_t5" in lm_repo.lower() or "prott5" in params["LM_name"].lower():
                 # T5 specific loading
                 from transformers import T5EncoderModel, T5Tokenizer
                 
                 class T5ForSequenceClassification(nn.Module):
                    def __init__(self, model_name, num_labels=2, trust_remote_code=False):
                        super().__init__()
                        self.encoder = T5EncoderModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
                        self.config = self.encoder.config
                        self.num_labels = num_labels
                        self.hidden_size = self.config.d_model
                        self.classifier = nn.Sequential(
                            nn.LayerNorm(self.hidden_size),
                            nn.Dropout(0.1),
                            nn.Linear(self.hidden_size, num_labels)
                        )
                        
                    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
                        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
                        last_hidden_state = outputs.last_hidden_state
                        
                        # Mean pooling
                        if attention_mask is not None:
                            mask = attention_mask.unsqueeze(-1).float()
                            sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
                            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
                            mean_embeddings = sum_embeddings / sum_mask
                        else:
                            mean_embeddings = last_hidden_state.mean(dim=1)
                            
                        logits = self.classifier(mean_embeddings)
                        
                        loss = None
                        if labels is not None:
                            loss_fct = nn.CrossEntropyLoss()
                            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                            
                        # Mimic HF output
                        return SequenceClassifierOutput(
                            loss=loss,
                            logits=logits,
                            hidden_states=outputs.hidden_states,
                            attentions=outputs.attentions,
                        )
                    
                    def save_pretrained(self, save_directory):
                        # Save encoder and potentially classifier (though PEFT handles adapter saving separately)
                        self.encoder.save_pretrained(save_directory)
                        # TODO: Save classifier weights if not standard? 
                        # For LoRA, we only save adapters. If we fine-tune head, we might need to save it. 
                        # But PEFT usually handles saving trainable params if configured.
                        
                    def print_trainable_parameters(self):
                         # Helper for debugging
                         pass

                 model = T5ForSequenceClassification(lm_repo, num_labels=len(id2label), trust_remote_code=trust_remote)

            else:
                model = AutoModelForSequenceClassification.from_pretrained(
                    lm_repo,
                    trust_remote_code=trust_remote,
                    num_labels=len(id2label),
                    id2label=id2label,
                    label2id=label2id,
                )

            return model

        else:  # default/fallthrough
            raise RuntimeError(f"Unknown model \"{params['model_name']}\" for setup")

from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
def setupLoggers(params, modelFolder):
    # Create TensorBoardLogger
    logger_TB = TensorBoardLogger(
        save_dir=modelFolder,
        #name=params['model_file_name'],
        version=params['model_file_name'],
        default_hp_metric=False)
    log(f"Getting model {params['model_file_name']}")

    # Creat Weights and Biases Logger
    log(f"Setup Weights and Biases")
    wandb_run_name = params['model_file_name']
    # Prepend dataset tag to disambiguate runs across different datasets
    # (e.g. RIC vs bressin19 both use embeddingSubfolder=RIC, so model_file_name alone is ambiguous)
    dataset_tag = _dataset_tag_from_params(params)
    if dataset_tag and not wandb_run_name.startswith(dataset_tag):
        wandb_run_name = f"{dataset_tag}__{wandb_run_name}"
    if params.get("model_name") == "Lora":
        wandb_run_name = build_wandb_run_name(params, modelFolder)
    params["wandb_run_name"] = wandb_run_name
    logger_WB = WandbLogger( # initialise the wandb logger and name your wandb project
        project='predict-rbp',
        name=wandb_run_name,
        config=params,
        )
    # Force run initialisation so the custom name sticks even if wandb gets touched elsewhere first
    logger_WB.experiment.name = wandb_run_name

    #logger_WB.experiment.config["batch_size"] = params["bs"] #one parameter
    #logger_WB.experiment.config.update(params) #all parameters (dict
    #logger_WB.log_hyperparams(params) #all parameters (dict)

    return logger_TB, logger_WB

#compute balance
def computeClassWeight(dataset):
    """
    Computes weights of binary classes based on their ratio. [0] = neg, [1] = pos
    If one class occurs X times more than the other, it will be X times less important for the loss computation.
    E.g. if the ratio is 2:1, the weights are 0.75, 1.5 (normalized: 1, 2)
    if the ratio is 10:1, weights = (0.55, 5.55) (normalized: 1, 10)
    """
    
    #potentially relevant for unbalanced training (e.g. Peng)
    positives = 0
    #sum(dataset["positive"] == True)
    negatives = 0
    #sum(self.dataSet_df["positive"] == False)

    #print(dir(dataset))
    
    for sample_id in dataset.indices:
        #sample = dataset.dataset.__getitem__(sample_id, includeEmbedding=False)
        #if sample["positive"]:
        #    positives+=+1
        #else:
        #    negatives-=-1

        #check of which type the dataset is
        if isinstance(dataset.dataset, LoraDataset): # Lora Dataset has differet return value/signature
            dict_entry = dataset.dataset.__getitem__(sample_id)
            pos = dict_entry['labels'].item()
        else: # if its a "normal" dataset
            pos, idx = dataset.dataset.__getitem__(sample_id, includeEmbedding=False)
            
        if pos >0:
            positives+=+1
        else:
            negatives-=-1

    try:
        loss_weight_pos = (positives+negatives) / (2.0 * positives) 
        loss_weight_neg = (positives+negatives) / (2.0 * negatives) 
    except ZeroDivisionError as e:
        #print(f"P={positives}, N={negatives}, total={len(dataset)}")
        raise e
    
    crit_weight= torch.tensor([loss_weight_neg,loss_weight_pos])

    return crit_weight

#Define the custom Trainer class for imbalance datasets
from transformers import Trainer, TrainerCallback
import torch.distributed as dist
class WeightedLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        logits = outputs.get('logits')
        labels = inputs.get('labels')
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
        # When running under DDP, each process/GPU should see only a shard of the dataset.
        # DistributedSampler takes care of:
        #   1) splitting the dataset evenly across all ranks (so you don't train on the same samples)
        #   2) shuffling in sync across processes each epoch
        
        use_ddp = dist.is_available() and dist.is_initialized()
        if use_ddp:
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
            shuffle = False
        else:
            sampler = None  # or RandomSampler(self.train_dataset)
            shuffle = True

        # Build the DataLoader just as HF Trainer would, but with our sampler plugged in.
        return DataLoader(
            self.train_dataset,
            sampler=sampler,                
            shuffle=shuffle,               # <-- shuffle only if not using DistributedSampler to get RandomSampler
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,   # ← use correct attr
        )

    def get_eval_dataloader(self, eval_dataset=None):
        # Evaluation should also be sharded so each GPU only computes on its slice,
        # then results are gathered. We set shuffle=False to keep eval order deterministic.
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        
        use_ddp = dist.is_available() and dist.is_initialized()
        if use_ddp:
            sampler = DistributedSampler(eval_dataset, shuffle=False)
            shuffle = False
        else:
            sampler = None  # or RandomSampler(self.train_dataset)
            shuffle = False

        return DataLoader(
            eval_dataset,
            sampler=sampler,   # <-- shard eval data
            shuffle=shuffle,               
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,   # ← use correct attr
        )
        
        

def freeze_lora_layers(model):
    for name, param in model.named_parameters():
        #if 'lora' in name and param.requires_grad: #and (freezing_counter - 1) <= number_of_layers_to_freeze:
        #    #if str(freezing_counter - 1) in name:
        #    param.requires_grad = False
        #    print(f"freezing {name}")
        if 'classifier' not in name and param.requires_grad:
            param.requires_grad = False
            print(f"freezing {name}")
        if param.requires_grad:
            print(f"leaving {name} still trainable")
            
def defreeze_lora_layers(model):
    for name, param in model.named_parameters():
        if 'lora_' in name and not param.requires_grad:
        
            param.requires_grad = True
            print(f"defreezing {name}")

def create_trainer(model, training_args, datasets, tokenizer, compute_metrics):
    return WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train_dataset"],
        eval_dataset=datasets["val_dataset"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

def train_with_progressive_freezing(
    model,
    datasets,
    tokenizer,
    compute_metrics,
    training_args,
    params,
    num_freezing_stages=1,
    epochs_per_stage=3,
):

    torch.cuda.empty_cache()

    if isinstance(epochs_per_stage, (list, tuple)):
        stage_epochs = [max(1, int(ep)) for ep in epochs_per_stage]
        stage_count = len(stage_epochs)
    else:
        stage_count = max(1, int(num_freezing_stages) + 1)
        stage_epochs = [max(1, int(epochs_per_stage))] * stage_count

    if not params.get("use_two_stage_training", False):
        stage_count = 1
        stage_epochs = [sum(stage_epochs)]

    for stage in range(stage_count):
        print(f"Starting training stage {stage}")
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Update batch size and create new training arguments
        desired_train_bs = params["bs"] # Use global batch size from arguments
        eval_batch_size = max(15, desired_train_bs // 15)
        print(f"Desired global batch size: {desired_train_bs}")
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        per_device_bs = params.get("lora_per_device_bs", 2) # Use configured per-device batch size
        grad_accum = max(1, math.ceil(desired_train_bs / (per_device_bs * world_size)))

        print(f"Current Train batch size: {desired_train_bs} | per_device_bs: {per_device_bs} | grad_accum: {grad_accum} | world_size: {world_size} | eval_bs: {eval_batch_size}")
        #training_args.output_dir = f"{params['model_name']}-lora-binding-sites_{timestamp}_stage_{stage}"
        training_args.per_device_train_batch_size = per_device_bs
        training_args.per_device_eval_batch_size = eval_batch_size
        training_args.gradient_accumulation_steps = grad_accum
        training_args.num_train_epochs = stage_epochs[stage]

        # Unfreeze LoRA layers if not the first stage
        if stage == 0:
            for n, p in model.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
        else:
            freeze_lora_layers(model)

        # Create a new trainer
        print(f"Creating trainer for stage {stage}")
        trainer = create_trainer(model, training_args, datasets, tokenizer, compute_metrics)
        
        if isinstance(compute_metrics, SavePredictionsMetricsWrapper):
            compute_metrics.trainer = trainer

        #trainer.class_weights = computeClassWeight(datasets["train_dataset"])
        # log trainer params
        log_trainer_params(trainer, params, training_args)
        # Train for this stage
        trainer.train()
        
        if hasattr(trainer, "optimizer"):
            tot = sum(p.numel() for g in trainer.optimizer.param_groups for p in g["params"])
            print(f"[OPT] total params in optimizer: {tot}")

        # Clear GPU memory
        torch.cuda.empty_cache()
    
    return trainer, timestamp


def log_trainer_params(trainer, params, training_args):
    # modify the number of training epochs based on the number of layers we want to freeze during training
    print(f"Number of training epochs: {trainer.args.num_train_epochs}")

    # Set class imbalance
    # reverse crit weight to get the correct class weight
    params["crit_weight"] = torch.tensor([params["crit_weight"][0], params["crit_weight"][1]]) 
    trainer.class_weights = params["crit_weight"].to(device=trainer.args.device)
    #print training_args.__dict__
    print(f"crit_weight {params['crit_weight']}")
    print(f"training_args: {training_args}")
    

import wandb
import numbers
class WandbCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs, **kwargs):


        # Filter out unwanted keys or add custom keys
        eval_logs = {k.replace("eval_", "val_"): v for k, v in logs.items() if "eval_" in k and "prc" not in k and "roc" not in k}
        train_logs = {k:v for k,v in logs.items() if "train_" in k}
        if len(eval_logs) > 0:
            wandb.log(eval_logs)
        if len(train_logs) > 0:
            wandb.log(train_logs)
    def on_save(self, args, state, control, **kwargs):
        # Log the directory path that is being added, needed to save checkpoints in the right path during optuna HPO
        print("Attempting to add directory:", args.output_dir)
        super().on_save(args, state, control, **kwargs)


# define the hyperparameter search space and the objective function to use for optuna hyperparameter optimization 
# define the hyperparameter search space and the objective function to use for optuna hyperparameter optimization 
def hp_space(trial, lm_name=None):
    
    # Determine target modules based on model type
    if lm_name and ("prot_t5" in lm_name.lower() or "prott5" in lm_name.lower()):
        # T5 uses q, k, v, o. Common LoRA targets are q, v.
        target_choices = (("q", "v"), ("q", "k", "v"), ("q",), ("v",))
    else:
        # Default (ESM)
        target_choices = (("query", "key", "value"), ("query","key"), ("query", "value"), ("key", "value"))

    return {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True),
        #'lr_scheduler_type': trial.suggest_categorical('lr_scheduler_type', ['linear', 'cosine']),
        #'per_device_train_batch_size': trial.suggest_int('per_device_train_batch_size', 6,24),
        #'warmup_steps': trial.suggest_int('warmup_steps', 0, 3),
        #'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.3),
        #'num_train_epochs': trial.suggest_int('num_train_epochs', 2, 4),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3),
        'pe_dim': trial.suggest_categorical('pe_dim', [128, 256, 512]), # 0 means no positional encoding
        "r": trial.suggest_int("r", 2, 5),  
        "lora_alpha": trial.suggest_float("lora_alpha", 0.1, 5.0, log=True),  # Log scale for wide range search
        "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.5),  
        "bias": "none", #trial.suggest_categorical("bias", ["none", "all", "lora_only"]),
        "target_modules": trial.suggest_categorical("target_modules", target_choices)#["dense_h_to_4h"], ["dense_4h_to_h"]]),
    }

def compute_objective(metrics):
    return metrics['eval_AUPRC']


def print_adapter_config(adapter_dir, model):
    import json
    from pathlib import Path
    from torch import Tensor

    print("\n[SANITY] Saved adapter_config.json:")
    with open(Path(adapter_dir) / "adapter_config.json") as f:
        saved_cfg = json.load(f)
    print(json.dumps(saved_cfg, indent=2))

    live = getattr(model, "peft_config", {}).get("default", None)
    # extract only fields we care about and coerce to JSON-able types
    if live is not None:
        live_cfg = {}
        for k in ["r", "lora_alpha", "lora_dropout", "target_modules", "bias", "task_type", "inference_mode"]:
            v = getattr(live, k, None)
            if isinstance(v, (set, tuple)):
                v = list(v)
            elif isinstance(v, Tensor):
                v = v.detach().cpu().tolist()
            live_cfg[k] = v
    else:
        live_cfg = None

    print("\n[SANITY] Live model peft_config['default']:")
    print(json.dumps(live_cfg, indent=2))

def fingerprint_split_AB(model):
    A_tot = A_nz = 0
    B_tot = B_nz = 0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_A" in n:
                A_tot += p.numel(); A_nz += (p != 0).sum().item()
            if "lora_B" in n:
                B_tot += p.numel(); B_nz += (p != 0).sum().item()
    return dict(A_tot=A_tot, A_nz=A_nz, B_tot=B_tot, B_nz=B_nz)

def compare_saved_vs_live_keys(adapter_dir, model):
    saved = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    saved_keys = sorted(saved.keys())
    live_keys = sorted([n for n,_ in model.named_parameters() if "lora_" in n])
    missing = [k for k in saved_keys if k not in live_keys]
    extra   = [k for k in live_keys if k not in saved_keys]
    print(f"\n[SANITY] saved lora tensors: {len(saved_keys)} | live lora tensors: {len(live_keys)}")
    print(f"[SANITY] missing (in live, from saved): {len(missing)}")
    if missing: print("  e.g.", missing[:5])
    print(f"[SANITY] extra (in live, not in saved): {len(extra)}")
    if extra: print("  e.g.", extra[:5])

# ---- tiny helper: LoRA weight fingerprint so we can prove load vs scratch ----
def lora_fingerprint(model):
    import torch, hashlib, json
    tot = nz = 0
    l2 = 0.0
    names = []
    with torch.no_grad():
        for n, p in model.named_parameters():
            # PEFT consistently prefixes LoRA params with "lora_" in the name
            if "lora_" in n:
                tot += p.numel()
                nz  += (p != 0).sum().item()
                l2  += (p.float()**2).sum().item()
                names.append(n)
    digest = hashlib.sha1(",".join(sorted(names)).encode()).hexdigest()[:12]
    return {"digest": digest, "total": tot, "nonzero": nz, "l2_sum": l2}



def _collect_esm_head_state(model):
    state = {}
    for name, tensor in model.state_dict().items():
        if not name.startswith("base_model."):
            state[name] = tensor.detach().cpu()
    return state


def _make_serializable_metrics(metrics):
    if not metrics:
        return {}
    out = {}
    for key, value in metrics.items():
        out[key] = _ensure_float(value)
    return out


def _build_ft_split_paths(params):
    ft_prefix = f"{params['data_set_name'].replace('pre-training','fine-tuning')[:-4]}_{params['model_name'].lower()}_seed_{params['seed']}_{params['LM_name']}"
    split_dir = DATA.joinpath("splits")
    return {
        "ft_train": str(split_dir.joinpath(f"{ft_prefix}__ft_train.tsv")),
        "ft_val": str(split_dir.joinpath(f"{ft_prefix}__ft_val.tsv")),
    }


def save_final_lora_checkpoint(
    trained_model,
    params,
    model_folder,
    dataset_dict,
    training_args,
    metrics_dict,
    lora_fp_after,
    *,
    is_world_process_zero=True,
):
    if not is_world_process_zero:
        return

    final_root = model_folder.joinpath("final_checkpoints")
    final_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = _sanitize_for_path(f"{params['model_file_name']}__{timestamp}")
    export_dir = final_root.joinpath(run_label)
    export_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = export_dir.joinpath("tokenizer")
    adapter_dir = export_dir.joinpath("adapter")
    head_path = export_dir.joinpath("esm_with_pe_head.pt")
    metadata_path = export_dir.joinpath("metadata.json")

    base_model = getattr(trained_model, "base_model", trained_model)

    # Save tokenizer (optional)
    tokenizer = dataset_dict.get("tokenizer")
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(tokenizer_dir)

    # Save LoRA adapter weights
    if hasattr(base_model, "save_pretrained"):
        base_model.save_pretrained(adapter_dir)
    else:
        raise RuntimeError("Base model does not support save_pretrained; cannot export final checkpoint.")

    # Persist EsmWithPE head parameters (if any)
    head_state = _collect_esm_head_state(trained_model)
    torch.save(
        {
            "state_dict": head_state,
            "pe_dim": getattr(trained_model, "pe_dimension", None),
            "num_labels": getattr(trained_model, "num_labels", None),
        },
        head_path,
    )

    dataset = dataset_dict.get("dataset")
    uses_positional = bool(getattr(dataset, "return_pe", False))
    tokenizer_max_length = getattr(dataset, "max_length", None)
    metadata = {
        "created_at": timestamp,
        "model_file_name": params.get("model_file_name"),
        "seed": params.get("seed"),
        "data_set_name": params.get("data_set_name"),
        "embedding_subfolder": params.get("embeddingSubfolder"),
        "lm_name": params.get("LM_name"),
        "lm_repo": params.get("LM_path") or resolve_lm_repo(params),
        "trust_remote_code": params.get("trust_remote_code", False),
        "pe_dim": getattr(trained_model, "pe_dimension", None),
        "num_labels": getattr(trained_model, "num_labels", None),
        "uses_positional_encoding": uses_positional,
        "tokenizer_max_length": tokenizer_max_length,
        "adapter_subdir": "adapter",
        "tokenizer_subdir": "tokenizer" if tokenizer is not None else None,
        "esm_head_state": head_path.name,
        "lora_target_modules": params.get("lora_target_modules"),
        "lora_r": params.get("lora_r"),
        "lora_alpha": params.get("lora_alpha"),
        "lora_dropout": params.get("lora_dropout"),
        "lora_learning_rate": params.get("lora_learning_rate"),
        "lora_weight_decay": params.get("lora_weight_decay"),
        "lora_num_train_epochs": params.get("lora_num_train_epochs"),
        "use_two_stage_training": params.get("use_two_stage_training", False),
        "training_args": {
            "learning_rate": getattr(training_args, "learning_rate", None),
            "weight_decay": getattr(training_args, "weight_decay", None),
            "warmup_ratio": getattr(training_args, "warmup_ratio", None),
            "warmup_steps": getattr(training_args, "warmup_steps", None),
            "lr_scheduler_type": str(getattr(training_args, "lr_scheduler_type", "")),
            "gradient_accumulation_steps": getattr(training_args, "gradient_accumulation_steps", None),
            "per_device_train_batch_size": getattr(training_args, "per_device_train_batch_size", None),
            "per_device_eval_batch_size": getattr(training_args, "per_device_eval_batch_size", None),
        },
        "metrics": {
            "train": _make_serializable_metrics(metrics_dict.get("train_metrics") if metrics_dict else None),
            "val": _make_serializable_metrics(metrics_dict.get("val_metrics") if metrics_dict else None),
        },
        "lora_fingerprint": lora_fp_after,
        "dataset_paths": {
            "fine_tuning": str(DATA_SETS.joinpath(params["data_set_name"])),
            "embedding_folder": str(EMBEDDINGS.joinpath(params["LM_name"]).joinpath(params["embeddingSubfolder"])),
        },
        "splits": _build_ft_split_paths(params),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[EXPORT] Final LoRA checkpoint saved to {export_dir}")
    return export_dir


def init_fresh_model(base_model_class, base_model_config, base_state_dict):
    """Instantiate a fresh copy of the base model with the provided config and weights."""

    cfg = copy.deepcopy(base_model_config)
    fresh_model = base_model_class(cfg)
    fresh_model.load_state_dict(base_state_dict, strict=True)
    return fresh_model


class SavePredictionsMetricsWrapper:
    def __init__(self, val_dataset, original_compute_metrics, params):
        self.val_dataset = val_dataset
        self.original_compute_metrics = original_compute_metrics
        self.params = params
        self.trainer = None

    def __call__(self, eval_pred):
        # Run original metrics
        results = self.original_compute_metrics(eval_pred)

        if self.trainer is None:
            return results

        try:
            # logits are numpy arrays here
            logits, labels = eval_pred
            if isinstance(logits, tuple):
                logits = logits[0]

            # Apply Softmax to get probabilities (assuming binary classification with 2 outputs)
            # logits shape: (N, 2)
            probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
            
            # Recover Gene IDs
            # val_dataset is a Subset -> dataset -> LoraDataset (or DataSet_PE/Residual)
            ds = self.val_dataset
            indices = None
            
            # Unpack subset(s)
            while hasattr(ds, "dataset"):
                if hasattr(ds, "indices"):
                    if indices is None:
                        indices = ds.indices
                    else:
                        # If nested subsets, this logic might need adjustment, 
                        # but typically it's just one Subset level.
                        # Mapping indices logic: indices = [parent_indices[i] for i in indices]
                        indices = [ds.dataset.indices[i] for i in indices] 
                        pass 
                ds = ds.dataset

            # Now ds should be the root dataset (LoraDataset or DataSet_PE/Residual)
            # And indices should be the indices in that root dataset
            
            if indices is not None and hasattr(ds, "dataSet_df"):
                # Retrieve Gene_IDs
                gene_ids = ds.dataSet_df.iloc[indices]["Gene_ID"].values
                
                # Check lengths
                if len(gene_ids) != len(probs):
                    print(f"[SavePreds] Warning: Length mismatch gene_ids={len(gene_ids)} vs probs={len(probs)}")
                    return results

                # Create DataFrame
                df_res = pd.DataFrame({
                    "Gene_ID": gene_ids,
                    "score": probs,
                    "label": labels
                })

                # Construct path
                save_dir = self.trainer.args.output_dir
                epoch = self.trainer.state.epoch
                if epoch is None:
                    epoch = 0
                
                # Use model_file_name as prefix if available, ensure integer epoch
                prefix = self.params.get("model_file_name", "run")
                epoch_int = int(epoch)
                
                # Save
                p = Path(save_dir).joinpath(f"{prefix}_predictions_epoch_{epoch_int}.tsv")
                df_res.to_csv(p, sep="\t", index=False)
                print(f"[SavePreds] Saved validation predictions to {p}")
            else:
                 print("[SavePreds] Could not resolve dataset to Gene_IDs")

        except Exception as e:
            print(f"[SavePreds] Failed to save predictions: {e}")

        return results


def run_lora_training(
    active_model,
    params,
    model_folder,
    dataset_dict,
    active_peft_config,
    active_training_args,
    get_metrics_fn,
    evaluate_fn,
    use_positional_encoding=None,
):
    """Execute LoRA pre-training and/or fine-tuning using the provided configuration."""

    torch.cuda.empty_cache()
    lm_repo = params.get("LM_path", "")
    lm_suffix = lm_repo.replace("/", "__") if lm_repo else params["LM_name"]
    
    seed_suffix = f"_seed{params['seed']}" if 'seed' in params else ""

    adapter_dir = model_folder.joinpath(f"{params['model_name']}_{params['data_set_name'][:3]}_{lm_suffix}{seed_suffix}_adapters")
    print(f"Adapter dir: {adapter_dir}")
    if not os.path.exists(adapter_dir):
        os.makedirs(adapter_dir, exist_ok=True)

    result_dict = None

    if "pre-training" in params['data_set_name']:
        active_model = get_peft_model(active_model, active_peft_config)
        print("peft_config keys:", list(getattr(active_model, "peft_config", {}).keys()))
        print("active adapters:", getattr(active_model, "active_adapters", None))
        print("Trainable parameters report:")
        active_model.print_trainable_parameters()
        active_model.set_adapter("default")
        print("LoRA fingerprint (init):", json.dumps(lora_fingerprint(active_model), indent=2))

    if "fine-tuning" in params['data_set_name']:
        pe_enabled = use_positional_encoding if use_positional_encoding is not None else True
        # check if adapter_dir exists and has a valid adapter_config.json
        if os.path.exists(adapter_dir) and os.path.exists(os.path.join(adapter_dir, "adapter_config.json")):
            from peft import PeftModel
            print(f"Loading model trained on all species from {adapter_dir} for fine-tuning")
            active_model = PeftModel.from_pretrained(
                active_model,
                adapter_dir,
            )
            active_model.set_adapter("default")
            assert set(active_model.peft_config["default"].target_modules) == set(active_peft_config.target_modules)
            if pe_enabled:
                log(f"Loading positional encoding with dim {params['pe_dim']}")
                active_model = EsmWithPE(active_model, params["pe_dim"])
            print_adapter_config(adapter_dir, active_model)
            print("[SANITY] A/B nz after LOAD:", fingerprint_split_AB(active_model))
            compare_saved_vs_live_keys(adapter_dir, active_model)

            defreeze_lora_layers(active_model)
        else:
            print("No pre-trained model found, loading base model for fine-tuning")
            active_model = get_peft_model(active_model, active_peft_config)
            print("peft_config keys:", list(getattr(active_model, "peft_config", {}).keys()))
            print("active adapters:", getattr(active_model, "active_adapters", None))
            print("Trainable parameters report:")
            active_model.print_trainable_parameters()
            active_model.set_adapter("default")
            print("LoRA fingerprint (init):", json.dumps(lora_fingerprint(active_model), indent=2))
            if pe_enabled:
                active_model = EsmWithPE(active_model, params["pe_dim"])

        print("peft_config keys (loaded):", list(getattr(active_model, "peft_config", {}).keys()))
        print("active adapters (loaded):", getattr(active_model, "active_adapters", None))
        active_model.print_trainable_parameters()
        print("LoRA fingerprint (loaded):", json.dumps(lora_fingerprint(active_model), indent=2))

        epochs_per_stage_ft = (
            [max(1, int(active_training_args.num_train_epochs))]
            if not params.get("use_two_stage_training", False)
            else [max(1, int(active_training_args.num_train_epochs // 2))] * 2
        )
        fp_before = lora_fingerprint(active_model)
        trainer, _ = train_with_progressive_freezing(
            active_model,
            dataset_dict,
            dataset_dict["tokenizer"],
            SavePredictionsMetricsWrapper(dataset_dict["val_dataset"], get_metrics_fn, params),
            active_training_args,
            params,
            num_freezing_stages=(len(epochs_per_stage_ft) - 1),
            epochs_per_stage=epochs_per_stage_ft,
        )
        fp_after = lora_fingerprint(active_model)

        result_dict = evaluate_fn(trainer, dataset_dict, params)

        print("[FINETUNE] eval metrics (after FT):", result_dict["val_metrics"])
        print("LoRA fingerprint (after FT):", json.dumps(lora_fingerprint(active_model), indent=2))
        print("ΔL2:", fp_after["l2_sum"] - fp_before["l2_sum"])

        if trainer.is_world_process_zero:
            export_dir = save_final_lora_checkpoint(
                active_model,
                params,
                model_folder,
                dataset_dict,
                active_training_args,
                result_dict,
                fp_after,
            )
            if export_dir is not None:
                params["final_checkpoint_path"] = str(export_dir)

    if "pre-training" in params['data_set_name']:
        print("\n[PRETRAIN] starting…")
        fp_before = lora_fingerprint(active_model)
        for n, p in active_model.named_parameters():
            if "lora_" in n:
                assert p.requires_grad, f"FROZEN LoRA param: {n}"

        b_seen = 0
        for n, p in active_model.named_parameters():
            if "lora_B" in n:
                print("[BEFORE] ", n, "requires_grad=", p.requires_grad,
                      "abs_sum=", float(p.detach().abs().sum()))
                b_seen += 1
                if b_seen >= 3:
                    break
        total_epochs = max(1, int(round(active_training_args.num_train_epochs)))
        if params.get("use_two_stage_training", False):
            stage_count = min(2, total_epochs)
            base_epochs = total_epochs // stage_count
            stage_epochs = [base_epochs] * stage_count
            for i in range(total_epochs % stage_count):
                stage_epochs[i] += 1
        else:
            stage_epochs = [total_epochs]

        trainer, _ = train_with_progressive_freezing(
            active_model,
            dataset_dict,
            dataset_dict["tokenizer"],
            SavePredictionsMetricsWrapper(dataset_dict["val_dataset"], get_metrics_fn, params),
            active_training_args,
            params,
            num_freezing_stages=max(0, len(stage_epochs) - 1),
            epochs_per_stage=stage_epochs,
        )

        fp_after = lora_fingerprint(active_model)
        b_seen = 0
        for n, p in active_model.named_parameters():
            if "lora_B" in n:
                print("[AFTER]  ", n, "requires_grad=", p.requires_grad,
                      "abs_sum=", float(p.detach().abs().sum()))
                b_seen += 1
                if b_seen >= 3:
                    break

        opt = trainer.optimizer
        total = 0
        has_lora = 0
        for i, g in enumerate(opt.param_groups):
            group_has = any(
                "lora_" in name
                for name, param in active_model.named_parameters()
                if param.requires_grad and param.data_ptr() in {id(p) for p in g["params"]}
            )
            print(f"[OPT] group {i} lr={g.get('lr')} wd={g.get('weight_decay')} size={len(g['params'])} has_lora={group_has}")
            total += len(g['params'])
            has_lora += int(group_has)
        print(f"[OPT] groups={len(opt.param_groups)} params_total={total} groups_with_lora={has_lora}")

        result_dict = evaluate_fn(trainer, dataset_dict, params)

        print("[PRETRAIN] eval metrics (snapshot):", result_dict["val_metrics"])
        print("LoRA fingerprint (after pretrain):", json.dumps(lora_fingerprint(active_model), indent=2))
        print("ΔL2:", fp_after["l2_sum"] - fp_before["l2_sum"])

        if trainer.is_world_process_zero:
            print(f"[SAVE] adapters -> {adapter_dir}")
            active_model.save_pretrained(adapter_dir)
            tokenizer = dataset_dict.get("tokenizer")
            if tokenizer is not None:
                tokenizer.save_pretrained(adapter_dir)

        print_adapter_config(adapter_dir, active_model)
        print("[SANITY] A/B nz after PRETRAIN:", fingerprint_split_AB(active_model))

    if result_dict is None:
        raise RuntimeError("LoRA training finished without producing evaluation metrics.")

    return active_model, result_dict
