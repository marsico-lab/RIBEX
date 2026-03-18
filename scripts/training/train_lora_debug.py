#!/usr/bin/env python3
import os, argparse, hashlib, json
import torch

# repo imports
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *
from scripts.training.utils import (
    parseArguments, setupFolders, getDataset, getLatestModel,
    train_with_progressive_freezing, defreeze_lora_layers
)
from scripts.training.analyze_utils import evaluateModel, manualLogging, getMetrics

from transformers import TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import json
from safetensors.torch import load_file

initialize(__file__)

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

def build_training_args(output_dir, seed, pin_memory=True, num_workers=2,
                        train_bs=8, eval_bs=8, epochs=5, lr=3e-4):
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        evaluation_strategy="epoch",
        save_strategy="no",
        seed=seed,
        fp16=True,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=pin_memory,
        dataloader_num_workers=num_workers,
        logging_steps=20,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pretrain", "finetune"], required=True)
    parser.add_argument("--adapter_dir", type=str,
                        default="/vol/storage/RBP_IG/lora_binding_sites/Lora_binding_sites_pre-trained_debug",
                        help="Where to save/load LoRA adapters")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3.5e-4)
    parser.add_argument("--train_bs", type=int, default=8)
    parser.add_argument("--eval_bs", type=int, default=8)
    parser.add_argument("--target", type=str, default="key,value",
                        help='Comma list of target modules (e.g. "key,value" or "query,key,value")')
    parser.add_argument("--rank", type=int, default=3)
    parser.add_argument("--lora_alpha", type=float, default=0.42)
    parser.add_argument("--lora_dropout", type=float, default=0.45)
    parser.add_argument("--save_after", action="store_true")
    args_cli, _ = parser.parse_known_args()

    # ---- use your existing CLI to pick dataset/model/etc. ----
    params = parseArguments()
    modelFolder, embeddingFolder, dataSetPath = setupFolders(params)
    print(f"Dataset: {dataSetPath}\nModel folder: {modelFolder}\nEmbeddings: {embeddingFolder}")
    datasets = getDataset(params, dataSetPath, embeddingFolder)

    # base HF model (EsmForSequenceClassification in your getLatestModel for Lora)
    base = getLatestModel(params, modelFolder)

    # LoRA config (match your train.py defaults)
    targets = [t.strip() for t in args_cli.target.split(",") if t.strip()]
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args_cli.rank,
        lora_alpha=args_cli.lora_alpha,
        lora_dropout=args_cli.lora_dropout,
        target_modules=targets,
        bias="none",
    )

    if args_cli.mode == "pretrain":
        # wrap the base with LoRA
        model = get_peft_model(base, peft_config)
        print("peft_config keys:", list(getattr(model, "peft_config", {}).keys()))
        print("active adapters:", getattr(model, "active_adapters", None))
        print("Trainable parameters report:")
        model.print_trainable_parameters()
        model.set_adapter("default")  # optional, ensures correct adapter is trainable
        print("LoRA fingerprint (init):", json.dumps(lora_fingerprint(model), indent=2))
    
    # minimal TrainingArguments
    out_dir = f"{params['model_name']}-lora-debug"
    training_args = build_training_args(
        output_dir=out_dir,
        seed=params["seed"],
        pin_memory=True,
        num_workers=params.get("num_workers", 2),
        train_bs=args_cli.train_bs,
        eval_bs=args_cli.eval_bs,
        epochs=args_cli.epochs,
        lr=args_cli.lr,
    )

    # ---- PRETRAIN: train adapters from scratch and save only adapters ----
    if args_cli.mode == "pretrain":
        print("\n[PRETRAIN] starting…")
        
        fp_before = lora_fingerprint(model)
        # 0) sanity: make sure we aren’t freezing LoRA by accident
        for n,p in model.named_parameters():
            if "lora_" in n:
                assert p.requires_grad, f"FROZEN LoRA param: {n}"
        # 1) check a few B matrices have gradients turned on and contain non-zeros BEFORE train
        b_seen = 0
        for n,p in model.named_parameters():
            if "lora_B" in n:
                print("[BEFORE] ", n, "requires_grad=", p.requires_grad,
                      "abs_sum=", float(p.detach().abs().sum()))
                b_seen += 1
                if b_seen >= 3:
                    break
        
        trainer, _ = train_with_progressive_freezing(
            model, datasets, datasets["tokenizer"], getMetrics, training_args, params
        )
        fp_after = lora_fingerprint(model)
        
        b_seen = 0
        for n,p in model.named_parameters():
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
            group_has = any("lora_" in n for n,_ in model.named_parameters() if _.requires_grad and _.data_ptr() in {id(p) for p in g["params"]})
            print(f"[OPT] group {i} lr={g.get('lr')} wd={g.get('weight_decay')} size={len(g['params'])} has_lora={group_has}")
            total += len(g['params']); has_lora += int(group_has)
        print(f"[OPT] groups={len(opt.param_groups)} params_total={total} groups_with_lora={has_lora}")
        
        # quick eval just to log something
        resultDict = evaluateModel(trainer, datasets, params)
        print("[PRETRAIN] eval metrics (snapshot):", resultDict["val_metrics"])
        print("LoRA fingerprint (after pretrain):", json.dumps(lora_fingerprint(model), indent=2))
        print("ΔL2:", fp_after["l2_sum"] - fp_before["l2_sum"])

        opt = trainer.optimizer
        total = 0
        has_lora = 0
        for i, g in enumerate(opt.param_groups):
            group_has = any("lora_" in n for n,_ in model.named_parameters() if _.requires_grad and _.data_ptr() in {id(p) for p in g["params"]})
            print(f"[OPT] group {i} lr={g.get('lr')} wd={g.get('weight_decay')} size={len(g['params'])} has_lora={group_has}")
            total += len(g['params']); has_lora += int(group_has)
        print(f"[OPT] groups={len(opt.param_groups)} params_total={total} groups_with_lora={has_lora}")

        # save ONLY adapters
        os.makedirs(args_cli.adapter_dir, exist_ok=True)
        model.save_pretrained(args_cli.adapter_dir)
        datasets["tokenizer"].save_pretrained(args_cli.adapter_dir)  # optional, convenient
        print(f"[SAVE] adapters -> {args_cli.adapter_dir}")
        print_adapter_config(args_cli.adapter_dir, model)
        print("[SANITY] A/B nz after PRETRAIN:", fingerprint_split_AB(model))

    # ---- FINETUNE: load adapters, show difference, fine-tune, optionally save ----
    elif args_cli.mode == "finetune":
        print("\n[FINETUNE] loading adapters from:", args_cli.adapter_dir)
        if not os.path.exists(args_cli.adapter_dir):
            raise FileNotFoundError(f"Adapter dir not found: {args_cli.adapter_dir}")

        # attach saved adapters to the SAME base arch
        model = PeftModel.from_pretrained(base, args_cli.adapter_dir)
        model.set_adapter("default")
        
        assert set(model.peft_config["default"].target_modules) == set(peft_config.target_modules)
        
        print_adapter_config(args_cli.adapter_dir, model)
        print("[SANITY] A/B nz after LOAD:", fingerprint_split_AB(model))
        compare_saved_vs_live_keys(args_cli.adapter_dir, model)
        
        
        defreeze_lora_layers(model)  # your helper, ensures LoRA params are trainable
        

        print("peft_config keys (loaded):", list(getattr(model, "peft_config", {}).keys()))
        print("active adapters (loaded):", getattr(model, "active_adapters", None))
        model.print_trainable_parameters()
        print("LoRA fingerprint (loaded):", json.dumps(lora_fingerprint(model), indent=2))

        # build trainer and do a PRE-TRAIN eval to verify we're not identical to scratch
        fp_before = lora_fingerprint(model)
        trainer, _ = train_with_progressive_freezing(
            model, datasets, datasets["tokenizer"], getMetrics, training_args, params
        )
        fp_after = lora_fingerprint(model)
        # (train_with_progressive_freezing trains immediately per your utils; if you prefer a
        # pure "pre-eval" first, you can run trainer.evaluate() right after creating the trainer.)

        resultDict = evaluateModel(trainer, datasets, params)
        print("[FINETUNE] eval metrics (after FT):", resultDict["val_metrics"])
        print("LoRA fingerprint (after FT):", json.dumps(lora_fingerprint(model), indent=2))
        print("ΔL2:", fp_after["l2_sum"] - fp_before["l2_sum"])

        if args_cli.save_after:
            model.save_pretrained(args_cli.adapter_dir)
            print(f"[SAVE] updated adapters -> {args_cli.adapter_dir}")

    print("\n[done]")

if __name__ == "__main__":
    main()
    
    
# # %%
# from safetensors.torch import load_file
# import torch

# path = "/vol/storage/RBP_IG/lora_binding_sites/Lora_binding_sites_pre-trained_debug/adapter_model.safetensors"
# tensors = load_file(path)  # dict[str, torch.Tensor]

# total = 0
# nonzero = 0
# l2_sum = 0.0
# per_tensor = []

# for name, t in tensors.items():
#     nz = int((t != 0).sum().item())
#     total += t.numel()
#     nonzero += nz
#     l2 = float(torch.linalg.vector_norm(t).item())
#     l2_sum += l2
#     per_tensor.append((name, t.shape, nz, l2))

# print(f"TOTAL elements: {total}")
# print(f"NONZERO elements: {nonzero}")
# print(f"L2 norms sum: {l2_sum:.6f}")
# print("Top 10 tensors by L2 norm:")
# for name, shape, nz, l2 in sorted(per_tensor, key=lambda x: -x[3])[:10]:
#     print(f"  {name:60s} {str(shape):16s} nz={nz:8d}  ||t||2={l2:.6f}")


# # %%
