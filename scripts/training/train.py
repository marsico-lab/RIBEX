#Torch and Lightning stuff
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import faulthandler; faulthandler.enable()
import torch
import torch.distributed as dist

import shutil
import tempfile

# Initialize global environment and import useful utility functions
import sys
from pathlib import Path

# Enable wandb logging (set WANDB_MODE externally if you need to disable it)
import os
import json
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *
#Outsources functionality
from scripts.training.utils import (
    parseArguments,
    setupFolders,
    getDataset,
    getLatestModel,
    setupLoggers,
    train_with_progressive_freezing,
    defreeze_lora_layers,
    lora_fingerprint,
    print_adapter_config,
    fingerprint_split_AB,
    compare_saved_vs_live_keys,
    run_lora_training,
    init_fresh_model,
)
from peft import get_peft_model, LoraConfig, TaskType
from scripts.training.utils import computeClassWeight, WeightedLossTrainer, WandbCallback, EsmWithPE
from scripts.training.analyze_utils import evaluateModel, manualLogging

initialize(__file__)
#os.environ['WANDB_DISABLED'] = 'false'

# TODO: bootstrapping (requires iteration amount, automatic increase/change of seed )

def main():
    
    ## Parameters & Initializaion ## 
    params = parseArguments()

    # Dataloader paramater
    params["prefetch_factor"] = 2
    params["num_workers"] = 3
    params["pin_memory"] = True
    params["persistent_workers"] = True
    params["pin_memory_device"] = ""  # TODO: whats this?

    # Training Paramaters
    torch.set_float32_matmul_precision('high')
    params["weight_decay"] = 5e-4
    params["factor"] = 0.85
    if(-1 in params["devices"]): #use cpu
        params["accelerator"] = "cpu"
        params["devices"] = params["threads"]
    else: #use GPU or better
        params["accelerator"] = "auto"  #uses best. Options: "cpu", "gpu", "tpu", "ipu", "auto"
        #devices list from commandline arguments are taken

    # Print/Log paramaters
    log("Parameters:")
    for key in params.keys():
        log(f"\t{key}: {params[key]}")

    # Seed everything (numpy, torch, lightning, even workers)
    pl.seed_everything(params["seed"], workers=True)


    ## Input/output structures ##
    modelFolder, embeddingFolder, dataSetPath = setupFolders(params)

    ## Dataset ##
    # if not params["is_hpo"]: # no need to load dataset for HPO as this is done in the objective function
    if not params["is_hpo"]:
        datasetDict = getDataset(params, dataSetPath, embeddingFolder)
    #Get Model (load from checkpoint or create new instance)
    model = getLatestModel(params, modelFolder)

    ## Setup Loggers ##
    logger_TB, logger_WB = setupLoggers(params, modelFolder)

    ## Setup Trainer ##
    if params["model_name"] in ["Lora"]: # Hugging Face models
        import os
        #os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        #os.environ["WANDB_PROJECT"] = "predict-rbp"
        #os.environ["WANDB_LOG_MODEL"] = "checkpoint"
        from torch.utils.data import DataLoader

        # Helper Functions and Data Preparation
        from analyze_utils import getMetrics 
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        from transformers import Trainer, TrainingArguments
        from transformers.trainer_utils import is_main_process

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        base_global_bs = max(1, int(params.get("bs", 2)))
        per_device_bs = max(1, int(params.get("lora_per_device_bs", 2)))
        eff_per_step = max(1, per_device_bs * world_size)
        grad_accum = max(1, (base_global_bs + eff_per_step - 1) // eff_per_step)
        run_tag = params.get("run_tag", "")
        run_suffix = f"_{run_tag}" if run_tag else ""
        

        #Trial 30 finished with value: 0.9764611124992371 and parameters: {'learning_rate': 0.00034999854550633457, 'lr_scheduler_type': 'cosine',
        # 'warmup_steps': 1, 'weight_decay': 0.0005327834289253654, 'r': 3, 'lora_alpha': 0.4225760315766923, 
        # 'lora_dropout': 0.4460401894969171, 'target_modules': ['key', 'value']}. Best is trial 30 with value: 0.9764611124992371.
        #{'learning_rate': 6.042923243390243e-05, 'warmup_ratio': 0.007678713369382084, 'num_train_epochs': 2, 'weight_decay': 0.0003558260079168068,
        # 'pe_dim': 128, 'r': 3, 
        # 'lora_alpha': 0.3596292537810787, 'lora_dropout': 0.4981824714966394, 'target_modules': ['query', 'key', 'value']}
        
        #{'learning_rate': 1.3270493558198412e-05,
        #'weight_decay': 0.00013442617790611984,
        #'pe_dim': 128,
        #'r': 4,
        #'lora_alpha': 4.0034401385773295,
        #'lora_dropout': 0.16953055418144058,
        #'target_modules': ['query', 'key']}

        # Convert the model into a PeftModel
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, 
            inference_mode=False, 
            r=params["lora_r"],
            lora_alpha=params["lora_alpha"],
            target_modules=params["lora_target_modules"], #["query", "key", "value"], # also try "dense_h_to_4h" and "dense_4h_to_h"
            lora_dropout=params["lora_dropout"],
            bias="none" # or "all" or "lora_only" 
        )
        # print all lora params
        print(peft_config.__dict__)

        # Training setup
        training_args = TrainingArguments(
            output_dir=f"{params['model_name']}-lora-binding-sites{run_suffix}_{timestamp}",
            learning_rate=params["lora_learning_rate"],
            lr_scheduler_type= "cosine",#"linear", #"cosine",
            gradient_accumulation_steps=grad_accum,
            warmup_ratio=0.03,
            max_grad_norm=1.0,
            per_device_train_batch_size=per_device_bs,  # was 2, now 1 if using 2 GPUs
            per_device_eval_batch_size=per_device_bs,
            num_train_epochs=params["lora_num_train_epochs"], # TODO: tune?
            weight_decay=params["lora_weight_decay"],#params["weight_decay"],
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            greater_is_better=True,
            metric_for_best_model="AUPRC",#"AUPRC", 
            push_to_hub=False,
            logging_dir=None,
            logging_first_step=False,
            logging_steps=200,
            save_total_limit=7,
            no_cuda=False,
            seed=params["seed"],
            gradient_checkpointing=False,
            fp16=True, # to support mixed precision training
            report_to=None,
            remove_unused_columns=False, #Recommended by marc
            dataloader_pin_memory=params["pin_memory"],        # ← enable pinned memory, default: True
            dataloader_num_workers=params["num_workers"],  # ← match your intended workers
            save_safetensors=False if "prott5" in params["LM_name"].lower() else True, # Disable safetensors for T5 (shared weights)
            #place_model_on_device=True, #move model to device automatically
        )
        

        #print training_args.__dict__
        print(f"training_args: {training_args}")

    elif params["model_name"] in ["Peng", "Peng6", "Linear_pytorch", "FiLM_PE"]:
        
        # Define Last and best Checkpoints to be saved
        log(f"Setup ModelCheckpoint")
        checkpoint_callback = ModelCheckpoint(
            filename="{epoch}-{step}-{val_loss:.4f}",
            save_top_k=5,
            monitor="val_loss",
            mode="min"
            #dirpath=modelFolder
        )

        # Setup Trainer
        log(f"Setup trainer")

        print(f"acc={params['accelerator']}, dev={params['devices']}")
        trainer = pl.Trainer(
            accelerator=params["accelerator"], # is "cpu" if -1 in devices
            devices=params["devices"], # is None for -1 in devices commandline parameter
            #num_nodes=1, #TODO?
            max_epochs=params["epochs"],  # Stopping epoch
            logger=[logger_TB, logger_WB],
            callbacks=[checkpoint_callback],  # You may add here additional call back that saves the best model
            # limit_train_batches=150
            # detect_anomaly=True,
            #strategy="ddp"
            # limit_train_batches=limit_train_batches,  # Train only x % (if float) or train only on x batches (if int)
            #limit_val_batches=0.5,
            #strategy=("ddp_find_unused_parameters_true" if params["devices"] > 1 else "auto"),  # for distributed compatibility
        )

    elif params["model_name"] in ["Linear", "RandomForest", "XGBoost", "Random_SK"]: # sklearn models
        pass
        #No trainer setup is needed for sklearn models
    else:
        raise RuntimeError(f"For model {params['model_name']} is no trainer implemented yet.")


    ## Training ##
    from scripts.training.utils import hp_space
    import optuna
    import copy
    log(f"Start training {params['model_file_name']}")

    if params["model_name"] in ["Lora"]:
        trained_model = model
        resultDict = None
        # Train and Save Model
        if params["is_hpo"]: # Hyperparameter optimization
        
            # Prepare a factory that returns a fresh base model for every trial
            base_model_class = model.__class__
            base_model_config = copy.deepcopy(getattr(model, "config", None))
            if base_model_config is None:
                raise RuntimeError("Base model config missing; cannot initialise fresh models for HPO.")

            base_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            # optuna setting
            # before defining objective:
            peft_config_template = peft_config
            training_args_template = training_args

            def objective(trial):
                peft_config_hpo = copy.deepcopy(peft_config_template)
                training_args_hpo = copy.deepcopy(training_args_template)
                # Get hyperparameters
                hyperparameters = hp_space(trial, params["LM_name"])
                # Set hyperparameters for training
                training_args_hpo.learning_rate = hyperparameters["learning_rate"]
                training_args_hpo.lr_scheduler_type = "cosine" ##hyperparameters["lr_scheduler_type"]
                training_args_hpo.warmup_ratio = 0.5 #hyperparameters["warmup_ratio"]
                training_args_hpo.weight_decay = hyperparameters["weight_decay"]
                training_args_hpo.save_strategy = "no"
                training_args_hpo.load_best_model_at_end = False
                training_args_hpo.output_dir = tempfile.mkdtemp(prefix="optuna_", dir=str(modelFolder))
                # Set number of epochs
                training_args_hpo.num_train_epochs = 2 #hyperparameters["num_train_epochs"]
                # Set hyperparameters for LoRA
                peft_config_hpo.r = hyperparameters["r"]
                peft_config_hpo.lora_alpha = hyperparameters["lora_alpha"]
                peft_config_hpo.lora_dropout = hyperparameters["lora_dropout"]
                peft_config_hpo.bias = hyperparameters["bias"]
                peft_config_hpo.target_modules = hyperparameters["target_modules"]

                model_fresh = init_fresh_model(base_model_class, base_model_config, base_state_dict)
                # use positional only for fine-tuning 
                use_positional = False if "pre-training" in params["data_set_name"] else True

                params['pe_dim'] = hyperparameters["pe_dim"]
                model_hpo = EsmWithPE(model_fresh, params["pe_dim"])


                # No need for DataParallel—use DDP launcher instead:
                # (Trainer will wrap this in DDP if you launch with torch.distributed.launch)
                #model_hpo.gradient_checkpointing_enable()

                # print all model trainable params (including lora) and freese the non lora ones
                for name, param in model_hpo.named_parameters():
                    if param.requires_grad:
                        if "lora" not in name:
                            if "query" in name:
                                param.requires_grad = False
                                print(f"freeze {name}")
                            if "key" in name:
                                param.requires_grad = False
                                print(f"freeze {name}")
                            if "value" in name:
                                param.requires_grad = False
                                print(f"freeze {name}")
                        else:
                            print(f"leaving {name} trainable")
                # load dataset
                datasetDict = getDataset(params, dataSetPath, embeddingFolder, is_positional_encoding_none=not use_positional)
                print(f"crit_weight {params['crit_weight']}")
                # update trainer
                trainer_hpo, _ = train_with_progressive_freezing(model_hpo, datasetDict, datasetDict["tokenizer"], getMetrics, training_args_hpo, params)
                try:
                    # Run training
                    trainer_hpo.train()
                    # Evaluate
                    metrics = trainer_hpo.evaluate()
                finally:
                    shutil.rmtree(training_args_hpo.output_dir, ignore_errors=True)

                return metrics["eval_AUPRC"]

            optuna_sampler = optuna.samplers.TPESampler()
            #optuna_sampler = optuna.samplers.RandomSampler()
            optuna_pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=15, interval_steps=10)
            study_name = 'lora_fine_tuning'
            input_data_dir = params["embeddingSubfolder"] + "_"
            hpo_base_dir = DATA.joinpath("optuna_trials") 
            hpo_base_dir.mkdir(parents=True, exist_ok=True)
            storage = 'sqlite:///' + str(hpo_base_dir) + '/lora_fine_tuning_' + input_data_dir + embeddingFolder.parts[-2] + params["data_set_name"][:-4] + params["model_name"] + params['data_set_name'][-16:-4] + '.db'
            
            # remove existing study with same name
            #if os.path.exists(hpo_base_dir.joinpath('lora_fine_tuning_' + input_data_dir + embeddingFolder.parts[-2] + params["data_set_name"][:-4] + params["model_name"] + params['data_set_name'][-16:-4] + '.db')):
            #    os.remove(hpo_base_dir.joinpath('lora_fine_tuning_' + input_data_dir + embeddingFolder.parts[-2] + params["data_set_name"][:-4] + params["model_name"] + params['data_set_name'][-16:-4] + '.db'))
            #    log(f"Removed existing optuna study with same name at {hpo_base_dir.joinpath('lora_fine_tuning_' + input_data_dir + embeddingFolder.parts[-2] + params['data_set_name'][:-4] + params['model_name'] + params['data_set_name'][-16:-4] + '.db')}")
            log(f"Optuna study storage at {storage}")
            study = optuna.create_study(study_name=study_name, storage=storage, sampler=optuna_sampler, pruner=optuna_pruner, direction='maximize', load_if_exists=True)

            study.optimize(objective, n_trials=50, n_jobs=1)
            log(f"Number of finished trials: {len(study.trials)}")
            log("Best trial:")  
            trial = study.best_trial
            log(f"  Value: {trial.value}")
            log("  Params: ")
            for key, value in trial.params.items():
                log(f"    {key}: {value}")
                
        else:
            # save train dataset to file for analysis
            trained_model, resultDict = run_lora_training(
                trained_model,
                params,
                modelFolder,
                datasetDict,
                peft_config,
                training_args,
                getMetrics,
                evaluateModel,
            )

        if resultDict is not None:
            manualLogging(resultDict["train_metrics"], resultDict["val_metrics"], params, logger_WB, logger_TB)

    elif params["model_name"] in ["Peng", "Peng6", "Linear_pytorch", "Random", "FiLM_PE"]:
        
        if "checkpoint_path" in params.keys(): # a checkpoint path was generated
            trainer.fit(model, datasetDict["train_loader"], datasetDict["val_loader"], ckpt_path=params["checkpoint_path"])
        else:
            trainer.fit(model, datasetDict["train_loader"], datasetDict["val_loader"])

    elif params["model_name"] in ["Linear", "RandomForest", "XGBoost", "Random_SK"]: # sklearn models
        model.fit(datasetDict["X_train"], datasetDict["Y_train"])

        print(f"Training done. Evaluating...")
        
        resultDict = evaluateModel(model, datasetDict, params)
        manualLogging(resultDict["train_metrics"], resultDict["val_metrics"], params, logger_WB, logger_TB)
            
        print(f"Saving model...") # This needs to be done manually as sklearn has no own save/load functionality
        #Setup location
        checkpointFolder = modelFolder.joinpath("lightning_logs").joinpath(params['model_file_name']).joinpath("checkpoints")
        checkpointFolder.mkdir(parents=True, exist_ok=True)
        checkpointPath = checkpointFolder.joinpath(f"model.pkl")
        print(f"Saving model to {checkpointPath}")
        model.params = params #save params in model
        #Save model
        import pickle
        with open(checkpointPath, 'wb') as f:
            pickle.dump(model, f)

    else:
        raise RuntimeError(f"For model {params['model_name']} is no training implmented yet.")


    log(f"done.")

if __name__ == "__main__":
    main()
    
    
