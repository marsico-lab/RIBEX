# Initialize global environment and import useful utility functions
import sys
from pathlib import Path

sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *
initialize(__file__)

from analyze_utils import *
from utils import getModelFromCkpt, getDataset, setupFolders, setupLoggers
from scripts.embeddings.utils import pretrainedModelInfos
import pytorch_lightning as pl
import wandb

from tqdm import tqdm
from natsort import natsorted
import pickle
import argparse
from pprint import pformat, pprint

parser = argparse.ArgumentParser(description="Training analysis script")
parser.add_argument(
    "-D", "--device", dest="device", action="store", type=str, help='GPUs to be used', default="cuda:1" #"auto" uses all
)

args = parser.parse_args()
# write commandline arguments to params dict
cmd_params = {}
for key, value in args._get_kwargs():
    cmd_params[key] = value
cmd_params["accelerator"] = "auto" #uses best. Options: "cpu", "gpu", "tpu", "ipu", "auto"


# Compute Metrics (MCC, BACC, etc.) on classififer original validation set
for modelFolder in list(MODELS.iterdir()):
    modelName = modelFolder.name
    cmd_params["model_name"] = modelName
    #If model name starts with "Peng" then skip
    if modelName.startswith("Peng"): #TODO: tis is just for debugging purposes
        continue
    log(f"Processing {modelName}")

    lightningLogsFolder = modelFolder.joinpath("lightning_logs")
    for instanceFolder in lightningLogsFolder.iterdir():
        instanceName = instanceFolder.name
        log(f"{instanceName}", indentation=1)
        
        # Get latest Checkpoint
        checkpointFolder = instanceFolder.joinpath("checkpoints")
        if not checkpointFolder.exists() == True:
            log(f"No ckpt folder", indentation=2)
            continue

        ckptName = natsorted([path.name for path in checkpointFolder.iterdir()])[
            -1
        ]  # get last checkpoint
        ckptPath = checkpointFolder.joinpath(ckptName)
        cmd_params["checkpoint_path"] = ckptPath
        log(f"\tCheckpoint={ckptName}", indentation=2)
        #print(f"cmd_params:\n\t{pformat(cmd_params)}")

        # Get Model & Params
        model = getModelFromCkpt(params=cmd_params)
        if(modelName in ["Peng", "Peng_6", "Lora", "Linear_pytorch"]): # set pytroch model to inference state
            model.eval()  # disable randomness, dropout, etc...
        params = model.params
        #print(f"params:\n\t{pformat(params)}")
        
        #override training parameters with cmd params for inference
        for key in cmd_params: 
            params[key] = cmd_params[key]
        
        # Seeding
        pl.seed_everything(params["seed"], workers=True)

        # Get Training and Validation Dataset
        modelFolder, embeddingFolder, dataSetPath = setupFolders(params)
        datasetDict = getDataset(params, dataSetPath, embeddingFolder)
        
        ## Setup Loggers ##
        logger_TB, logger_WB = setupLoggers(params, modelFolder)

        # Run classififer on original dataset
        resultDict = evaluateModel(model, datasetDict, params)
        manualLogging(resultDict["train_metrics"], resultDict["val_metrics"], params, logger_WB, logger_TB)
        logger_TB.finalize("success")
        wandb.finish()
        
        # Performance: general metrics on validation set
        log(f"Performance Analysis: General metrics on validation set", indentation=3)

        metricsDict = getMetrics(preds=resultDict["val_probs"], gt_labels=resultDict["val_labels"])
        #print(f"metricsDict:\n\t{pformat(metricsDict)}")   
        logMetrics(metricsDict)
        plotCurves(
            metricsDict, fileNamePrefix=f"{modelName}__{instanceName}__{ckptName}__"
        )


        if(False): #TODO: refactor/fix that code. We are missing ids values, otherwise everything should be tehre for refactoring
            
            # Performance: IDR-only vs RBD-having
            log(f"Performance Analysis: IDR-only vs RBD-having proteins", indentation=3)
            analyzeProteinAnnotationTypes(
                probabilities=probabilities,
                labels=labels,
                dataSet_df=dataSet_df,
                idx=idx,
                indentation=4,
            )
            # TODO: difference, p-value, effect size

            # Performance: truncated vs non truncated
            log(
                f"Performance Analysis: truncated vs non-truncated embeddings",
                indentation=3,
            )

            embeddingFolder = EMBEDDINGS.joinpath(params["LM_name"]).joinpath(
                params["embeddingSubfolder"]
            )
            analyzeProteinTruncationEffect(
                probabilities=probabilities,
                labels=labels,
                dataSet_df=dataSet_df,
                embeddingFolder=embeddingFolder,
                idx=idx,
                indentation=4,
            )
            # TODO: difference, p-value, effect size


log("done.")
