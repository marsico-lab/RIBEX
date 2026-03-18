# initialise environment
# Should be included in all scripts that are called with a new python interpreter instance


import os
from pathlib import Path
import torch
import datetime

## Logging

# gets called when child script is run
def initLogger(scriptPath):
    global LOGS, LOGFILE
    thisScript=Path(scriptPath)
    script_parent = thisScript.parent.name
    script_name = thisScript.name
    start_time = datetime.datetime.now().isoformat()[:-7]

    if LOGS.exists() == False:
        LOGS.mkdir(parents=True, exist_ok=True)

    logfileName = f"{script_name}_{start_time}.log"
    logfilePath = LOGS.joinpath(script_parent) #subfolder
    if logfilePath.exists() == False:
        logfilePath.mkdir()
    logfilePath = logfilePath.joinpath(logfileName) #actual file

    LOGFILE = open(logfilePath, "a")#actual file handle

def log(message, newline=True, doPrint=True, doSave=True, indentation=0):
    global logfilePath
    global LOGFILE

    if(indentation > 0):
        message = "\t"*indentation+message

    if doPrint:
        print(message)

    if(newline):
        message += "\n"

    if(doSave):
        LOGFILE.write(message)
        LOGFILE.flush() #because we do not close the file after write


## General script parameters
threads = 16
webWorkerThreads = 32 #8->3/s (stable) , 16->3.5/s (unstable), 32->3s (unstable) 64->2/s (unstable), 128->3/s, 192->
bufferSize = 100 #mainly/only for dataset_raw generation...

thresholdPos = 1


## Setup environment

#Set general folders
DEFAULT_REPOSITORY = "/path/to/RBP_IG_storage"
REPOSITORY = Path(os.getenv("REPOSITORY", DEFAULT_REPOSITORY))
DATA = REPOSITORY.joinpath("data")

LOGS = DATA.joinpath("logs")
FIGURES = DATA.joinpath("figures")

#DATA subfolders
DATA_ORIGINAL = DATA.joinpath("data_original")
DATA_RAW = DATA.joinpath("data_raw")
DATA_SETS = DATA.joinpath("data_sets")
EMBEDDINGS = DATA.joinpath("embeddings")
CACHE = DATA.joinpath("cache")
TORCH_MODEL_CACHE = DATA.joinpath("torch_model_cache")
MODELS = DATA.joinpath("models")
DATA_CLUST = DATA_RAW.joinpath("clust")
ATTRIBUTIONS = DATA.joinpath("attributions")

## Initialization function (needs to be called by the child script)
def initialize(scriptPath):

    #initialize logger so we can log stuff
    initLogger(scriptPath)

    #set wandb environment variable for compatibility with linux systems
    os.environ["WANDB_START_METHOD"] = "thread"

    #Print variables
    log(f"Environment Initialization:")
    log(f"\tREPOSITORY=\t{REPOSITORY}")
    log(f"\tLOGS=\t\t{LOGS}")
    log(f"\tFIGURES=\t{FIGURES}")
    log(f"\tDATA=\t\t{DATA}")
    log(f"\t\tDATA_ORIGINAL=\t{DATA_ORIGINAL}")
    log(f"\t\tDATA_RAW=\t{DATA_RAW}")
    log(f"\t\tDATA_SETS=\t{DATA_SETS}")
    log(f"\t\tEMBEDDINGS=\t{EMBEDDINGS}")
    log(f"\t\tCACHE=\t{CACHE}")
    log(f"\t\tTORCH_MODEL_CACHE=\t{TORCH_MODEL_CACHE}")
    log(f"\t\tMODELS=\t{MODELS}")
    log(f"\t\tDATA_CLUST=\t{DATA_CLUST}")
    log(f"\t\tATTRIBUTIONS=\t{ATTRIBUTIONS}")

    #Check folder that need to exist beforehand
    if(not REPOSITORY.exists()):
        raise UserWarning(f"Folder does not exist: {REPOSITORY}")
    if(not DATA.exists()):
        raise UserWarning(f"Folder does not exist: {DATA}")
    if(not DATA_ORIGINAL.exists()):
        raise UserWarning(f"Folder does not exist: {DATA_ORIGINAL}")

    #Create project related folders if necessary
    DATA_RAW.mkdir(exist_ok=True)
    DATA_SETS.mkdir(exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    TORCH_MODEL_CACHE.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    #Set torch model directory
    torch.hub.set_dir(TORCH_MODEL_CACHE)
