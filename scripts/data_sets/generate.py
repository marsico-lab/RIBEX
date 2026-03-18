# should mainly filter based on criteria relevant for us

# create pre-trainign set, fine-tuning set and experimental (RIC) set 


from generate_utils import *
initialize(__file__)

#from scripts.data_sets.generate_utils import *

import pickle 

# Parameters
import sys
from pathlib import Path
workingDir = Path(".")
repo_folder = workingDir.absolute()
sys.path.append(str(repo_folder))
from scripts.initialize import *


dataSetNames = { "bressin19", "RIC", "InterPro"}

#Load datasets
dataSetsRaw = {}
for datasetName in dataSetNames:
    match datasetName:
        case "bressin19": # Bressin (original/baseline)
            path = DATA_RAW.joinpath("bressin19.tsv")
        case "RIC": # RIC (experimental/ best)
            path = DATA_RAW.joinpath("RIC.tsv")
        case "InterPro": # interPro (original approach but better/bigger)
            path = DATA_RAW.joinpath("InterPro.tsv")

    log(f"Loading raw {datasetName} data from {path}")
    dataSetsRaw[datasetName] = pd.read_csv(path, sep="\t")

#### PROTEIN FILTER CRITERIA ####

# apply basic filters to get baseDatasets
baseDataSets = {}
log(f"Filtering raw datasets by column consistency")
for datasetRawName in dataSetsRaw.keys():
    log(f"\t{datasetRawName}")
    datasetRaw = dataSetsRaw[datasetRawName]
    baseDataSets[datasetRawName] = filterProteins(datasetRaw)

### CHECK EMBEDDINGS ###

log(f"Filtering raw datasets by embedding existence")
relevantLMs = ["esm1b_t33_650M_UR50S"] #TODO, "esm2_t33_650M_UR50D", "esm2_t36_3B_UR50D", "esm2_t48_15B_UR50D"] 

for datasetRawName in dataSetsRaw.keys():
    log(f"\t{datasetRawName}")
    required = set(list(baseDataSets[datasetRawName]["Gene_ID"]))
    print(f"\t\tReuqired: {len(required)}")
    missingIDs = set()

    for LM_name in relevantLMs:
        embeddingFolder = EMBEDDINGS.joinpath(LM_name).joinpath(datasetRawName)
        if embeddingFolder.exists() == False:
            raise RuntimeError(f"Embedding folder does not exists: {embeddingFolder}")
        
        existing = set([p.name for p in embeddingFolder.iterdir()])
        missing = required-existing
        print(f"\t\t\t{LM_name} misses { (len(missing)/(len(required))) * 100 :.02f}% ({len(missing)} entries)")

        missingIDs.update(missing)

    print(f"\t\tTotal missing: { (len(missingIDs)/(len(required))) * 100:.02f}% ({len(missingIDs)} entries)")

    existing = required-missing
    selector = [(ID in existing)  for ID in list(baseDataSets[datasetRawName]["Gene_ID"])]

    baseDataSets[datasetRawName] = baseDataSets[datasetRawName][selector]

#### DATASET GENERATION ####
# aggregate/filter proteins based on specific criteria
# creates multiple datasets potentially from the same proteins or multiple protein sources

# Make dataset:
# - filter taxon
# - filter sequence length
# - filter annotation type existence

## Generate specific datasets
dataSets = {}

## BRESSIN 19 ##

if("bressin19" in dataSetNames):
    # Bressin all
    # - all bressin species, used for statistics
    name = "bressin19"
    log(f"Generate {name}")
    dataSets[name] = generateDataset(
        baseDataset=baseDataSets["bressin19"],
        taxons=None,  # all
        #sequenceLength=(50, 6000),  # Bressin 2019 length criteria
        annotationTypes=None  # do not matter for pre training
    )
    with open(DATA_SETS.joinpath(f"{name}.pkl"), 'wb') as f:
        pickle.dump(dataSets[name], f)

    # Bressin pre-train
    # - all bressin species sets but human, used to pre-train a peng network
    name = "bressin19_human_pre-training"
    log(f"Generate {name}")
    dataSets[name] = generateDataset(
        baseDataset=baseDataSets["bressin19"],
        taxons=[9606], allBut=True,  # all but human
        sequenceLength=(50, 6000),  # Bressin 2019 length criteria
        annotationTypes=None  # do not matter for pre training
    )
    with open(DATA_SETS.joinpath(f"{name}.pkl"), 'wb') as f:
        pickle.dump(dataSets[name], f)

    # Bressin human fine tune
    # - used for fine tuning a human peng classifier
    name = "bressin19_human_fine-tuning"
    log(f"Generate {name}")
    dataSets[name]  = generateDataset(
        baseDataset=baseDataSets["bressin19"],
        taxons=[9606],  # only human
        sequenceLength=(50, 6000),  # Bressin 2019 length criteria
        annotationTypes=None  # do not matter for pre training
    )
    with open(DATA_SETS.joinpath(f"{name}.pkl"), 'wb') as f:
        pickle.dump(dataSets[name], f)

## RIC ##
if("RIC" in dataSetNames):
    # RIC all (thr1)
    # - all RIC species, used for statistics
    name = "RIC"
    log(f"Generate {name}")
    dataSets[name] = generateDataset(
        baseDataset=baseDataSets["RIC"],
        taxons=None,  # all
        #sequenceLength=(50, 6000),  # Bressin 2019 length criteria
        annotationTypes=None  # do not matter for pre training
    )
    dataSets[name].positive = dataSets[name].positiveCount >= 1
    with open(DATA_SETS.joinpath(f"{name}.pkl"), 'wb') as f:
        pickle.dump(dataSets[name], f)

    # RIC pre-train (thr1)
    # - all RIC species but human, usable for pre-training
    name = "RIC_human_pre-training"
    log(f"Generate {name}")
    dataSets[name] = generateDataset(
        baseDataset=baseDataSets["RIC"],
        taxons=[9606],  # all but human
        allBut= True,
        sequenceLength=(50, 6000),  # Bressin 2019 length criteria (why not use here too?)
        annotationTypes=None  # do not matter for pre training
    )
    dataSets[name].positive = dataSets[name].positiveCount >= 1
    with open(DATA_SETS.joinpath(f"{name}.pkl"), 'wb') as f:
        pickle.dump(dataSets[name], f)

    # RIC human fine-tuning (thr1)
    name = "RIC_human_fine-tuning"
    log(f"Generate {name}")
    dataSets[name] = generateDataset(
        baseDataset=baseDataSets["RIC"],
        taxons=[9606],  # human
        sequenceLength=(50, 6000),  # Bressin 2019 length criteria (why not use here too?)
        annotationTypes=None  # do not matter for pre training
    )
    dataSets[name].positive = dataSets[name].positiveCount >= 1
    with open(DATA_SETS.joinpath(f"{name}.pkl"), 'wb') as f:
        pickle.dump(dataSets[name], f)


## InterPro ##
if("InterPro" in dataSetNames):
    # InterPro all
    # - all InterPro species, used for statistics
    name = "InterPro"
    log(f"Generate {name}")
    dataSets[name] = generateDataset(
        baseDataset=baseDataSets["InterPro"],
        taxons=None,  # all
        #sequenceLength=(50, 6000),  # Bressin 2019 length criteria
        annotationTypes=None  # do not matter for pre training
    )
    with open(DATA_SETS.joinpath(f"{name}.pkl"), 'wb') as f:
        pickle.dump(dataSets[name], f)

    # InterPro human pre-training
    # - more current version of bressin human?
    name = "InterPro_human_pre-training"
    log(f"Generate {name}")
    dataSets[name] = generateDataset(
        baseDataset=baseDataSets["InterPro"],
        taxons=[9606],  # all but human
        allBut=True, 
        sequenceLength=(50, 6000),  # Bressin 2019 length criteria (why not use on inter pro too?)
        annotationTypes=None  # do not matter for pre training
    )
    with open(DATA_SETS.joinpath(f"{name}.pkl"), 'wb') as f:
        pickle.dump(dataSets[name], f)

    # InterPro Human Fine-Tuning
    # - more current version of bressin human?
    name = "InterPro_human_fine-tuning"
    log(f"Generate {name}")
    dataSets[name] = generateDataset(
        baseDataset=baseDataSets["InterPro"],
        taxons=[9606],  # only human
        sequenceLength=(50, 6000),  # Bressin 2019 length criteria (why not use on inter pro too?)
        annotationTypes=None  # do not matter for pre training
    )
    with open(DATA_SETS.joinpath(f"{name}.pkl"), 'wb') as f:
        pickle.dump(dataSets[name], f)


## FOR OVERLAP ANALYSIS
if("RIC" in dataSetNames):
    # RIC human
    name = "RIC_human"
    log(f"Generate {name}")
    dataSets[name] = generateDataset(
        baseDataset=baseDataSets["RIC"],
        taxons=[9606],  # human
        annotationTypes=None  # do not matter for pre training
    )
    with open(DATA_SETS.joinpath(f"{name}.pkl"), 'wb') as f:
        pickle.dump(dataSets[name], f)

    # RIC human thr1
    # - all human RIC with positivity threshold 1
    name = "RIC_human_thr1"
    log(f"Generate {name}")
    dataSets[name] = dataSets["RIC_human_fine-tuning"].copy()
    dataSets[name].positive = dataSets[name].positiveCount >= 1
    with open(DATA_SETS.joinpath(f"{name}.pkl"), 'wb') as f:
        pickle.dump(dataSets[name], f)

    # RIC human th2
    # - for overlap analysis
    name = "RIC_human_thr2"
    log(f"Generate {name}")
    dataSets[name] = dataSets["RIC_human_fine-tuning"].copy()
    dataSets[name].positive = dataSets[name].positiveCount >= 2
    with open(DATA_SETS.joinpath(f"{name}.pkl"), 'wb') as f:
        pickle.dump(dataSets[name], f)

    # RIC human th3
    # - for overlap analysis
    name = "RIC_human_thr3"
    log(f"Generate {name}")
    dataSets[name] = dataSets["RIC_human_fine-tuning"].copy()
    dataSets[name].positive = dataSets[name].positiveCount >= 3
    with open(DATA_SETS.joinpath(f"{name}.pkl"), 'wb') as f:
        pickle.dump(dataSets[name], f)

