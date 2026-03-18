import pickle

# Initialize global environment and import useful utility functions 
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *
initialize(__file__)

from analyze_utils import *
from natsort import natsorted

#Data set columns/keys: "Gene_ID", "Gene_Name", "taxon_ID", "canonical", "positive", "annotations", "sequence"
dataSets = {}
for filePath in list(DATA_SETS.glob("*")):
    name = filePath.stem
    with open(filePath, 'rb') as f:
        dataSets[name] = pickle.load(f)

log(f"Loaded datasets: {list(dataSets.keys())}")

##Analyze properties

log(f"General Analysis")
for dataSetName in natsorted(list(dataSets.keys())):

    log(f"\t{dataSetName}")
    dataSet = dataSets[dataSetName]

    #analyze balance of dataset
    log(f"\t\tBalance")
    analyze_Balance(dataSet)

    #analyze annotations
    log(f"\t\tAnnotations")
    analyze_Annotations(dataSet)


## All analysis

n1 = 'RIC'
n2 = 'bressin19'
n3 = 'InterPro'
plotOverlap3(dataSets, n1, n2, n3)


## Pretrain analysis
n1 = 'RIC_human_pre-training'
n2 = 'bressin19_human_pre-training'
n3 = 'InterPro_human_pre-training'
plotOverlap3(dataSets, n1, n2, n3)


## Human analysis
n1 = 'RIC_human_fine-tuning'
n2 = 'bressin19_human_fine-tuning'
n3 = 'InterPro_human_fine-tuning'
plotOverlap3(dataSets, n1, n2, n3)

plotOverlapLists3(
    dataSets[n1].loc[dataSets[n1].positive == True]["Gene_ID"],
    dataSets[n2].loc[dataSets[n2].positive == True]["Gene_ID"],
    dataSets[n3].loc[dataSets[n3].positive == True]["Gene_ID"],
    n1, n2, n3,
    suffix=f"[RBP+]"
)


## Threshold analysis
n2 = 'bressin19_human_fine-tuning'
n3 = 'InterPro_human_fine-tuning'

n1 = 'RIC_human_thr1'
plotOverlapLists3(
    dataSets[n1].loc[dataSets[n1].positive == True]["Gene_ID"],
    dataSets[n2].loc[dataSets[n2].positive == True]["Gene_ID"],
    dataSets[n3].loc[dataSets[n3].positive == True]["Gene_ID"],
    n1, n2, n3,
    suffix=f"[RBP+]"
)

n1 = 'RIC_human_thr2'
plotOverlapLists3(
    dataSets[n1].loc[dataSets[n1].positive == True]["Gene_ID"],
    dataSets[n2].loc[dataSets[n2].positive == True]["Gene_ID"],
    dataSets[n3].loc[dataSets[n3].positive == True]["Gene_ID"],
    n1, n2, n3,
    suffix=f"[RBP+]"
)

n1 = 'RIC_human_thr3'
plotOverlapLists3(
    dataSets[n1].loc[dataSets[n1].positive == True]["Gene_ID"],
    dataSets[n2].loc[dataSets[n2].positive == True]["Gene_ID"],
    dataSets[n3].loc[dataSets[n3].positive == True]["Gene_ID"],
    n1,n2,n3,
    suffix=f"[RBP+]"
)


# Thr       1   2   3   4   5   6   7   8
# of RIC    17  23  27  30  33  37  41  45
# of InterP 73  77  60  55  47  40  33  6
# total     16  21  23  24  24  24  22  5


## IDEAS:

# TODO: sequence length visualization (see Task1_data_preparation.ipynb)
# TODO: check for intersection between pretrain and trarget set (see Task1_data_preparation.ipynb)
#       on uniprodID basis? on sequence basis? or even on species basis (should be trivial?)