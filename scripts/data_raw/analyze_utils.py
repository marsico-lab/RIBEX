import pandas as pd
import numpy as np

# Initialize global environment and import useful utility functions 
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *
initialize(__file__)

# analze bressin/interpro positivity (bressin criteria) 
def analyze_UncertainPositivity(dataSet):
    """Print Positivity distribution.

    Makes mostly sense for datasets that can have ambiguous positivity (InterPro)

    Args:
        dataSet (_type_): _description_
    """
    total = len(dataSet)
    count = sum(pd.isnull(dataSet["positive"]) == True)
    log(f"\tuncertain positivity: {(count/total)*100:.4f} %\t ({count} of {total}) ")
    onlyBressinPos = sum(dataSet.positive == True)
    onlyBressinNeg = sum(dataSet.positive == False)
    bressinNone = sum(pd.isnull(dataSet.positive))
    log(f"\tbinding +: {(onlyBressinPos/total)*100:.2f}%\t ({onlyBressinPos})")
    log(f"\tbinding -: {(onlyBressinNeg/total)*100:.2f}%\t ({onlyBressinNeg})")
    log(f"\tbinding ?: {(bressinNone/total)*100:.2f}%\t ({bressinNone})")

#analyze balance of dataset
def analyze_Balance(dataSet):
    total = len(dataSet)
    pos = sum(dataSet.positive == True)
    neg = sum(dataSet.positive == False)
    none = sum(pd.isnull(dataSet.positive))
    log(f"\t\tbinding None: {(none/total)*100:.2f}%\t ({none})")
    log(f"\t\tbinding +: {(pos/total)*100:.2f}%\t ({pos})")
    log(f"\t\tbinding -: {(neg/total)*100:.2f}%\t ({neg})")
    log(f"\t\tbalance = 1 :  {(neg/pos):.2f}")

#analyze annotations
def analyze_Annotations(dataSet, taxon_ID=None):
    total = len(dataSet)
    counter = [0,0,0]
    for id in dataSet.index:
        
        annotations = dataSet.at[id, "annotations"]
        if(type(annotations) != str): #if its nan
            continue
        if(taxon_ID != None and dataSet.at[id, "taxon_ID"] != taxon_ID): #filtering for one species
            continue

        annotations = eval(annotations)
        counter_local = [0,0,0]
        for annotation in annotations:
            (fr, to, ty, name, sName) = annotation
            counter_local[ty] += 1
        
        counter[0] = counter[0]+1 if counter_local[0] > 0 else counter[0]
        counter[1] = counter[1]+1 if counter_local[1] > 0 else counter[1]
        counter[2] = counter[2]+1 if counter_local[2] > 0 else counter[2]
        
    log(f"\t\tHaving RBD: {(counter[1]/total)*100:.2f}%\t ({counter[1]})")
    log(f"\t\tHaving IDR: {(counter[2]/total)*100:.2f}%\t ({counter[2]})")
    log(f"\t\tHaving other: {(counter[0]/total)*100:.2f}%\t ({counter[0]})")

# analyzes RIC positive test counter
def analyze_RICpositivesCount(RICdataSet):
    """Print distribution of RIC postive tests count.

    Makes only sense for RIC dataset.

    Args:
        RICdataSet (_type_): _description_
    """
    total = len(RICdataSet)
    cum = 0
    log("\tpos\t#\t%\tcumulative\tcumulative%")
    for pos, count in np.transpose(np.unique(RICdataSet["positiveCount"], return_counts=True)):
        r = total - cum
        log(f"\t{pos}\t{count} \t{(count/total)*100:.3f}%\t{r}\t{(r/total)*100:.3f}%")
        cum += count

## General analysis
def analyze_general(dataSet):

    total = len(dataSet)
    log(f"\tTotal size: {total}")

    #analyze empty cells in all columns
    log(f"\tColumn analysis - empty ( None | '' | [] ) ")
    for columnKey in dataSet.keys():
        column = dataSet[columnKey]
        empty = np.sum( pd.isnull(column) ) + np.sum( column=="") #count nulls and empty strings
        for e in column: #count empty lists
            if type(e) == list and len(e) == 0:
                empty += 1
        log(f"\t\t{columnKey}:\t { (empty/total)*100 :.04f} %\t ({empty} of {total})")

    # RBP vs IDP analysis
    # neither_sum = np.sum(pd.isnull(dataSet.score_rel_1or2))
    # log(f"Proteins having RBDs or IDRs: {neither_sum}")
    # rbd_sum = np.sum(pd.isnull(dataSet.score_abs_1)==False)
    # log(f"Proteins having RBDs: {rbd_sum}")
    # idr_sum = np.sum(pd.isnull(dataSet.score_rel_2)==False)
    # log(f"Proteins having IDRs: {idr_sum}")
    # both_sum = np.sum(pd.isnull(dataSet.score_rel_1and2)==False)
    # log(f"Proteins having RBDs and IDRs: {both_sum}")
    # other_sum = np.sum(pd.isnull(dataSet.score_rel_0)==False)
    # log(f"Proteins having other: {other_sum}")
    # log(f"Proteins having only RBDs: {rbd_sum-both_sum}")
    # idr_sum = np.sum(pd.isnull(dataSet.score_rel_2)==False)
    # log(f"Proteins having only IDRs: {idr_sum-both_sum}")

    # for Unipord:
    # Proteins either RBDs or IDRs: 1205
    # Proteins having RBDs: 737
    # Proteins having IDRs: 1005
    # Proteins having both: 152
    # Proteins having other: 0
    # Proteins having only RBDs: 585
    # Proteins having only IDRs: 853

    # for InterPro
    # Total: 13606
    # Proteins having RBDs and IDRs: 11189
    # Proteins having RBDs or IDRs: 5484
    # Proteins having RBDs: 343
    # Proteins having IDRs: 8036
    # Proteins having RBDs and IDRs: 75
    # Proteins having other: 11189
    # Proteins having only RBDs: 268
    # Proteins having only IDRs: 7961
