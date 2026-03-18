
import numpy as np
import pandas as pd
from tqdm import tqdm

# Initialize global environment and import useful utility functions 
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *

def filterProteins(datasetRaw):
    """Filter our protein entries from raw datasset that do not fulfill minimal requirements.

    Requirements are:
    - has gene_ID
    - has taxon_ID
    - has sequence
    - is either RBP or non-RBP (some might be ambiguous)

    Args:
        datasetRaw (_type_): Dataset which is to be processed
        datasetName (str, optional): Name of dataset for better logging. Defaults to "Dataset".

    Returns:
        _type_: Filtered dataset
    """

    # Collect issues for each protein
    filteredIDs = []
    issueCounter = {
        "No gene ID": 0,
        "No taxon ID": 0,
        "No sequence": 0,
        # "No annotations": 0, #RIC might not have annotations!
        "No certain positive flag": 0,
    }

    Gene_ID_list = list(datasetRaw.Gene_ID)
    for i in tqdm(datasetRaw.index):

        # Check Gene ID
        Gene_ID = datasetRaw.at[i, "Gene_ID"]
        
        if pd.isnull(Gene_ID) == True:
            issueCounter["No gene ID"] += 1
            filteredIDs.append(i)
        else:
            occurance = Gene_ID_list.count(Gene_ID)
            #if occurance > 1:  # check if there are multiple occurrences of this gene ID
            #    log(f"\tWARNING: {Gene_ID} occurs {occurance} times!")

        # Check Gene Name (e.g. PUM2)
        #Gene_Name = datasetRaw.at[i, "Gene_Name"]
        # -> this is purely optional, so no checks required

        # Check taxon_ID (i.e. 9606)
        taxon_ID = datasetRaw.at[i, "taxon_ID"]
        if pd.isnull(taxon_ID) == True:
            issueCounter["No taxon ID"] += 1
            filteredIDs.append(i)
        # -> at this pipeline stage we do not filter for specific taxons, we just make sure an ID exists

        # Check sequence
        sequence = datasetRaw.at[i, "sequence"]
        if pd.isnull(sequence) == True or sequence == "":
            issueCounter["No sequence"] += 1
            filteredIDs.append(i)
        # -> at this pipeline stage we do not filter for specific seq length

        # Check annotation # triplet: (from, to, type)
        # annotations = datasetRaw.at[i, "annotations"]
        # if pd.isnull(annotations) == True or annotations == [] or annotations == [None]:
        #    issueCounter["No annotations"] += 1
        #    filteredIDs.append(i)
        # -> actually, RIC data might not have annotations and is still relevant for us!

        # Check canonical flag
        # TODO: as we do not know wheter the inter Pro are really canonical, I do not know what
        # to check for

        # Check positive flag
        positive = datasetRaw.at[i, "positive"]
        if pd.isnull(positive) == True or positive == []:
            issueCounter["No certain positive flag"] += 1
            filteredIDs.append(i)

    # Remove samples with issues
    filteredIDs = np.unique(filteredIDs)
    total = len(datasetRaw)
    log(f"\t\tRaw entries: {total}")
    datasetFiltered = datasetRaw.drop(index=filteredIDs)
    dropped = len(filteredIDs)
    log(f"\t\tDropped {dropped} ({(dropped/total)*100:.2f} %) entries")
    # Report issues
    log(f"\t\tIssues were:")
    for issueName in issueCounter:
        issueCount = issueCounter[issueName]
        log(f"\t\t\t{issueName}: {issueCount} ({(issueCount/total)*100:.2f}%)")

    #Remaining
    filtered = len(datasetFiltered)
    log(f"\t\tRemaining entries: {len(datasetFiltered)} ({(filtered/total)*100:.2f}%)")

    return datasetFiltered


def generateDataset(baseDataset, taxons=None, allBut=False, sequenceLength=None, annotationTypes=None):
    """Filters dataset on more specific criteria.

    These might be:
    - specific taxons/species
    - specific sequence length
    - specific annotations that proteins mut have at least one of

    Args:
        baseDatasets: Where to take the protein from.
        taxons (_type_, optional): WHich species to use (nor to not use, see "allBut"). None for all in base set. Defaults to None.
        allBut (bool, optional): If allBut is set to True, all taxons will be used EXCEPT what is specified by the taxons agrument
        sequenceLength (_type_, optional): (from, to) range of sequence lengths. None for all. Defaults to None.
        annotationTypes (_type_, optional): list of annotation types where one must at least exist per protein, 0=other, 1=RBD, 2=IDR. Defaults to None.

    Returns:
        _type_: Filtered dataset

    """
    subset = baseDataset

    # filter for Taxon IDs
    if taxons != None:
        log(f"\tFiltering for taxons: {taxons}")
        if(allBut): # -> all taxons except taxons list
            selectorList = [not (taxon_ID in taxons) for taxon_ID in subset.taxon_ID]
        else: # -> only taxons list
            selectorList = [(taxon_ID in taxons) for taxon_ID in subset.taxon_ID]
        subset = subset.loc[selectorList]
        before, after = len(selectorList), sum(selectorList)
        log(f"\t\t{before} -> {after} ({(after/before)*100:.2f}%)")

    # filter for sequence length
    if sequenceLength != None:
        log(f"\tFiltering for sequence length range: {sequenceLength}")
        selectorList = []
        for sequence in subset.sequence:
            select = True
            if sequenceLength[0] != None and len(sequence) < sequenceLength[0]:  # lower limit
                select = False
            if sequenceLength[1] != None and len(sequence) > sequenceLength[1]:  # upper limit
                select = False
            selectorList.append(select)
        subset = subset.loc[selectorList]
        before, after = len(selectorList), sum(selectorList)
        log(f"\t\t{before} -> {after} ({(after/before)*100:.2f}%)")

    # filter for annotation type
    if annotationTypes != None:
        log(f"\tFiltering for annotation types: {annotationTypes}")
        selectorList = []
        for annotations in subset.annotations:
            select = False
            for annotation in annotations:
                fr, to, ty, name, sName = annotation
                if ty in annotationTypes:  # found one fitting annotation
                    select = True
                    break
            if(select == False):
                print(f"\t\tremove: {annotation}")
            selectorList.append(select)
        subset = subset.loc[selectorList]
        before, after = len(selectorList), sum(selectorList)
        log(f"\t\t{before} -> {after} ({(after/before)*100:.2f}%)")

    return subset

