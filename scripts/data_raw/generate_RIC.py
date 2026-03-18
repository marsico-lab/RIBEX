import numpy as np

# import torch
# import os
# import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# import json
# import requests  # for UniProd web API
from pprint import pprint

# import pickle  # to save some query that take long to execute
from multiprocessing import Pool, Lock  # for multithreading

# import time
from Bio import SeqIO  # to read fasta file

# import h5py

# Initialize global environment and import useful utility functions 
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *
initialize(__file__)

# get bressin definitions
from bressin_negativePfamTerms import PfamTerms as BressinPfamTerms
from bressin_negativeSetGOterms import GOterms as BressinGOterms

nonNegative_GOterms = BressinGOterms.GO_term[BressinGOterms.GO_term.notnull()].unique()
nonNegative_IPRterms = BressinPfamTerms.IPR[BressinPfamTerms.IPR.notnull()].unique()

# Get online request stuff
from generate_utils import getProtein_InterPro, getIDR_MobiDB, getSuffix
from generate_utils import (
    IPR_Protein_Cache,
    IPR_ProteinByName_Cache,
    IPR_Annotation_Cache,
    MobiDB_Cache,
)  # so we do not have to make requests again
from generate_utils import APIroot_InterPro as APIroot

# Dataset Notes:
# - Gene_Name, is always present for RIC (UNIQUE) e.g. TAAR9 but not the Gene ID (uniprod ID)
#   we try to get the uniprod ID from Inter Pro (getProtein_InterPro(byName=True))
#   but if that does not work, the Gene ID is simply None and the sample will be filtered out
# - we dont really know if the sequence is canonical, do we? (TODO)

# Why not use ENSMUSG column:
# - only present for human and mouse (missing for other species)
# - can have multiple entries
# - using a mapping service will only result in the UnitProtSwissProtID entries that are present anyways


RICcolumns = {  # file name (without extension) : (speciesID, [relevantColumns])
    # HUMAN - Homo sapiens
    "RBPbase_Hs_DescriptiveID": (
        9606,
        [
            "RBPBASE000000008.1",
            "RBPBASE000000009.1",
            "RBPBASE000000034.1",
            "RBPBASE000000035.1",
            "RBPBASE000000059.1",
            "RBPBASE000000060.1",
            "RBPBASE000000061.1",
            "RBPBASE000000062.1",
        ],
    ),
    # MOUSE - Mus musculus
    "RBPbase_Mm_DescriptiveID": (
        10090,
        [
            "RBPBASE000000014.1",
            "RBPBASE000000016.1",
            "RBPBASE000000017.1",
            "RBPBASE000000019.1",
            "RBPBASE000000053.1",
            "RBPBASE000000054.1",
            "RBPBASE000000055.1",
        ],
    ),
    # 9MUSC - Drosophila #TODO: is that the correct entry?
    "RBPbase_Dm_DescriptiveID": (7215, ["RBPBASE000000005.1", "RBPBASE000000006.1"]),
    # ARATH - Arabidopsis thaliana
    "RBPbase_At_DescriptiveID": (
        3702,
        ["RBPBASE000000001.1", "RBPBASE000000002.1", "RBPBASE000000003.1"],
    ),
    # 9HYME - Chrysis elegans #TODO: is that the correct entry?
    # "RBPbase_Ce_DescriptiveID": ( 212608, [ ])
    # YEAST - accharomyces cerevisiae (strain ATCC 204508 / S288c)
    "RBPbase_Sc_DescriptiveID": (559292, ["RBPBASE000000020.1" "RBPBASE000000024.1"]),
}


def preprocessRIC(filePath, RICcolumns):
    # reads all relevant informations from RIC file
    raw = pd.read_csv(str(filePath), sep="\t", header=0, encoding="latin-1")

    if not filePath.stem in RICcolumns.keys():
        raise RuntimeError(f'Unknown file stem for RIC files "{filePath.stem}" Known stems: [{list(RICcolumns.keys())}]')

    taxonID, relevantColumnNames = RICcolumns[filePath.stem]


    #Get gene Names
    Gene_Names_raw = raw["UNIQUE"]  # e.g. PUM2
    # fix gene name suffix if missing
    Gene_Names = []
    suffix = getSuffix(taxonID=taxonID)
    for Gene_Name in Gene_Names_raw:
        if len(suffix) > 0 and Gene_Name[-len(suffix) :] == suffix:
            Gene_Names.append(Gene_Name)
        else:
            Gene_Names.append(f"{Gene_Name}{suffix}")

    #Get Gene IDs (if uniprod ID column exists)
    match taxonID:
        case 9606: #"RBPbase_Hs_DescriptiveID.tsv":
            uniprodIDcolumnName = "UnitProtSwissProtID-Hs\nRBPANNO000000043.1"
        case 10090: #"RBPbase_Mm_DescriptiveID.tsv":
            uniprodIDcolumnName = "UnitProtSwissProtID-Mm\nRBPANNO000000044.1"
        case 559292: #"RBPbase_Sc_DescriptiveID.tsv":
            uniprodIDcolumnName = "UnitProtSwissProtID-Sc\nRBPANNO000000045.1"
        case _: #all other species do not have the default UnitProtSwissProtID
            uniprodIDcolumnName = None

    if(uniprodIDcolumnName == None):
        Gene_IDs = [None]*len(Gene_Names)
    else:
        Gene_IDs = raw[uniprodIDcolumnName]  # uniprod Gene ID e.g. Q8TB72 or "" or multiple seperated with "|" 

    # raw["ID"], e.g.  ENSG00000055917 for PUM2
    taxon_IDs = [taxonID] * len(raw)

    # Count Positive Tests

    #get relevant columns
    relevantColumnNames_real = []
    for realName in raw.keys():
        for targetName in relevantColumnNames:  # find relevant columns
            if targetName in realName:
                relevantColumnNames_real.append(realName)
    relevantColumnNames = relevantColumnNames_real

    counterVector = np.zeros(len(raw), dtype=int)
    for key in relevantColumnNames:
        columnData = raw[key] == "YES"

        counterVector += columnData

    return Gene_IDs, Gene_Names, taxon_IDs, counterVector


def process_getRIC(packed_parameters):
    (row, caches) = packed_parameters

    (
        IPR_Protein_Cache,
        IPR_ProteinByName_Cache,
        IPR_Annotation_Cache,
        MobiDB_Cache,
    ) = caches

    IPR_Annotation_Cache_update = {}
    IPR_Protein_Cache_update = {}
    IPR_ProteinByName_Cache_update = {}
    MobiDB_Cache_update = {}

    # Existing row information:
    assert pd.isnull(row["Gene_Name"]) == False
    assert pd.isnull(row["taxon_ID"]) == False
    assert pd.isnull(row["positiveCount"]) == False
    assert pd.isnull(row["positive"]) == False

    # Missing
    # Gene_ID (potentially)
    # Sequence
    # Annotations
    # Canonical (TODO)

    if pd.isnull(row["Gene_ID"]) or row["Gene_ID"] == "NA":  # if there is no unique identifier available
        # Nan or other pandas None types. Or more than one name (seperated with | )
        row["Gene_ID"] = None
    elif("|" in row["Gene_ID"]):
        log(f'\t[{row["Gene_Name"]}, {row["taxon_ID"]}]: multiple Gene_IDs: {row["Gene_ID"]}')
        row["Gene_ID"] = None


    # log(row)

    # Get protein information from InterPro
    sequence, IPannotations = None, None
    byName = row["Gene_ID"] == None
    if byName:  # no gene ID present, try to get by Gene Name
        ID = row["Gene_Name"]
    else:  # default: get infos by Gene ID
        ID = row["Gene_ID"]
    ret = getProtein_InterPro(
        ID,
        IPR_Annotation_Cache,
        IPR_Protein_Cache,
        IPR_ProteinByName_Cache,
        byName=byName,
    )

    if type(ret) == int:  # something went wrong, we just have the error message
        if ret == 204 or ret == 404 or (ret >= 500 and ret < 600):  # no data, not found or some server error
            if byName:
                IPR_ProteinByName_Cache_update[row["Gene_Name"]] = ret  # update cache with that error code
            else:
                IPR_Protein_Cache_update[row["Gene_ID"]] = ret  # update cache with that error code

            # if(ret == 404): #make extra warning for 404 (maybe type? wrong formatting)
            #    log(f"getProtein_InterPro({Gene_Name}): INFO: Gene_Name was not found (HTTP ERROR {ret})")
        else:  # print error and do not save (should be fixes by user!)
            log(f"getProtein_InterPro({row['Gene_Name']}): WARNING: HTTP ERROR: {ret}")
    else:
        (
            uniprodID,
            RBP_Name,
            taxon_ID,
            sequence,
            IPannotations,
            bressinPossibleNegative,
        ), IPR_Annotation_Cache_update = ret
        if byName:
            IPR_ProteinByName_Cache_update[row["Gene_Name"]] = (
                uniprodID,
                taxon_ID,
                sequence,
                IPannotations,
                bressinPossibleNegative,
            )
            #log(f'\tINFO: [{row["Gene_Name"]}, {row["taxon_ID"]}]: recovered Gene_ID: {row["Gene_ID"]} -> {uniprodID}')
            row["Gene_ID"] = uniprodID
        else:
            IPR_Protein_Cache_update[row["Gene_ID"]] = (
                RBP_Name,
                taxon_ID,
                sequence,
                IPannotations,
                bressinPossibleNegative,
            )
        
        #Assertions
        if(taxon_ID != row["taxon_ID"]):
            log(f'\tERROR: [{row["Gene_Name"]}, {row["taxon_ID"]}]: recovered taxon_ID = {taxon_ID} (mismatch to {row["taxon_ID"]} !!)')
            assert taxon_ID == row["taxon_ID"]

    row["sequence"] = sequence

    # Get annotations
    if IPannotations != None:
        row["annotations"] = []  # list of triplets: (from, to, type, name)
        row["annotations"].extend(IPannotations)  # type=0 and type=1 (RBD)

        # type=2 (IDR)
        ret = getIDR_MobiDB(row["Gene_Name"] if byName else row["Gene_ID"], MobiDB_Cache, taxon_ID)  # get MobiDBannotations
        if type(ret) == int:  # something went wrong, we just have the HTTP error code
            if (
                ret == 204 or ret >= 500 and ret < 600
            ):  # no data or some server error, this normally means that they do not have this protein in their database
                if byName:  # update cache with that error code
                    MobiDB_Cache_update[row["Gene_Name"]] = ret
                else:
                    MobiDB_Cache_update[row["Gene_ID"]] = ret
            else:  # print error and do not save (should be fixes by user!)
                log(f"MobiDB_getIDR({row['Gene_ID']}): WARNING: HTTP ERROR: {ret}")
        else:
            if byName:  # update cache with that error code
                MobiDB_Cache_update[row["Gene_Name"]] = ret
            else:
                MobiDB_Cache_update[row["Gene_ID"]] = ret
            for IDR in ret:  # convert to our annotation format
                fr, to = IDR
                row["annotations"].append((fr, to, 2, "IDR", "MobiDB-lite IDR"))

    # Output
    cache_updates = (
        IPR_Protein_Cache_update,
        IPR_ProteinByName_Cache_update,
        IPR_Annotation_Cache_update,
        MobiDB_Cache_update,
    )
    return row, cache_updates


def getRIC(
    outputFilePath,
    folderPath,
    thresholdPos=3,  # if x columns are positive
    forceRefresh=False,
    silent=True,
    threads=16,
    bufferSize=1000,  # buffered rows until write to filesystem
):
    global RICcolumns
    global IPR_Protein_Cache, IPR_ProteinByName_Cache, IPR_Annotation_Cache, MobiDB_Cache  # get all the caches

    RICfiles = [  # Filename, taxonID
        ("RBPbase_Hs_DescriptiveID.tsv", 9606),  # HUMAN - Homo sapiens (1914 RBPs)
        ("RBPbase_Mm_DescriptiveID.tsv", 10090),  # MOUSE - Mus musculus (1393 RBPs)
        ("RBPbase_Sc_DescriptiveID.tsv", 559292),  # YEAST - Saccharomyces cerevisiae (strain ATCC 204508 / S288c) (1393 RBPs)
        # ab hier KEINE UnitProtSwissProtID column mehr!
        ("RBPbase_Dm_DescriptiveID.tsv", 7227),  # 9MUSC - Drosophila (777 RBPs) 
        ("RBPbase_At_DescriptiveID.tsv", 3702),  # ARATH - Arabidopsis thaliana (719 RBPs)
        # (
        #    "RBPbase_Ce_DescriptiveID.tsv", #TODO: Marc fragen was für columns hier relevant sind, da fehlt mir nämlich die referenz datei!
        #    212608,
        # ),  # 9HYME - Chrysis elegans #TODO: this is a parent (593 RBPs)
    ]

    Gene_IDs, Gene_Names, taxon_IDs, positiveCounts = [], [], [], []
    for fileName, taxonID in RICfiles:
        log(f"\tPre-Processing {fileName}")
        filePath = folderPath.joinpath(fileName)
        Gene_IDs_file, Gene_Names_file, taxon_IDs_file, positiveCounts_file = preprocessRIC(filePath, RICcolumns)  # offline stuff
        # log(f"SET:{set(pd.unique(Gene_IDs_file))}")
        # for name in list(set(Gene_IDs_file)):
        #    log(name)
        Gene_IDs.extend(Gene_IDs_file)
        Gene_Names.extend(Gene_Names_file)
        taxon_IDs.extend(taxon_IDs_file)
        positiveCounts.extend(positiveCounts_file)
        log(f"\t-> {len(Gene_IDs_file)} rows extracted")

    # read save file
    if forceRefresh or outputFilePath.exists() == False:
        if not silent:
            log(f"\tGenerating new empty output file {outputFilePath}")
        df = pd.DataFrame(
            {
                "Gene_ID": [],
                "Gene_Name": [],
                "taxon_ID": [],
                "positiveCount": [],
                "canonical": [],
                "positive": [],
                "annotations": [],
                "sequence": [],
            }
        )
        # RBPs.set_index("Gene_ID", inplace=True) #do not use Gene_ID as index
        df.to_csv(outputFilePath, sep="\t", header=True, mode="w", index=False)  # creates new TSV
    else:
        if not silent:
            log(f"\tLoading data from {outputFilePath}")
        df = pd.read_csv(outputFilePath, sep="\t")

    # check existing rows
    required = set(Gene_Names)
    existing = set(df.Gene_Name)
    missingNames = required - existing  # todo = required - existing
    log(f"\t{len(existing)} of {len(required)} ({ (len(existing)/len(required))*100 :.06f} %) rows exist in {outputFilePath}")

    # generate missing
    missingNames = list(missingNames)

    if len(missingNames) > 1:
        # setup buffer
        df_dict_buffer = {
            "Gene_ID": [],
            "Gene_Name": [],
            "taxon_ID": [],
            "positiveCount": [],
            "canonical": [],
            "positive": [],
            "annotations": [],
            "sequence": [],
        }

        # Generate
        log("\tGenerating")
        with tqdm(total=len(missingNames)) as pbar:
            for batch_index in range(0, len(missingNames), threads):  # work in batches
                # batch = Gene_IDs[batch_index : batch_index + threads]
                batch_Gene_Names = missingNames[batch_index : batch_index + threads]

                # create input data
                threadData = []
                for Gene_Name in batch_Gene_Names:
                    # pack existing information
                    preprocessing_index = Gene_Names.index(Gene_Name)  # get index of list where Gene_Name is
                    row = {
                        "Gene_ID": Gene_IDs[preprocessing_index],
                        "Gene_Name": Gene_Name,
                        "taxon_ID": taxon_IDs[preprocessing_index],
                        "positiveCount": positiveCounts[preprocessing_index],
                        "sequence": None,
                        "annotations": None,
                        "canonical": None,
                        "positive": positiveCounts[preprocessing_index] >= thresholdPos,
                    }

                    # caches
                    caches = (
                        IPR_Protein_Cache,
                        IPR_ProteinByName_Cache,
                        IPR_Annotation_Cache,
                        MobiDB_Cache,
                    )

                    # append data
                    threadData.append((row, caches))

                # execute parallel processes
                with Pool(len(threadData)) as p:
                    returnData = p.map(process_getRIC, threadData)

                # extract output data
                for dataSet in returnData:  # the return data ordering might be different from the thread data ordering
                    # log(f"ret: {dataSet}")
                    (row, cache_updates) = dataSet

                    # set row data
                    for key in row.keys():
                        df_dict_buffer[key].append(
                            row[key]
                        )  # Gene_ID, Gene_Name, taxon_ID, sequence, positiveCount, annotations, canonical, positive

                    # update caches
                    (
                        IPR_Protein_Cache_update,
                        IPR_ProteinByName_Cache_update,
                        IPR_Annotation_Cache_update,
                        MobiDB_Cache_update,
                    ) = cache_updates
                    IPR_Protein_Cache.update(IPR_Protein_Cache_update)
                    IPR_ProteinByName_Cache.update(IPR_ProteinByName_Cache_update)
                    IPR_Annotation_Cache.update(IPR_Annotation_Cache_update)
                    MobiDB_Cache.update(MobiDB_Cache_update)

                # flush buffer if full
                if len(df_dict_buffer["Gene_ID"]) >= bufferSize:
                    l = len(df_dict_buffer["Gene_ID"])
                    # log(f"flush buffer with {l} rows to {outputFilePath}")
                    df_buffer = pd.DataFrame(df_dict_buffer)
                    df_buffer.to_csv(outputFilePath, sep="\t", header=False, mode="a", index=False)  # append to TSV

                    # empty buffer
                    for key in df_dict_buffer.keys():
                        df_dict_buffer[key] = []

                pbar.update(len(batch_Gene_Names))

            # final flush buffer if full
            if len(df_dict_buffer["Gene_ID"]) > 0:
                l = len(df_dict_buffer["Gene_ID"])
                log(f"flush buffer with {l} rows to {outputFilePath}")
                df_buffer = pd.DataFrame(df_dict_buffer)
                df_buffer.to_csv(outputFilePath, sep="\t", header=False, mode="a", index=False)  # append to TSV

    # read csv
    df = pd.read_csv(outputFilePath, sep="\t")

    return df


if __name__ == "__main__":
    log("Getting RIC")
    RIC = getRIC(
        folderPath=DATA_ORIGINAL.joinpath("RIC/"),
        outputFilePath=DATA_RAW.joinpath("RIC.tsv"),
        thresholdPos=thresholdPos,  # if three columns are positive
        forceRefresh=False,
        silent=True,
        threads=8,#webWorkerThreads,
        bufferSize=bufferSize,
    )
