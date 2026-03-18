# from collections.abc import Callable, Iterable, Mapping
from typing import Any
import numpy as np

# import torch
# import os
# import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import json
import requests  # for UniProd web API
from pprint import pprint

# import pickle  # to save some query that take long to execute
from multiprocessing import Pool, Lock, Process  # for multithreading

# import threading
# import time
# from Bio import SeqIO  # to read fasta file

# Get global Parameters
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
from generate_utils import getProtein_InterPro, getIDR_MobiDB
from generate_utils import (
    IPR_Protein_Cache,
    IPR_Annotation_Cache,
    MobiDB_Cache,
)  # so we do not have to make requests again
from generate_utils import APIroot_InterPro as APIroot

# Dataset Notes:
# - We do not really know for all samples whether the sequence is canonical (TODO). But I think we
#   can assume it!
# - RBP positivity flag is not always unambiguous! These samples are "None" in column "positive"


# Original datasets from https://www.ebi.ac.uk/interpro/protein/reviewed/entry/InterPro/#table with:
# - reviewed
# - complete sequence ( ?is_fragment=false )
# - some species ( taxonomy/uniprot/9606 )
# -> export as JSON
# - Extra fields: go_terms
# Human (9606):             https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/InterPro/taxonomy/uniprot/9606/?extra_fields=go_terms
# Mouse (10090):            https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/InterPro/taxonomy/uniprot/10090/?extra_fields=go_terms
# Arab. Thaliana (3702):    https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/InterPro/taxonomy/uniprot/3702/?extra_fields=go_terms
# E.Coli (83333):           https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/InterPro/taxonomy/uniprot/83333/?extra_fields=go_terms
# Drosophila (7227):        https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/InterPro/taxonomy/uniprot/7227/?extra_fields=go_terms
# Last downloaded: 2023-07-07
# Salmonella (590):         https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/InterPro/taxonomy/uniprot/590/?extra_fields=go_terms
# Last downloaded: 2023-07-21

# gets all proteins that were downloaded with or without annotated RBP binding
# ATTENTION: the molecule wide GO annotations seem to be too few, therefore we update that flag later
# (probably because most of molecule wide GO of RBP are simply missing in Inter Pro for some reason)
def preprocessInterPro(filePath):
    Gene_IDs = []
    taxon_IDs = []
    positives_genomewide = []  # needs to be updated later! (because that is sometimes not present!)
    # missing: Gene_Names, sequences, (positives), annotations [not actually, but we get this afterwards anyways], canonical (TODO)

    with open(filePath, "r") as f:
        file_dict = json.load(f)  # list of dicts?

    for protein_dict in file_dict:
        # Get gene ID
        Gene_ID = protein_dict["metadata"]["accession"]

        # get taxon ID
        taxon_ID = protein_dict["metadata"]["source_organism"]["taxId"]
        taxon_ID2 = protein_dict["taxa"][0]["accession"]
        if len(protein_dict["taxa"]) > 1:
            raise RuntimeError(f"\t[{Gene_ID}]: more than 1 'taxa': {[d['accession'] for d in protein_dict['taxa']]}")
        if taxon_ID != taxon_ID2:
            raise RuntimeError(f"\t[{Gene_ID}]: taxon ID mismatch: {taxon_ID} VS {taxon_ID2}")
        else:
            taxon_ID = int(taxon_ID)

        # get positivity only if positive
        positive_genomewide = None
        go_terms = protein_dict["extra_fields"]["go_terms"]
        if go_terms != None and len(go_terms) > 0:
            for go_term in go_terms:
                if go_term["identifier"] == "GO:0003723":
                    positive_genomewide = True
                    break

        # append protein data
        Gene_IDs.append(Gene_ID)
        taxon_IDs.append(taxon_ID)
        positives_genomewide.append(positive_genomewide)

    return Gene_IDs, taxon_IDs, positives_genomewide


def process_getInterPro(packed_parameters):
    (row, caches) = packed_parameters

    IPR_Protein_Cache, IPR_Annotation_Cache, MobiDB_Cache = caches

    IPR_Annotation_Cache_update = {}
    IPR_Protein_Cache_update = {}
    MobiDB_Cache_update = {}

    # Existing row information:
    assert pd.isnull(row["Gene_ID"]) == False
    assert pd.isnull(row["taxon_ID"]) == False
    #assert pd.isnull(row["positive_genomewide"]) == False #might also ne None...

    # Missing:
    # Gene Name
    # sequence
    # positive
    # Annotations
    # Canonical (TODO)

    # Get protein information from InterPro
    RBP_Name = row.get("Gene_Name")           
    sequence = row.get("sequence")
    taxon_ID = row["taxon_ID"]
    IPannotations = None
    bressinPossibleNegative = False
    if row.get("annotations") is None:
        row["annotations"] = []               
        
    
    ret = getProtein_InterPro(row["Gene_ID"], IPR_Annotation_Cache, IPR_Protein_Cache, IPR_ProteinByName_Cache=None)

    if type(ret) == int:  # something went wrong, we just have the HTTP error code
        if ret == 204 or (ret >= 500 and ret < 600):  # no data or some server error
            IPR_Protein_Cache_update[row["Gene_ID"]] = ret  # update cache with that error code
        else:  # print error and do not save (should be fixes by user!)
            log(f"getProtein_InterPro({row['Gene_ID']}): WARNING: HTTP ERROR: {ret}")
    else:
        (uniprodID, RBP_Name, taxon_ID, sequence, IPannotations, bressinPossibleNegative), IPR_Annotation_Cache_update = ret
        IPR_Protein_Cache_update[row["Gene_ID"]] = (RBP_Name, taxon_ID, sequence, IPannotations, bressinPossibleNegative)

    # Log if we still don't have a name/sequence (error path or empty result)
    if RBP_Name is None:
        log(f"process_getInterPro({row['Gene_ID']}): INFO: RBP_Name not set (leaving as None)")
    if sequence is None:
        log(f"process_getInterPro({row['Gene_ID']}): INFO: sequence not set (leaving as None)")



    row["Gene_Name"] = RBP_Name
    row["sequence"] = sequence

    # Get annotations
    if IPannotations != None:
        row["annotations"] = []  # list of triplets: (from, to, type, name)
        row["annotations"].extend(IPannotations)  # type=0 and type=1 (RBD)

        # type=2 (IDR)
        ret = getIDR_MobiDB(row["Gene_ID"], MobiDB_Cache, taxon_ID)  # get MobiDBannotations
        if type(ret) == int:  # something went wrong, we just have the HTTP error code
            if (
                ret == 204 or ret >= 500 and ret < 600
            ):  # no data or some server error, this normally means that they do not have this protein in their database
                MobiDB_Cache_update[row["Gene_ID"]] = ret  # update cache with that error code
            else:  # print error and do not save (should be fixes by user!)
                log(f"MobiDB_getIDR({row['Gene_ID']}): WARNING: HTTP ERROR: {ret}")
        else:
            MobiDB_Cache_update[row["Gene_ID"]] = ret
            for IDR in ret:  # convert to our annotation format
                fr, to = IDR
                row["annotations"].append((fr, to, 2, "IDR", "MobiDB-lite IDR"))

    # Check Bressin binding set membership
    # Positive Set: member if molecule wide "GO:0003723" exists or any of the annotations is binding
    # Negative Set: Member if no go term or IDP term indicates non binding behavior (bressin criteria)

    # binding domain in annotations?
    hasInterProBindingDomain = False
    for fr, to, ty, sName, name in row.get("annotations", []):
        hasInterProBindingDomain = True
        break
    
    # get positive binding
    positive = None
    if(row["positive_genomewide"]): #if InterPro genome wide annotation is binding+ -> binding+
        positive = True
        #Bressin19: positive definition
    elif hasInterProBindingDomain and bressinPossibleNegative==False: # if some binding domain and not in negative set -> binding+
        positive = True
        #Bressin19: not sure?
    elif hasInterProBindingDomain == False and bressinPossibleNegative:
        positive = False
        #Bressin19: negative definition
        # Bressin negatives must have: 50-6000len, no global binding, no local/potential binding (Pfam)
    elif hasInterProBindingDomain and bressinPossibleNegative :
        # if there is a binding domain but bressin does not exclude it from the negative set -> ???
        positive = None #this case makes up 50% of the dataset!
    elif hasInterProBindingDomain==False and bressinPossibleNegative==False:
        # there is no intermedite indication that the protein is binding but it is not in the negative set
        #log(f"\t\t[{row['Gene_ID']}, {row['taxon_ID']}): INFO: protein has no indication (InterPro) of binding and but is excluded from negative set (bressin)!")
        positive = None #we dont know if its binding or not...
    else:
        raise NotImplementedError("All cases should be handeled explicitly!")

    row["positive"] = positive

    # Output
    cache_updates = IPR_Protein_Cache_update, IPR_Annotation_Cache_update, MobiDB_Cache_update
    return row, cache_updates


# gets dataset from InterPro
def getInterPro(
    folderPath,
    outputFilePath,
    forceRefresh=False,
    silent=True,
    threads=16,
    bufferSize=1000,
):
    global IPR_Protein_Cache, IPR_Annotation_Cache, MobiDB_Cache  # get all the caches

    log("\tPreprocessing Files")
    InterProFiles = [  # Filename
        ("HUMAN_9606_rev_complete_extra-go-terms.json"),  # HUMAN - Homo sapiens
        ("MOUSE_10090_rev_complete_extra-go-terms.json"),  # MOUSE - Mus musculus
        ("DROS_7227_rev_complete_extra-go-terms.json"),  # DROS - Drosophila Melanogaster
        ("ECOLI_83333_rev_complete_extra-go-terms.json"),  # E. COLI - Escherichia Coli
        ("ARATH_3702_rev_complete_extra-go-terms.json"),  # ARATH - Arabidopsis thaliana
        ("SAL_590_rev_complete_extra-go-terms.json"),  # SAL - Salmonella
    ]

    Gene_IDs, taxon_IDs, positives_genomewide = [], [], []
    for fileName in InterProFiles:
        log(f"\tPre-Processing {fileName}")
        filePath = folderPath.joinpath(fileName)
        Gene_IDs_file, taxon_IDs_file, positives_file = preprocessInterPro(filePath)  # offline stuff
        Gene_IDs.extend(Gene_IDs_file)
        taxon_IDs.extend(taxon_IDs_file)
        positives_genomewide.extend(positives_file)
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
                "canonical": [],
                "positive_genomewide": [],
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
    required = set(Gene_IDs)
    existing = set(df.Gene_ID)
    missingNames = required - existing  # todo = required - existing

    log(f"\t{len(required)-len(missingNames)} of {len(required)} ({ 100-(len(missingNames)/len(required))*100 :.06f} %) rows exist in {outputFilePath}")

    # generate missing
    missingNames = list(missingNames)

    if len(missingNames) > 1:
        # setup buffer
        df_dict_buffer = {
            "Gene_ID": [],
            "Gene_Name": [],
            "taxon_ID": [],
            "positive_genomewide": [],
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
                batch_Gene_IDs = missingNames[batch_index : batch_index + threads]

                # create input data
                threadData = []
                for Gene_Name in batch_Gene_IDs:
                    # pack existing information
                    preprocessing_index = Gene_IDs.index(Gene_Name)  # get index of list where Gene_Name is
                    row = {
                        "Gene_ID": Gene_IDs[preprocessing_index],
                        "Gene_Name": None,
                        "taxon_ID": taxon_IDs[preprocessing_index],
                        "positive_genomewide": positives_genomewide[preprocessing_index],
                        "positive": None,
                        "sequence": None,
                        "annotations": None,
                        "canonical": None,
                    }

                    # caches
                    caches = IPR_Protein_Cache, IPR_Annotation_Cache, MobiDB_Cache

                    # append data
                    threadData.append((row, caches))

                # execute parallel processes
                with Pool(len(threadData)) as p:
                    returnData = p.map(process_getInterPro, threadData)

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
                        IPR_Annotation_Cache_update,
                        MobiDB_Cache_update,
                    ) = cache_updates
                    IPR_Protein_Cache.update(IPR_Protein_Cache_update)
                    IPR_Annotation_Cache.update(IPR_Annotation_Cache_update)
                    MobiDB_Cache.update(MobiDB_Cache_update)

                # flush buffer if full
                if len(df_dict_buffer["Gene_ID"]) >= bufferSize:
                    l = len(df_dict_buffer["Gene_ID"])
                    # log(f"f\tlush buffer with {l} rows to {outputFilePath}")
                    df_buffer = pd.DataFrame(df_dict_buffer)
                    df_buffer.to_csv(outputFilePath, sep="\t", header=False, mode="a", index=False)  # append to TSV

                    # empty buffer
                    for key in df_dict_buffer.keys():
                        df_dict_buffer[key] = []

                pbar.update(len(batch_Gene_IDs))

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
    log("Getting InterPro")
    InterPro = getInterPro(
        folderPath=DATA_ORIGINAL.joinpath(f"InterPro/"),
        outputFilePath=DATA_RAW.joinpath("InterPro.tsv"),
        forceRefresh=False,
        silent=False,
        threads=webWorkerThreads,  # this is mostly web server heavy therefore we can use more CPU power
        # nevertheless we consume roughly 10 GB RAM per 100 workers
        # also: your ISP (i.e. MWN/LRZ) might not like so many web queries per minute and temporarily blocks your access...
        bufferSize=bufferSize,
    )
