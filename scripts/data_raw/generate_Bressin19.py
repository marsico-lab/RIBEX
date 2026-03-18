import pandas as pd

# from pprint import pprint
from multiprocessing import Pool  # for multithreading
from tqdm import tqdm


# Get global Parameters
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *
initialize(__file__)

from generate_utils import getProtein_InterPro, getIDR_MobiDB, getSuffix
from generate_utils import IPR_Protein_Cache, IPR_Annotation_Cache, MobiDB_Cache  # so we do not have to make requests again

# Dataset Notes:
# - the original set Peng et al. 19 used


def preprocessBressin19(fileName, taxonID):
    Gene_IDs = []
    Gene_Names = []
    taxon_IDs = []
    sequences = []
    # missing: positives (done in loop above)

    with open(fileName, "r") as f:
        lines = f.readlines()

    for description, sequence in list(zip(lines[::2], lines[1::2])):
        sequence = sequence[:-1]
        description = description[:-1]
        split = description.split("|")

        uniprodID = split[1]

        # append missing "_HUMAN"
        # fix gene name suffix if missing
        Gene_Name = split[2]

        Gene_IDs.append(uniprodID)
        Gene_Names.append(Gene_Name)
        taxon_IDs.append(taxonID)
        sequences.append(str(sequence))

    # reformat gene names
    suffix = getSuffix(taxonID=taxonID)
    len_suffix = len(suffix)
    if len_suffix > 0:
        for i, Gene_Name in enumerate(Gene_Names):
            if Gene_Name[-len_suffix:] != suffix:  # suffix not present
                Gene_Names[i] = f"{Gene_Name}{suffix}"

    return Gene_IDs, Gene_Names, taxon_IDs, sequences


def process_getBressin19(packed_parameters):
    (row, caches) = packed_parameters

    IPR_Protein_Cache, IPR_Annotation_Cache, MobiDB_Cache = caches

    IPR_Annotation_Cache_update = {}
    IPR_Protein_Cache_update = {}
    MobiDB_Cache_update = {}

    # Existing row information:
    assert pd.isnull(row["Gene_ID"]) == False
    assert pd.isnull(row["Gene_Name"]) == False
    assert pd.isnull(row["taxon_ID"]) == False
    assert pd.isnull(row["sequence"]) == False
    assert pd.isnull(row["positive"]) == False

    # Missing:
    # Annotations
    # Canonical (TODO)

    # Get protein information from InterPro
    IPannotations = None
    ret = getProtein_InterPro(row["Gene_ID"], IPR_Annotation_Cache, IPR_Protein_Cache, IPR_ProteinByName_Cache=None)

    if type(ret) == int:  # something went wrong, we just have the HTTP error code
        if ret == 204 or (ret >= 500 and ret < 600):  # no data or some server error
            IPR_Protein_Cache_update[row["Gene_ID"]] = ret  # update cache with that error code
        else:  # print error and do not save (should be fixes by user!)
            log(f"getProtein_InterPro({row['Gene_ID']}): WARNING: HTTP ERROR: {ret}")
    else:
        (uniprodID, RBP_Name, taxon_ID, sequence, IPannotations, bressinPossibleNegative), IPR_Annotation_Cache_update = ret
        IPR_Protein_Cache_update[row["Gene_ID"]] = (RBP_Name, taxon_ID, sequence, IPannotations, bressinPossibleNegative)

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

    # Output
    cache_updates = IPR_Protein_Cache_update, IPR_Annotation_Cache_update, MobiDB_Cache_update
    return row, cache_updates


def getBressin19(
    folderPath,
    outputFilePath,
    forceRefresh=False,
    silent=True,
    threads=16,
    bufferSize=1000,  # buffered rows until write to filesystem
):
    global IPR_Protein_Cache, IPR_Annotation_Cache, MobiDB_Cache  # get all the caches

    log(f"\tPreprocessing  Files")
    # Bressin 2019, https://github.com/marsicoLab/TriPepSVM/tree/master/allData
    bressinFiles = [  # Filename, taxonID, positivity
        ("NRBP_9606.fasta", 9606, False),  # Human
        ("RBP_9606.fasta", 9606, True),
        ("NRBP_590.fasta", 590, False),  # Salmonella
        ("RBP_590.fasta", 590, True),
        ("NRBP_561.fasta", 561, False),  # E.coli
        ("RBP_561.fasta", 561, True),
    ]

    Gene_IDs, Gene_Names, taxon_IDs, sequences, positives = [], [], [], [], []
    for fileName, taxonID, positive in bressinFiles:
        log(f"\tPre-Processing {fileName}")
        filePath = folderPath.joinpath(fileName)
        Gene_IDs_file, Gene_Names_file, taxon_IDs_file, sequences_file = preprocessBressin19(filePath, taxonID)  # offline stuff
        Gene_IDs.extend(Gene_IDs_file)
        Gene_Names.extend(Gene_Names_file)
        taxon_IDs.extend(taxon_IDs_file)
        sequences.extend(sequences_file)
        positives.extend([positive] * len(Gene_IDs_file))
        log(f"\t-> {len(Gene_IDs_file)} rows extracted")

    # read save file
    if forceRefresh or outputFilePath.exists() == False:
        if not silent:
            log(f"\tGenerating new empty output file {outputFilePath}")
        df = pd.DataFrame(
            {"Gene_ID": [], "Gene_Name": [], "taxon_ID": [], "canonical": [], "positive": [], "annotations": [], "sequence": []}
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

    log(f"\t{len(existing)} of {len(required)} ({ (len(existing)/len(required))*100 :.06f} %) rows exist in {outputFilePath}")

    # generate missing
    missingNames = list(missingNames)

    if len(missingNames) > 1:
        # setup buffer
        df_dict_buffer = {
            "Gene_ID": [],
            "Gene_Name": [],
            "taxon_ID": [],
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
                for Gene_ID in batch_Gene_IDs:
                    # pack existing information
                    preprocessing_index = Gene_IDs.index(Gene_ID)  # get index of list where Gene_Name is
                    row = {
                        "Gene_ID": Gene_IDs[preprocessing_index],
                        "Gene_Name": Gene_Names[preprocessing_index],
                        "taxon_ID": taxon_IDs[preprocessing_index],
                        "sequence": sequences[preprocessing_index],
                        "annotations": None,
                        "canonical": None,
                        "positive": positives[preprocessing_index],
                    }

                    # caches
                    caches = IPR_Protein_Cache, IPR_Annotation_Cache, MobiDB_Cache

                    # append data
                    threadData.append((row, caches))

                # execute parallel processes
                with Pool(len(threadData)) as p:
                    returnData = p.map(process_getBressin19, threadData)

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
    log("Generating Bressin19")
    bressin19 = getBressin19(
        folderPath=DATA_ORIGINAL.joinpath(f"bressin19/"),  # Bressin 2019, https://github.com/marsicoLab/TriPepSVM/tree/master/allData
        outputFilePath=DATA_RAW.joinpath("bressin19.tsv"),
        forceRefresh=False,
        silent=False,
        threads=webWorkerThreads,
        bufferSize=bufferSize,
    )
