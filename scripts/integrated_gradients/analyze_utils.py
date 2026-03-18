#based on "Integradted_gradients_fun.py"

#import torch
#import torch.nn as nn
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *
from captum.attr import IntegratedGradients
import argparse
from natsort import natsorted
import scipy
import pickle

## Parameters

#Curve smoothing
window_size=31
pol_deg=5


## General

def parseArguments():
    """Parse commandline arguments for IG generation script and return them as "params" dictionary"""

    parser = argparse.ArgumentParser(
                        prog='generate.py',
                        description='Generate IG values given a classifier model')

    #parser.add_argument('-D', '--device', dest='device', action='store',
    #                    help='LM device, either "cuda" for GPU & VRAM or "cpu" for CPU & RAM',
    #                    default="cpu")
    parser.add_argument(
            "-M", "--modelName", dest="model_name", action="store", help="See scripts/models for options", default="Peng"
        )
    parser.add_argument(
            "-lm",
            "--languageModel",
            dest="LM_name",
            action="store",
            help="Language Model. Options: esm1b_t33_650M_UR50S, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D, esm2_t48_15B_UR50D",
            default="esm1b_t33_650M_UR50S",
        )
    parser.add_argument(
            "-ef",
            "--embeddingSubfolder",
            dest="embeddingSubfolder",
            action="store",
            help="Embedding subfolder. Options: bressin19, RIC, InterPro",
            default="bressin19",
        )
    parser.add_argument("-S", "--seed", dest="seed", type=int, default=2023, help="Training seed")
    parser.add_argument("--useToken", dest="useToken", action="store", default=None, help="Token from which to create baseline AA sequence for integrated gradients. Options: mask, pad, unk, cls, eof. If not provided, scalar is used")
    parser.add_argument("--scalar", dest="scalar", type=float, default=None, help="Scaling factor for the IG attributions (None if not used)")
    parser.add_argument("--maskN", dest="maskN", type=int, default=None, help="Calculate attribution by masking N-tuples of AAs (mutually exclusive with integrated gradients i.e. useToken and scalar)")
    parser.add_argument("--zeroN", dest="zeroN", type=int, default=None, help="Calculate attribution by setting N-tuples of AAs to zero (mutually exclusive with maskN integrated gradients i.e. useToken and scalar)")

    args = parser.parse_args() #parse arguments

    # write commandline arguments to params dict
    params = {}
    for key, value in args._get_kwargs():
        params[key] = value

    assert not (params["scalar"] is not None and params["useToken"] != None), "Scalar and useToken are mutually exclusive"
    assert not (params["maskN"] is not None and (params["scalar"] is not None or params["useToken"] != None)), "maskN and scalar/useToken are mutually exclusive"
    assert not (params["zeroN"] is not None and (params["scalar"] is not None or params["useToken"] != None)), "zeroN and scalar/useToken are mutually exclusive"


    return params


def setupFolders(params):

    if( params["useToken"] is not None):
        suffix = params["useToken"]
    elif( params["scalar"] is not None):
        suffix = f"scalar_{params['scalar']}"
    elif( params["maskN"] is not None):
        suffix = f"maskN_{params['maskN']}"
    elif( params["zeroN"] is not None):
        suffix = f"zeroN_{params['zeroN']}"

    # Attributions
    attributionsFolder = ATTRIBUTIONS.joinpath(params["LM_name"]).joinpath(params["embeddingSubfolder"])
    attributionsFolder = attributionsFolder.joinpath(suffix)
    attributionsFolder.mkdir(exist_ok=True, parents=True) # create

    #Figures
    figureFolder = attributionsFolder.joinpath(f"figures")#.joinpath(params["LM_name"])#.joinpath(params["embeddingSubfolder"])
    #figureFolder = figureFolder.joinpath(suffix)
    figureFolder.mkdir(exist_ok=True, parents=True) # create

    dataSetPath = DATA_SETS.joinpath(params["data_set_name"])

    # IG mask embeddings cache
    tokenEmbeddingsFolder = CACHE.joinpath(f"embeddings_{params['useToken']}").joinpath(params["LM_name"]).joinpath(params["embeddingSubfolder"])
    tokenEmbeddingsFolder.mkdir(exist_ok=True, parents=True) # create

    #Embeddings
    embeddingFolder = EMBEDDINGS.joinpath(params["LM_name"]).joinpath(params["embeddingSubfolder"])

    

    return attributionsFolder, figureFolder, dataSetPath, embeddingFolder, tokenEmbeddingsFolder


## Attribution score generation

def plotAttributions(
        attributions_original,
        attributions, #smoothed version
        IG_delta, p_base, p_seq,
        Gene_Name, Gene_ID,
        annotations,
        figureFolder,
        figSize=[6.4*2, 4.8*2],
):

    plt.figure(figsize=figSize)

    plt.plot(attributions_original, alpha=0.3, color="grey")
    plt.plot(attributions, color="black")
    #score_rel = annotations["score"]["rel"]["1or2"] #which score is used for naming the file and reporting in figure?
    AUC = sum(attributions)*1000000 #1e-6
    AUC_rel = AUC/len(attributions)
    title = f"{Gene_ID} ({Gene_Name})" #\nscore either rel="+(f"{score_rel:.04}" if score_rel != None else "None")+f"\nscore abs={abs_score:.2e}"
    
    title += f"\np_base = {p_base:.02f}, AUC={AUC:.02f}*e-6, AUC_rel= {AUC_rel:.04f}*e-6"
    if IG_delta is not None:
        title += f", IG_delta = {IG_delta:.04f}"
    if p_seq is not None:
        title += f", p_seq = {p_seq:.02f}"

    plt.title(title)# \nscore either rel="+(f"{score_rel:.04}" if score_rel != None else "None")+f"\nscore abs={abs_score:.2e}")
    plt.ylabel("Attribution")

    ## motif specific plot ##
    v_min = np.min(attributions_original)
    v_max = np.max(attributions_original)

    ## Iterate through annotations and draw them
    for annotation in annotations:
        #print(f"Annotation: {annotation}")
        fr, to, ty, name, sName = annotation  #(fr, to, ty, name, sName) 
        if(ty == 0):
            color="lavender"
        elif(ty == 1):
            color="springgreen"
        elif(ty == 2):
            color="tomato" 
        # ty 0: no go annotation, the domains is not RNA
        # ty 1: RNA binding go annotation
        # ty 2: intrinsically disordered region (MobiDBLite) annotation

        if(fr >= len(attributions_original)):
            continue

        annotationValues = attributions_original[fr:to+1]
        try:
            annotationValues_mean = sum(annotationValues)/len(annotationValues)
        except ZeroDivisionError:
            print(f"ZeroDivisionError for {Gene_Name} : {fr} - {to}")
        #draw annotation
        plt.hlines(y=annotationValues_mean,xmin=fr,xmax=to,color=color,lw=10,alpha=0.8) #vertical box
        plt.vlines(x=[fr,to],ymin=v_min, ymax=v_max,color=color, linestyles="dashed")

        #txt = f"{name}\n({sName})"
        txt = f"{sName}"
        plt.text(x=(fr+to)/2, y=annotationValues_mean, s=txt, ha='center', va='center', #text
                    fontsize=6, color="black", fontweight="bold" )

    #plt.savefig(figureFolder+f"rel="+(f"{score_rel:.4f}" if score_rel != None else "None")+f"_abs={abs_score:.2e}_{Gene_ID}_{RBP_Name}.svg")
    fileName = f"{Gene_ID}_{Gene_Name}.svg"
    fileName = fileName.replace(" ", "_")
    fileName = fileName.replace("/", "_")
    figurePath = figureFolder.joinpath(fileName)
    plt.savefig(figurePath)
    #plt.show()
    plt.clf()
    plt.close()

def generateAttributionScores(
        scoresFile, scoresFile_statistics,
        dataSet_df, attributionsFolder,
        forceRegenerateExisting = False,
        figureFolder = None):
    """
    Returns modified dataSet_df (also has sideeffects!) with added columns for attributions and masks:
    - attribs_original
    - attribs 
    - RBDmask : boolean np.ndarray
    - IDRmask : boolean np.ndarray
    - otherMask : boolean np.ndarray

    Plot figure for all genes if figureFolder is provided.
    """
    global window_size, pol_deg

    # Get (existing) attribution scores
    if(forceRegenerateExisting or scoresFile.exists() == False):
        dataSet_df = dataSet_df.copy() # copy df without scores
        
        #add new columns
        dataSet_df["attribs_original"] = None # original attribution values (noisy)
        dataSet_df["attribs"] = None # smoothed attribution values (what we actually work on)
        dataSet_df["IG_delta"] = None # delta between original and smoothed attributions
        dataSet_df["p_base"] = None # baseline sequence pos probability
        #dataSet_df["p_seq"] = None # target sequence pos probability
        dataSet_df["RBDmask"] = None # where is RBD
        dataSet_df["IDRmask"] = None # where is IDR
        dataSet_df["otherMask"] = None # where is other (not RBD, not IDR)

        #Statistics dict
        statistics_dict = {
            "all": {
                "other": [],
                "RBD": [],
                "IDR": []
                },
            "perMotif": {} #will be filled with sNames and their count
        }
    else:
        with open(scoresFile, 'rb') as f:
            dataSet_df = pickle.load(f)
        with open(scoresFile_statistics, 'rb') as f:
            statistics_dict = pickle.load(f)

        ## For legacy reasons: rename columns with IG_attribs to attribs
        if "IG_attribs" in dataSet_df.columns:
            dataSet_df.rename(columns={"IG_attribs": "attribs"}, inplace=True)
            dataSet_df.rename(columns={"IG_attribs_mean": "attribs_mean"}, inplace=True)
            #save
            with open(scoresFile, 'wb') as f:
                pickle.dump(dataSet_df, f)


    #Get relevant missingGeneIDs for which to generate scores
    #print(f"Check existing scores")
    #missingGeneIDs = existing #all
    GeneIDs_positives = set(dataSet_df.loc[dataSet_df["positive"] == True]["Gene_ID"])

    #Check existing (where attribs is not None)
    missingGeneIDs= set(dataSet_df.loc[
        np.logical_and(
            dataSet_df["attribs"].isnull(),    # Only those that are not yet processed
            dataSet_df["positive"] == True      # Only positive proteins are relevant
            )
            ]["Gene_ID"])

    print(f"{len(GeneIDs_positives)-len(missingGeneIDs)} of {len(GeneIDs_positives)} ({ ((len(GeneIDs_positives)-len(missingGeneIDs))/len(GeneIDs_positives))*100 :.06f} %) of required scorings for (positive) proteins exist in {scoresFile}")
    #print(f"Missing: {missing}")

    #missingGeneIDs= {"Q8TB72"} #DEBUGGING

    ## Generate attribution scores and add them to the dataframe
    #print(f"Generate missing scores")
    figureFolder.mkdir(exist_ok=True, parents=True)
    for i, geneID in enumerate(tqdm(list(missingGeneIDs))):
        index = dataSet_df.index[dataSet_df["Gene_ID"] == geneID].tolist()[0]
        row = dataSet_df[dataSet_df["Gene_ID"] == geneID].iloc[0]
        Gene_Name = row["Gene_Name"]
        Gene_ID = row["Gene_ID"]
        #taxon_ID = row["taxon_ID"]
        #canonical = row["canonical"]
        #positive = row["positive"]

        try:
            annotations = eval(row["annotations"]) #convert string to list
        except TypeError:
            #print(f"FAILED to evaluate annotations for {Gene_Name} : {row['annotations']}")
            continue

        #sequence = row["sequence"]
        #print(f"sequence: {sequence}")

        ## Get attributions
        IGPath = attributionsFolder.joinpath(Gene_ID)
        with open(IGPath, "rb") as f: #load pickled attributions & context
            d = pickle.load(f)
        attribsKey = "attribs_mean" if "attribs_mean" in d.keys() else "attribs_mean"
        attribs_original = d[attribsKey][1:-1] #remove start and end token
        IG_delta = d["IG_delta"] if "IG_delta" in d.keys() else None
        p_base, p_seq = d["p_base"], d["p_seq"] if "p_seq" in d.keys() else None

        assert type(attribs_original) == np.ndarray
        dataSet_df.at[index, "attribs_original"] = attribs_original

        #print(f"attribs.shape: {attribs.shape}")

        if(len(attribs_original) <= window_size):
            print(f"Attribution values for {Gene_Name} too short ({len(attribs_original)}<={window_size})")
            continue

        ## Get smoothened attributions
        attribs = scipy.signal.savgol_filter(attribs_original, window_size, pol_deg) # window size 51, polynomial order 3
        dataSet_df.at[index, "attribs"] = attribs
                
        ## Get RBD and IDR masks
        RBDmask = np.zeros(len(attribs), dtype=bool)
        IDRmask = np.zeros(len(attribs), dtype=bool)
        for annotation in annotations:
            #print(f"Annotation: {annotation}")
            fr, to, ty, name, sName = annotation  #(fr, to, ty, name, sName) 
            if ty == 1:
                RBDmask[fr:to+1] = True
            elif ty == 2:
                IDRmask[fr:to+1] = True    
            # ty 0: no go annotation, the domains is not RNA
            # ty 1: RNA binding go annotation
            # ty 2: intrinsically disordered region (MobiDBLite) annotation

            if(fr >= len(attribs)):
                continue

            #Get annotation attribution values
            values = attribs[fr:to+1]
            values_mean = sum(values)/len(values)
            if sName not in statistics_dict["perMotif"]:
                statistics_dict["perMotif"][sName] = {"ty":ty, "means":[values_mean]}
            else:
                statistics_dict["perMotif"][sName]["means"].append(values_mean)

        ## Get non-RBD/IDR mask
        otherMask = ~(RBDmask | IDRmask)
        dataSet_df.at[index, "RBDmask"] = RBDmask
        dataSet_df.at[index, "IDRmask"] = IDRmask
        dataSet_df.at[index, "otherMask"] = otherMask
        
        #print(f"otherMask.shape: {otherMask.shape}")

        #print(RBDmask)
        #print(IDRmask)
        #print(otherMask)

        ## Get attributions for RBD, IDR and other regions
        try:
            statistics_dict["all"]["other"].extend(attribs[otherMask])
            statistics_dict["all"]["RBD"].extend(attribs[RBDmask])
            statistics_dict["all"]["IDR"].extend(attribs[IDRmask])
        except IndexError as e:
            print(f"Index error for {Gene_Name} : {e}")
            continue

        ## Plot individual attributions
        #if( any( (subName in Gene_Name) for subName in ["PUM","FUS","FOX","RRM","PTBP"])):
        if figureFolder is not None:
            plotAttributions(
                attributions_original=attribs_original,
                attributions=attribs,
                IG_delta=IG_delta, p_base=p_base, p_seq=p_seq,
                Gene_ID=Gene_ID,
                Gene_Name=Gene_Name,
                annotations=annotations,
                figureFolder=figureFolder)
            
            # Interesting:
            # Q8TB72_PUM2_HUMAN
            
        # Save scores & stats every N proteins (or at teh very end)
        if i % 50 == 0 or i == len(missingGeneIDs)-1:
            with open(scoresFile, 'wb') as f:
                pickle.dump(dataSet_df, f)
            with open(scoresFile_statistics, 'wb') as f:
                pickle.dump(statistics_dict, f)

    return dataSet_df, statistics_dict

def generatePerProteinThresholds(
        perProtein_statistics_file,
        dataSet_df, geneIDs, targetMetric="BACC",
        forcedRegenerateExisting = False,
        thrs = { #thresholds to sample/test
            "uniform": np.linspace(0.0,1.0,100),
            "zscore": np.linspace(-8.0,8.0,800)
        }
    ):

    if(forcedRegenerateExisting or perProtein_statistics_file.exists() == False):
        statistics_dict = {
            #gt_mask
            "RBD": {
                "uniform": { "thrs": [], "scores": [], "types":[] },
                "zscore": { "thrs": [], "scores": [], "types":[] },
                },
            "IDR": {
                "uniform": { "thrs": [], "scores": [], "types":[] },
                "zscore": { "thrs": [], "scores": [], "types":[] },
            },
            "RBD+IDR": {
                "uniform": { "thrs": [], "scores": [], "types":[] },
                "zscore": { "thrs": [], "scores": [], "types":[] },
            },
            "status": { } # <geneID>: "ok" / <error message>
        }
    else:
        with open(perProtein_statistics_file, 'rb') as f:
            statistics_dict = pickle.load(f)

    #Get relevant missingGeneIDs for which to generate scores
    #print(f"Check existing scores")
    existing = set(statistics_dict["status"].keys())
    missingGeneIDs = geneIDs - existing
    print(f"{len(existing)} of {len(geneIDs)} of required perProtein analysis exist in {perProtein_statistics_file}")

    from analyze_utils import thresholdUniform, thresholdZscore, optimizeThreshold

    for i,geneID in enumerate(tqdm(missingGeneIDs)):
        index = dataSet_df.index[dataSet_df["Gene_ID"] == geneID].tolist()[0]
        row = dataSet_df[dataSet_df["Gene_ID"] == geneID].iloc[0]
        #Gene_Name = row["Gene_Name"]
        Gene_ID = row["Gene_ID"]
        #taxon_ID = row["taxon_ID"]
        #canonical = row["canonical"]
        positive = row["positive"]
        attribs_original = row["attribs_original"]
        attribs = row["attribs"]
        RBDmask = row["RBDmask"]
        IDRmask = row["IDRmask"]
        otherMask = row["otherMask"]

        if positive != True:
            continue #we only want to analyze positive examples (or at least such with binding domains)

        #print(f"Analyzing {Gene_ID}")

        #Get type
        if sum(RBDmask) > 0 and sum(IDRmask) == 0:
            typeMask = "RBDonly"
        elif sum(RBDmask) == 0 and sum(IDRmask) > 0:
            typeMask = "IDRonly"
        elif sum(RBDmask) > 0 and sum(IDRmask) > 0:
            typeMask = "both"
        else:
            typeMask = "unknown"


        for gtMaskName, gtMask in zip(["RBD", "IDR", "RBD+IDR"], [RBDmask, IDRmask, (IDRmask | RBDmask)]):
            #print(f"Analyzing {gtMaskName} mask")
            if(sum(gtMask) == 0):
                #print(f"{geneID}\t: Skipping {gtMaskName} mask with 0 elements")
                if(gtMaskName == "RBD+IDR"):
                    errorMsg = "no known RBD or IDR!"
                    print(f"{geneID}\t: {errorMsg}")
                    statistics_dict["status"][geneID] = errorMsg
                continue

            # analyze uniform 
            uniformMasks = thresholdUniform(attribs,thrs=thrs["uniform"])
            opThr, opMetric, opMask = optimizeThreshold(thrs["uniform"], uniformMasks, gtMask, targetMetric=targetMetric)
            #print(f"{geneID}\t: {gtMaskName} uniform: thr={opThr:.04f}, score={opMetric:.04f}")
            statistics_dict[gtMaskName]["uniform"]["thrs"].append(opThr)
            statistics_dict[gtMaskName]["uniform"]["scores"].append(opMetric)
            statistics_dict[gtMaskName]["uniform"]["types"].append(typeMask)

            # analyze z-score
            zscoreMasks = thresholdZscore(attribs,thrs=thrs["zscore"])
            opThr, opMetric, opMask = optimizeThreshold(thrs["zscore"], zscoreMasks, gtMask, targetMetric=targetMetric)
            #print(f"{geneID}\t: {gtMaskName} zscore: thr={opThr:.04f}, score={opMetric:.04f}")
            statistics_dict[gtMaskName]["zscore"]["thrs"].append(opThr)
            statistics_dict[gtMaskName]["zscore"]["scores"].append(opMetric)
            statistics_dict[gtMaskName]["zscore"]["types"].append(typeMask)

            statistics_dict["status"][geneID] = "ok"

        #print(f"RBP: {Gene_ID} : score={score} : thr={thr}. RBDmask: {sum(RBDmask)}, IDRmask: {sum(IDRmask)}, otherMask: {sum(otherMask)}, positive: {positive}")

        # Save stats every N proteins (or at the very end)
        if i % 50 == 0 or i == len(missingGeneIDs)-1:
            with open(perProtein_statistics_file, 'wb') as f:
                pickle.dump(statistics_dict, f)

    return statistics_dict


## plot binding thresholds and scores
def plotHist(dataDict, figureFolder, title="", xlabel="", ylog=False):
    if(len(dataDict) == 0):
        print(f"No data for {title}")
        return
    plt.figure()
    #find min/max
    vmin, vmax = np.inf, -np.inf
    for data in dataDict.values():
        vmin = min(vmin, min(data))
        vmax = max(vmax, max(data))

    opacity = 1.0/len(dataDict.keys())
    for dataName in natsorted(dataDict.keys()):
        data = dataDict[dataName]
        plt.hist(data, bins=50, alpha=opacity, label=dataName, range=(vmin,vmax))
    if(ylog):
        plt.yscale('log')
        plt.ylabel('log(Frequency)')
    else:
        plt.ylabel('Frequency')
    plt.xlabel(xlabel)
    plt.legend()
    plt.title(title)

    figurePath = figureFolder.joinpath(f"{title}.png")
    plt.savefig(figurePath)


## Plot histogram of all attribution values (in 3 main categories)
from matplotlib.ticker import PercentFormatter
def plotAttributionHistogramAll(values_dict, figureFolder, r=None):
    plt.figure()
    if(r is None):
        vmin = min(values_dict["RBD"]+values_dict["IDR"]+values_dict["other"])
        vmax = max(values_dict["RBD"]+values_dict["IDR"]+values_dict["other"])
    else:
        vmin, vmax = r
    alpha = 0.33
    for key in ["other", "RBD", "IDR"]:
        h = plt.hist(values_dict[key], bins=50, alpha=alpha, range=(vmin,vmax), label=key, weights=np.ones(len(values_dict[key])) / len(values_dict[key]))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1))
    #log axis
    plt.ylabel('Frequency (%)')
    plt.yscale('log')
    #plot vertical line at 0
    plt.axvline(x=0, color='black', linestyle='--') 
    plt.xlabel('Attribution value')
    plt.legend()
    #plt.show()
    figurePath = figureFolder.joinpath(f"Histogram_Attributions_All.png")
    plt.savefig(figurePath)

colorCycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] #get default matplotlib color cycle

#plot boxplot of mean attribution values per motif (sorted by median) (use numpy)
# different color for RBD, IDR, other
def plotMotifBoxplots(statistics,figureFolder, minN=10, fs=10):
    #sort by median
    for sName in statistics["perMotif"]:
        statistics["perMotif"][sName]["median"] = np.median(statistics["perMotif"][sName]["means"])
    sortedMotifs = sorted(statistics["perMotif"].items(), key=lambda x: x[1]["median"])
    #remove insignificant occurrences
    relevantMotifs = [(sName, data) for sName, data in sortedMotifs if len(data["means"]) >= minN]
    #pprint(relevantMotifs)

    #plot
    plt.figure( figsize=(len(relevantMotifs)*0.4+0.5, 8))
    plt.hlines(y=0,xmin=0,xmax=len(relevantMotifs),color="black",linestyles="dashed") 
    
    values = []
    colors = []
    names = []
    minimum, maximum = float("inf"), float("-inf")
    for i, (sName, data) in enumerate(relevantMotifs):
        values.append(data["means"])
        colors.append(colorCycle[data["ty"]])
        names.append(sName)
        #plt.boxplot(data["means"], positions=[i], patch_artist=True, boxprops=dict(facecolor=color))
        #update extrema
        if(minimum > min(data["means"])):
            minimum = min(data["means"])
        if(maximum < max(data["means"])):
            maximum = max(data["means"])

    #print(f"values: {values}")
    #print(f"names: {names}")

    plt.boxplot(values, labels=names, widths=0.8, ) # plot boxplot
    plt.xticks(rotation=90)

    # Add N Ns at top and color names
    y_pos = minimum-(maximum-minimum)*0.035
    
    xticks = plt.gca().get_xticklabels() # for coloring

    for i,motif in enumerate(names):
        #N
        N=len(values[i])
        plt.text(x=i+1, y=y_pos, s=f"{N}", rotation="vertical", fontsize=fs, ha='center', fontweight="bold") #text
                            #rotation="vertical", fontsize=8, color="black",  va='center', )

        xticks[i].set_color(colors[i])
        xticks[i].set_fontsize(fs)

    plt.ylabel("Mean attribution")
    plt.title("Mean attribution per motif")
    plt.tight_layout()
    plt.savefig(figureFolder.joinpath("Boxplot_Mean_Attribution_per_Motif.png"))


## Finding the best threshold (and collect statistics)

#Add masks for RBD, IDR, RBD+IDR to statistics_dict as well as attribs (smoothed)
def addMasksAndAttribs(dataSet_df, geneIDs, statistics_dict):

    for geneID in tqdm(geneIDs):
        #index = dataSet_df.index[dataSet_df["Gene_ID"] == geneID].tolist()[0]
        row = dataSet_df[dataSet_df["Gene_ID"] == geneID].iloc[0]
        #Gene_Name = row["Gene_Name"]
        #Gene_ID = row["Gene_ID"]
        #taxon_ID = row["taxon_ID"]
        #canonical = row["canonical"]
        positive = row["positive"]
        #attribs_original = row["attribs_original"]
        attribs = row["attribs"]
        RBDmask = row["RBDmask"]
        IDRmask = row["IDRmask"]
        #otherMask = row["otherMask"]

        if positive != True:
            continue #we only want to analyze positive examples (or at least such with binding domains)

        #print(f"Analyzing {Gene_ID}")

        for gtMaskName, gtMask in zip(["RBD", "IDR", "RBD+IDR"], [RBDmask, IDRmask, (IDRmask | RBDmask)]):
            #print(f"Analyzing {gtMaskName} mask")
            if(sum(gtMask) == 0):
                #print(f"{geneID}\t: Skipping {gtMaskName} mask with 0 elements")
                continue
            statistics_dict[gtMaskName]["mask"].extend(gtMask) #make longer
            statistics_dict[gtMaskName]["attribs"].extend(attribs) #make longer

    return statistics_dict


#TODO: using newton method on individual genes would be better/faster

# OPtimize threshold value based on some metric
def optimizeThreshold(thrs, masks, gt_mask, targetMetric="BACC"):
    """
    thrs: list of thresholds
    masks: list of boolean arrays
    gt_mask: ground truth mask
    targetMetric: metric to optimize for

    return best threshold and best metric
    """
    assert len(thrs) == len(masks), "Length of thresholds and masks must be equal"
    assert targetMetric == "BACC", "Only BACC is manually implemented for now"

    bestThr, bestMetric, bestMask = None, None, None
    for thr,mask in zip(thrs,masks):
        #metrics_dict = getMetricsFromPreds(torch.tensor(mask), torch.tensor(gt_mask)) #returns everything as tensor
        #metricValue = metrics_dict[targetMetric].item()

        #NOTE: we self implement the metric for performance reasons
        gt_labels = gt_mask
        pred_labels = mask
        TP = sum( gt_labels &  pred_labels)
        TN = sum(~gt_labels & ~pred_labels)
        FP = sum(~gt_labels &  pred_labels)
        FN = sum( gt_labels & ~pred_labels)
        TPR = TP / (TP + FN)  # = recall = sensitivity
        TNR = TN / (TN + FP)  # = selectivity = specificiy
        BACC = 0.5 * (
            TPR + TNR
        )  # Balanced accuracy = (Sensitivity + Specificity)/2 = (TPR + FPR)/2
        metricValue = BACC

        if(bestMetric==None or metricValue > bestMetric):
            bestThr,bestMetric, bestMask = thr, metricValue, mask

    return bestThr, bestMetric, bestMask

# Do thresholding for: normalization + linear
def thresholdUniform(attribs, thrs):
    """
    Transforms attribution scores into [0;1] and thresholds them

    attribs: list of attribution scores
    thrs: list of thresholds

    retrun list of boolean arrays
    """
    v = attribs # set values (and truncate if necessary)
    v_min, v_max = np.min(v), np.max(v)
    v_norm = (v-v_min)/(v_max-v_min) #normalized signal into [0;1]
    masks = []
    for thr in thrs:
        masks.append( v_norm > thr )
    return masks

#Do thresholding for: z-score + std
def thresholdZscore(attribs, thrs):
    """
    Computes z-score and thresholds them

    attribs: list of attribution scores
    thrs: list of thresholds (std)

    retrun list of boolean arrays
    """
    v = attribs # set values (and truncate if necessary)
    v_mean, v_std = np.mean(v), np.std(v)
    v_norm = (v-v_mean)/v_std #normalized signal into [0;1]
    masks = []
    for thr in thrs:
        masks.append( v_norm > thr )
    return masks

## Get overall optimal threshold
def addOverallOptimalThreshold(statistics_dict, thrs, targetMetric):
    for gtMaskName in ["RBD", "IDR", "RBD+IDR"]:
        #convert to numyp array
        statistics_dict[gtMaskName]["attribs"] = np.array(statistics_dict[gtMaskName]["attribs"])
        statistics_dict[gtMaskName]["mask"] = np.array(statistics_dict[gtMaskName]["mask"])
        #get arrays
        attribs_total = statistics_dict[gtMaskName]["attribs"]
        gtMasks = statistics_dict[gtMaskName]["mask"]

        # analyze uniform
        uniformMasks = thresholdUniform(attribs_total,thrs=thrs["uniform"])
        opThr, opMetric, opMask = optimizeThreshold(thrs["uniform"], uniformMasks, gtMasks, targetMetric=targetMetric)
        print(f"\t{gtMaskName} uniform: thr={opThr:.04f}, score={opMetric:.04f}")
        statistics_dict[gtMaskName]["uniform"]["opThr"] = opThr
        statistics_dict[gtMaskName]["uniform"][targetMetric] = opMetric

        # analyze z-score
        zscoreMasks = thresholdZscore(attribs_total,thrs=thrs["zscore"])
        opThr, opMetric, opMask = optimizeThreshold(thrs["zscore"], zscoreMasks, gtMasks, targetMetric=targetMetric)
        print(f"\t{gtMaskName} zscore: thr={opThr:.04f}, score={opMetric:.04f}")
        statistics_dict[gtMaskName]["zscore"]["opThr"] = opThr
        statistics_dict[gtMaskName]["zscore"][targetMetric] = opMetric

    return statistics_dict

# Get sucess analysis fro different (non-optimal) thresholds
def addThresholdSuccessProbability(statistics_dict, thrs, thresholdsFile_statistics, forceRegenerateExisting=False):

    for gtMaskName in ["RBD", "IDR", "RBD+IDR"]:
        for method in ["uniform", "zscore"]:
            gtMask = statistics_dict[gtMaskName]["mask"]
            attribs = statistics_dict[gtMaskName]["attribs"]
            gt_labels = np.array(gtMask)  #rename for clarity
            print(f"\t{gtMaskName} {method}")
            for i, thr in enumerate(tqdm(thrs[method])):
                if thr in statistics_dict[gtMaskName][method]["certainty"]["thr"] and not forceRegenerateExisting:
                    continue #was already computed
                
                #Get dist
                #opThr = statistics_dict[gtMaskName][method]["opThr"]
                #dist = thr-opThr # if >0 then binding indication, if <0 no binding indication
                
                #Get prediction
                if(method == "uniform"):
                    pred_labels = thresholdUniform(attribs,thrs=[thr])[0]
                elif(method == "zscore"):
                    pred_labels = thresholdZscore(attribs,thrs=[thr])[0]

                #print(f"thr={thr:.04f}, opThr={opThr:.04f}, dist={dist:.04f}")
                #print(f"pred_labels: {pred_labels}")
                
                #positive predictions
                TP = float(sum(np.logical_and(gt_labels, pred_labels)))
                FN = float(sum(np.logical_and(gt_labels, ~pred_labels)))
                #print(f"TP / (TP + FN) = {TP} / ({TP} + {FN})")
                TPR = TP / (TP + FN)  # = recall = sensitivity

                #Compute positive predictive value
                P = float(sum(pred_labels))
                if(P == 0):
                    PPV = None
                else:
                    PPV = TP / P
                
                #Negative predictions
                TN = float(sum(~gt_labels & ~pred_labels))
                FP = float(sum(~gt_labels &  pred_labels))
                TNR = TN / (TN + FP)  # = selectivity = specificiy
                #print(f"TN / (TN + FP) = {TN} / ({TN} + {FP})")

                #Compute negative predictive value
                N = float(sum(~pred_labels))
                if(N == 0):
                    NPV = None
                else:
                    NPV = TN / N
                
                # Add values to dict
                statistics_dict[gtMaskName][method]["certainty"]["TPR"].append(TPR)
                statistics_dict[gtMaskName][method]["certainty"]["TNR"].append(TNR)
                statistics_dict[gtMaskName][method]["certainty"]["PPV"].append(PPV)
                statistics_dict[gtMaskName][method]["certainty"]["NPV"].append(NPV)
                statistics_dict[gtMaskName][method]["certainty"]["thr"].append(thr)

                # Save stats every N thrs (or at the very end)
                if i % 50 == 0 or i == len(thrs[method])-1: 
                    with open(thresholdsFile_statistics, 'wb') as f:
                        pickle.dump(statistics_dict, f)


    return statistics_dict

## LEGACY CODE Needs refactoring before being usefull! ##


def inferMotifs(
        RBPdomains,
        figureFolder,
        filePath_inf,
        thr_dom = 0.7,
        force_generate_inf=False, #force eval regeneration or just load filePath_eval if it exists
        plot=True,
        figSize=[6.4*2, 4.8*2],
        ):

    if(force_generate_inf or (os.path.isfile(filePath_inf) == False)):
        #add new columns
        if(not f"inf_mask" in RBPdomains.keys()):
            RBPdomains[f"inf_mask"] = [None]*len(RBPdomains.index) 
        if(not f"inf_motifs" in RBPdomains.keys()):
            RBPdomains[f"inf_motifs"] = [None]*len(RBPdomains.index)
        if(not f"inf_motifs_score" in RBPdomains.keys()): #scores of inferred motifs
            RBPdomains[f"inf_motifs_score"] = [None]*len(RBPdomains.index)

        if plot:
            try:
                os.mkdir(figureFolder+f"inf/")
            except FileExistsError:
                pass

        for i in tqdm(RBPdomains.index):
            RBP_Name = RBPdomains.RBP_Name[i]
            Protein_ID = RBPdomains.Protein_ID[i]
            Protein_seq = RBPdomains.Protein_seq[i]
            attribs = RBPdomains.attribs_mean[i]

            domains = RBPdomains.domains[i]

            
            v_original = attribs

            if plot: #need to create figure here
                plt.figure(figsize=figSize)
                plt.plot(v_original, alpha=0.3, color="grey") #plot non-smooth signal

            ##complete not-domain
            baselineMean = RBPdomains[f"baseline_mean"][i]
            if plot:
                plt.hlines(y=baselineMean,xmin=0,xmax=len(v_original),color="blue")

            #apply filter
            v = scipy.signal.savgol_filter(v_original, window_size, pol_deg) # window size 51, polynomial order 3
            v_min = np.min(v[:-5]) # TODO: why is the last value so low? why do we have to compensate this here?
            v_max = np.max(v)

            #get domains
            v_norm = (v-v_min)/(v_max-v_min) #normalized signal into [0;1]
            inf_mask = v_norm > thr_dom #inferred motif mask
            RBPdomains.at[i,f"inf_mask"] = inf_mask

            motif_regions = 0 #getPositiveRegions(inf_mask) #MODIFIED will not work with 0!
            motifs = [] #actual motif sequences
            motif_scores = []
            
            for fr, to in motif_regions:
                motifs.append(Protein_seq[fr:to]) #get actual motif

                #get motif score
                motifValues = v_original[fr:to] #does not need +1 because we built that region ourselfs (and properly)
                motifMean = sum(motifValues)/len(motifValues)
                motifScore = motifMean/baselineMean
                motif_scores.append(motifScore)

            RBPdomains.at[i,f"inf_motifs"] = motifs
            RBPdomains.at[i,f"inf_motifs_score"] = motif_scores
            
        
            #plot stuff
            if plot:
                #get scores for plotting
                score_rel = RBPdomains[f"score_rel_1or2"][i]
                score_abs = RBPdomains[f"score_abs_1or2"][i]

                #plot basic signal
                plt.plot(v, color="black")
                plt.title(f"{Protein_ID} ({RBP_Name}) \nscore rel="+(f"{score_rel:.04}" if score_rel != None else "None")+f"\nscore abs={score_abs:.2e}")
                #\n(IG_delta = {IG_delta:.04f})
                plt.ylabel("Attribution")

                # plot known motifs
                v_min = np.min(v_original)
                v_max = np.max(v_original)
                for domain in domains:
                    fr, to, ty, name, sName = domain
                    fr = int(fr)
                    to = int(to)

                    if(ty == 0):
                        color="lavender"
                    elif(ty == 1):
                        color="springgreen"
                    elif(ty == 2):
                        color="tomato"

                    motifValues = v_original[fr:to+1]
                    motifMean = sum(motifValues)/len(motifValues)
                    #draw motifs
                    plt.hlines(y=motifMean,xmin=fr,xmax=to,color=color,lw=10,alpha=0.8) #vertical box
                    plt.vlines(x=[fr,to],ymin=v_min, ymax=v_max,color=color, linestyles="dashed")

                    plt.text(x=(fr+to)/2, y=motifMean, s=f"{name}\n({sName})", ha='center', va='center', #text
                                                fontsize=8, color="black", fontweight="bold" )

                #plot new motifs
                for i, (fr, to) in enumerate(motif_regions):
                    motifValues = v_original[fr:to]
                    motifMean = sum(motifValues)/len(motifValues)
                    motif = motifs[i]

                    #draw motifs
                    plt.hlines(y=motifMean,xmin=fr,xmax=to,color="aqua",lw=10,alpha=0.8) #vertical box
                    plt.vlines(x=[fr,to],ymin=v_min, ymax=v_max,color="aqua", linestyles="dashed")

                    plt.text(x=(fr+to)/2, y=motifMean, s=motif, ha='center', va='center', #text
                                    fontsize=1200/len(Protein_seq), color="black", fontweight="bold")


                plt.savefig(
                        figureFolder+f"inf/"+f"rel="+(f"{score_rel:.4f}" if score_rel != None else "None")+f"_abs={score_abs:.2e}_{Protein_ID}_{RBP_Name}.svg")
                #plt.show()
                plt.clf()
                plt.close()
                    
        #save
        print("Saving Inference results")
        RBPdomains.to_pickle(filePath_inf)
    else:
        print("Loading Inference results")
        RBPdomains = pd.read_pickle(filePath_inf)

    return RBPdomains

