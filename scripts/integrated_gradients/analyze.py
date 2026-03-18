# Initialize global environment and import useful utility functions
import sys
from pathlib import Path

sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *
initialize(__file__)

from scripts.integrated_gradients.analyze_utils import parseArguments, setupFolders, plotAttributions
#from scripts.training.utils import getDataset, getModelFromCkpt
#from scripts.embeddings.utils import getModel

from tqdm import tqdm
import pickle
#from captum.attr import IntegratedGradients
import scipy
import matplotlib.pyplot as plt

import numpy as np
from pprint import pprint, pformat

params = parseArguments()

#Dataset parameters
params["data_set_name"] = f"{params['embeddingSubfolder']}.pkl" #The dataset filename that should be used. See $DATA/data_sets for options
dataSetPath = DATA_SETS.joinpath(params["data_set_name"])

# Print/Log paramaters
log("Parameters:")
for key in params.keys():
    log(f"\t{key}: {params[key]}")

# Seed everything (numpy, torch, lightning, even workers)
#TODO: requires?

RBP_IG_Folder, figureFolder,  dataSetPath, embeddingFolder, tokenEmbeddingsFolder = setupFolders(params)
attributionsFolder = RBP_IG_Folder.joinpath("attr_raw") #where the raw attributions are stored
#attributionsFolder = RBP_IG_Folder


## Get raw attributions

#Read dataset table
with open(dataSetPath, 'rb') as f:
    dataSet_df = pickle.load(f)
#Data set columns/keys: "Gene_ID", "Gene_Name", "taxon_ID", "canonical", "positive", "annotations", "sequence", "cluster"
# With annotation tuples: (fr, to, ty, name, sName) where ty: 0=other,1=RBD,2=IDR

#sanity check if all the required embeddings actually exist
required = set(dataSet_df["Gene_ID"])
existing = set([ p.name for p in attributionsFolder.iterdir()]) # Note: this might be more than "required" bit not all are relevant
missing = required-existing  # todo = required - everyFile
done = set.intersection(required,existing)
assert len(existing) > 0, f"No files to operate on exist."

log(f"\t\t{len(done)} of {len(required)} ({ (len(done)/len(required))*100 :.06f} %) required files exist in {attributionsFolder}")


## Get attribution scores
forceRegenerateExisting = False #set to true if you want to regenerate the table
scoresFile = RBP_IG_Folder.joinpath("dataset_with_scores.pkl")
scoresFile_statistics = RBP_IG_Folder.joinpath("dataset_with_scores_statistics.pkl")

from analyze_utils import generateAttributionScores
dataSet_df, statistics_dict = generateAttributionScores(
    scoresFile, scoresFile_statistics,
    dataSet_df, attributionsFolder,
    forceRegenerateExisting = False,
    figureFolder=figureFolder.joinpath("attributionGraphs") # for the individual plots
)

## Analyze attribution scores
geneIDs = set( dataSet_df.loc[dataSet_df["attribs"].notnull()]["Gene_ID"] ) #Get set of geneIDs with attribution scores (this is what we want to analyze)

## Plot histogram of all attribution values (in 3 main categories)
from analyze_utils import plotAttributionHistogramAll
print(f"Plot attribution histogram")
#r = (-0.001,0.001) if ( (params["zeroN"] is not None) or (params["maskN"] is not None)) else
r = (-0.0000001,0.0000001)
plotAttributionHistogramAll(values_dict=statistics_dict["all"], figureFolder=figureFolder, r=(-0.001,0.001))

#plot boxplot of mean attribution values per motif (sorted by median) (use numpy)
# different color for RBD, IDR, other
from analyze_utils import plotMotifBoxplots

#plot motif analysis (if enough data is present)
minN = 50
if(len(geneIDs)>2*minN):
    print(f"Plot Motif Boxplots")
    plotMotifBoxplots(statistics_dict, figureFolder, minN=minN)
else:
    print(f"NOT Plotting Motiv Boxplots (N={len(geneIDs)} < 2*{minN})")
    pass



## Find right thresholds for binding classification
from scripts.training.analyze_utils import getMetricsFromPreds
thrs = { #thresholds to consider / test
        "uniform": np.linspace(0.0,1.0,100),
        "zscore": np.linspace(-8.0,8.0,800)
    }

## Per-protein optimal threshold generation & analysis
targetMetric = "BACC"
perProtein_statistics_file = RBP_IG_Folder.joinpath(f"perProtein_threshold_{targetMetric}_statistics.pkl")

print(f"Analyse per-protein-optimal thresholds for binding classification")
from analyze_utils import generatePerProteinThresholds
statistics_dict = generatePerProteinThresholds(
    perProtein_statistics_file,
    dataSet_df, geneIDs, targetMetric,
    forcedRegenerateExisting = False,
    thrs = thrs
)

#pprint(statistics_dict)

## plot binding thresholds and scores
from natsort import natsorted
print(f"Plot per-protein binding thresholds and scores")
from analyze_utils import plotHist
for gtMaskName in ["RBD", "IDR", "RBD+IDR"]:
    for method in ["uniform", "zscore"]:
        for key in ["thrs", "scores"]:
            data = statistics_dict[gtMaskName][method][key]
            typeMask = statistics_dict[gtMaskName][method]["types"]
            dataDict = {}
            for value, typeMask in zip(data, typeMask):
                if typeMask not in dataDict:
                    dataDict[typeMask] = []
                dataDict[typeMask].append(value)
            
            xlabel = f"thr_{method}" if key == "thrs" else targetMetric
            ylog = True
            plotHist(dataDict, figureFolder, title=f"{gtMaskName}_{method}_{key}_{targetMetric}", xlabel=xlabel)


## Find overall best threshold
thresholdsFile_statistics = RBP_IG_Folder.joinpath("optimalThresholds_and_statistics.pkl")
forceRegenerateExisting = False #set to true if you want to regenerate the table

#Get dict (load from file or initialize)
if forceRegenerateExisting or thresholdsFile_statistics.exists() == False:
    statistics_dict = {
        "RBD": {
            "mask": [], "attribs": [],
            "uniform": { "opThr": None, "certainty": {"thr":[], "TPR":[], "TNR":[] , "PPV":[], "NPV":[] } },
            "zscore": { "opThr": None, "certainty": {"thr":[], "TPR":[], "TNR":[] , "PPV":[], "NPV":[] } },
        },
        "IDR": {
            "mask": [], "attribs": [],
            "uniform": { "opThr": None, "certainty": {"thr":[], "TPR":[], "TNR":[] , "PPV":[], "NPV":[] } },
            "zscore": { "opThr": None, "certainty": {"thr":[], "TPR":[], "TNR":[] , "PPV":[], "NPV":[] } },
        },
        "RBD+IDR": {
            "mask": [], "attribs": [],
            "uniform": { "opThr": None, "certainty": {"thr":[], "TPR":[], "TNR":[] , "PPV":[], "NPV":[] } },
            "zscore": { "opThr": None, "certainty": {"thr":[], "TPR":[], "TNR":[] , "PPV":[], "NPV":[] } },
        },
        "addedMasksAndAttribs": False,
        "addedOverallThresholds": False,
        "addedSuccessProbability": False
    }
    # Explainations:
    # "mask": [] #overall (concatinated) binding mask (boolean array)
    # "attribs": [] #overall (concatinated) IG attributions
    # "uniform": {
    #   "opThr": <float>, # optimal threshold for concatenated sequence
    #   "score": {<metricName>: value, ...}, # best value (@ opThr) each metric (key) for concatenated sequence
    #   "certainty": {"thr":[], "TPR":[], "TNR":[] , "PPV":[], "NPV":[] } #maps thr to True Positive/Negative Rate: (dist, TPR or TNR)
    #                           (depending wether you are under the threshold or over (dist<0 or dist>0))
    # }
else:
    with open(thresholdsFile_statistics, 'rb') as f:
        statistics_dict = pickle.load(f)

#Add masks and attributions
if(statistics_dict["addedMasksAndAttribs"] == False): #step is missing
    print(f"Adding masks and attributions to {thresholdsFile_statistics}")
    from analyze_utils import addMasksAndAttribs
    statistics_dict = addMasksAndAttribs(dataSet_df, geneIDs, statistics_dict)
    statistics_dict["addedMasksAndAttribs"] = True
    with open(thresholdsFile_statistics, 'wb') as f:
        pickle.dump(statistics_dict, f)
else:
    print(f"Already done: Adding masks and attributions to {thresholdsFile_statistics}")

# Get overall threshold
if(statistics_dict["addedOverallThresholds"] == False): #step is missing
    print(f"Adding overall optimal thresholds to {thresholdsFile_statistics}")
    from analyze_utils import addOverallOptimalThreshold
    statistics_dict = addOverallOptimalThreshold(
        statistics_dict, thrs, targetMetric
    )
    statistics_dict["addedOverallThresholds"] = True
    with open(thresholdsFile_statistics, 'wb') as f:
        pickle.dump(statistics_dict, f)
else:
    print(f"Already done: Adding overall optimal thresholds to {thresholdsFile_statistics}")


## Get success probability given different thresholds
if statistics_dict["addedSuccessProbability"] == False:
    from analyze_utils import addThresholdSuccessProbability
    print(f"Adding success probability given different thresholds to {thresholdsFile_statistics}")
    statistics_dict = addThresholdSuccessProbability(statistics_dict, thrs, thresholdsFile_statistics)
    statistics_dict["addedSuccessProbability"] = True
    with open(thresholdsFile_statistics, 'wb') as f:
        pickle.dump(statistics_dict, f)
else:
    print(f"Already done: Adding success probability given different thresholds to {thresholdsFile_statistics}")

## Plot success probability given different thresholds
print(f"Plot success probability given different thresholds")
colorCycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
for method in ["uniform", "zscore"]:
    opThr = statistics_dict[gtMaskName][method]["opThr"] # optimal threshold (point on X-axis)
    xs = thrs[method] #thresholds = X-axis

    for plotType, certantyIndicators in [("rate",["TPR", "TNR"]), ("predictive_value",["PPV", "NPV"])]:

        plt.figure()
        for i, gtMaskName in enumerate(["RBD", "IDR", "RBD+IDR"]):
            c = colorCycle[i]
            for certantyIndicator in certantyIndicators:
                indicator_values = statistics_dict[gtMaskName][method]["certainty"][certantyIndicator]
                ls = ":" if certantyIndicator in ["TNR", "NPV"] else "--" # dots for negative, dashes for positive
                plt.plot(xs, indicator_values, label=f"{gtMaskName}, {certantyIndicator}", color=c, linestyle=ls)

        plt.vlines(x=opThr, ymin=0, ymax=1, color="black", linestyles="dashed", label="opThr") #plot vertical line for threshold
            
        plt.legend()
        plt.title(f"{method}_{certantyIndicators}")
        plt.xlabel(f"thr_{method}")
        plt.ylabel(plotType)
        figurePath = figureFolder.joinpath(f"{method}_certainty_{plotType}.png")
        plt.savefig(figurePath)
        plt.clf()

##Print summary to file
summaryFile = RBP_IG_Folder.joinpath("summary.txt")
with open(summaryFile, 'w') as f:
    f.write(f"Summary of {targetMetric} optimal thresholds and scores\n")
    f.write("gtMaskName\tmethod\ttargetMetric\topThr\topMetric\n")
    for method in ["uniform", "zscore"]:
        for gtMaskName in ["RBD", "IDR", "RBD+IDR"]:
            opThr = statistics_dict[gtMaskName][method]["opThr"]
            opMetric = statistics_dict[gtMaskName][method][targetMetric]
            f.write(f"{gtMaskName}\t{method}\t{targetMetric}\t{opThr:.4f}\t{opMetric:.4f}\n")

print("done.")


#Hier:
# - make motif inference i.e. where bindign where not
# - get binding probability (first fix threshold and then check how
#    the distance to threshold correlates with correctness (i.e. if you are +30% over threshold you are 90% certain its binding))
#   - make nice visualization out of that (heatmap of AA sequence with color indicating certainty of binding)

#NOTE: until here everything is the same as in generate.py

##TODO: get pos/neg value (based on mean or some ?)
##TODO: do the other Kmer stuff (if possible without proper section finding)


exit()
## The following code is inspired by the "Integrated_Gradients.ipynb" of the original ml4rg repo

# 5. den restlichen code der in Integradted_gradients_fun.py war (woraus die integrated_gradients_utisl.py geacht ist)
#   in die entsprechenden "attributions" scripte integrieren. Evnetuell auch den "integrated_gradients" fodler auflösen

#Plot sum of attribution values versus length of sequences
plotAttributionVSLength(RBPdomains_IG,figureFolder, figSize=defaultFigSize) 
# -> what is that correlations?! why is that shape emerging?!