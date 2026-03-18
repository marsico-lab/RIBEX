
# Parameters

from analyze_utils import *

# DataFrame columns (minimum) = {
#            "Gene_ID": #uniprodID
#            "Gene_Name":
#            "taxon_ID":
#            "sequence": ,
#            "positive": ,
#            "annotations":

# annotation type is either:
# 0 = other-domains
# 1 = RNA binding domain (Go annotation 'GO:0003723')
# 2 = IDR
# sName is more specific
# e.g. for type=1: PUM-HD
# or for type=2: Mobidblt-Consensus Disorder

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


#General analysis
for datasetName in dataSetsRaw.keys():
    log(f"{datasetName}:")
    dataset=dataSetsRaw[datasetName]

    log(f"\tColumns: {list(dataset.keys())}") # column names
    analyze_general(dataset) #other stuff

    log(f"\tBalance: ")
    analyze_Balance(dataset)

    log(f"\tAnnotations - All: ")
    analyze_Annotations(dataset)

    log(f"\tAnnotations - Human: ")
    analyze_Annotations(dataset, taxon_ID=9606)

#Dataset specific analysis

#Bressin
log("bressin19 - Uncertain positives")
analyze_UncertainPositivity(dataSetsRaw["bressin19"])

#RIC
log("RIC - Positive count")
analyze_RICpositivesCount(dataSetsRaw["RIC"])

#InterPro
log("uncertain positives - InterPro")
analyze_UncertainPositivity(dataSetsRaw["InterPro"])