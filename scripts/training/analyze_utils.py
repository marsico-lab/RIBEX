import pandas as pd
import numpy as np

# - loose patience with phd students -> work/career thing, hyprocrasy, motivation/responsibility academia
#    -> remind of their responisbility
# - exmatricualtion law
# - sophie testing waters -> konsti message thankful -> which ones okay in germany


# plotting (does not work on JUWELS because )
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn2_circles, venn3, venn3_circles
import pytorch_lightning as pl
#from scripts.training.dataset import DataSet, DataSet_Residual
#from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np # as sklearn does not need to use pytorch
from scripts.initialize import *
from pprint import pprint, pformat
import random

FIGURES_TRAINING = FIGURES.joinpath("training")
if FIGURES_TRAINING.exists() == False:
    FIGURES_TRAINING.mkdir()

from torchmetrics import PrecisionRecallCurve, ROC
from torcheval.metrics import AUC
# only 50 evenly spaced thresholds → 51 precision/recall points
pr_curve = PrecisionRecallCurve(task="binary", thresholds=50)
roc_curve = ROC(task="binary", thresholds=50)
auc = AUC()


# compute certrainty of MCC and BACC
def calc_errs(
    tp, fp, tn, fn, iterations=1000
):  # This is a modified version of a function I have from Henrik!
    """
    Calculates the Error estimates for a variety of performance measurements from the given confusion matrix counts
    :param tp: True Positives
    :param fp: False Positives
    :param tn: True Negatives
    :param fn: False Negatives
    :return: MCC-error, BACC-error
    """
    data = ["tp"] * tp + ["fp"] * fp + ["tn"] * tn + ["fn"] * fn

    mccs = []
    bal_accs = []

    for i in range(iterations):
        # initialize local counts of tp, fp, tn, fn
        loc_tp = 0
        loc_fp = 0
        loc_tn = 0
        loc_fn = 0

        for j in range(tp + fp + tn + fn - 1):
            # randomly pick performances with replacement
            pick = random.randint(0, tp + fp + tn + fn - 1)
            temp = data[pick]
            if temp == "tp":
                loc_tp += 1
            if temp == "fp":
                loc_fp += 1
            if temp == "tn":
                loc_tn += 1
            if temp == "fn":
                loc_fn += 1

        # calculate the measurements for this iteration and append them to their respective list
        try:
            mcc = ((loc_tp * loc_tn) - (loc_fp * loc_fn)) / np.sqrt(
                    (loc_tp + loc_fp)
                    * (loc_tp + loc_fn)
                    * (loc_tn + loc_fp)
                    * (loc_tn + loc_fn)
                )
        except ZeroDivisionError as e:
            mcc = 0
        except RuntimeWarning as w:
            mcc = 0
        
        mccs.append(mcc)

        try:
            bal_acc = 0.5 * (loc_tp / (loc_tp + loc_fn) + loc_tn / (loc_tn + loc_fp))
        except ZeroDivisionError as e:
            bal_acc = 0
        bal_accs.append(bal_acc)

    # calculate standard deviation of the performance measurements
    std_mcc = np.std(mccs)
    std_bal_acc = np.std(bal_accs)

    return std_mcc, std_bal_acc


# plot precision recall curve
def plotPRC(PRC, fileName="PRC.png", dpi=200):
    global FIGURES_TRAINING

    precision, recall, PRC_thresholds = PRC
    plt.figure()
    plt.title("PRC")
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    if fileName != None:
        plt.savefig(FIGURES_TRAINING.joinpath(fileName), dpi=dpi)
    # plt.show()


# Plot receiver operating characteristic
def plotROC(ROC, fileName="ROC.png", dpi=200):
    global FIGURES_TRAINING

    FPR, TPR, ROC_thresholds = ROC
    plt.figure()
    plt.title("ROC")
    plt.plot(FPR, TPR)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    if fileName != None:
        plt.savefig(FIGURES_TRAINING.joinpath(fileName), dpi=dpi)
    # plt.show()


def scatterCategories(
    dataFrame, targetKey="p", violin=False, fileName="CategoriesScatter.png", dpi=200
):
    global FIGURES_TRAINING

    # Get points
    values = []
    labels = []
    for category in dataFrame.keys():
        df = dataFrame[category]
        targetValues = df[targetKey]
        N = len(targetValues)

        if violin:
            labels.append(category)
            if N <= 2:  # for violin plots lists cant be empty!
                values.append([float("nan"), float("nan")])
            else:
                values.append(targetValues)
        else:
            labels.extend([category] * N)
            values.extend(targetValues)

    plt.figure()
    plt.title(f"Categories vs. {targetKey}")
    if violin:
        plt.violinplot(dataset=values, showmeans=False, showmedians=True)
        plt.xticks(np.arange(1, len(labels) + 1), labels=labels)
    else:
        plt.scatter(x=labels, y=values)
    plt.xlabel("Categories")
    plt.ylabel(f"{targetKey}")
    if fileName != None:
        plt.savefig(FIGURES_TRAINING.joinpath(fileName), dpi=dpi)
    # plt.show()


def getMetricsFromPreds(pred_labels, gt_labels, prefix=""):

    # Ensure boolean tensor type (int tensor will result in wrong results)
    pred_labels, gt_labels = pred_labels.bool(), gt_labels.bool()

    # Get Metrics
    TP = sum(torch.logical_and(gt_labels, pred_labels))
    TN = sum(torch.logical_and(~gt_labels, ~pred_labels))
    FP = sum(torch.logical_and(~gt_labels, pred_labels))
    FN = sum(torch.logical_and(gt_labels, ~pred_labels))

    TPR = TP / (TP + FN)  # = recall = sensitivity
    TNR = TN / (TN + FP)  # = selectivity = specificiy
    BACC = 0.5 * (
        TPR + TNR
    )  # Balanced accuracy = (Sensitivity + Specificity)/2 = (TPR + FPR)/2
    MCC = ((TP * TN) - (FP * FN)) / torch.sqrt(
        (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    )  # mean square contingency coefficient

    # other metrics
    P = sum(pred_labels)
    N = sum(~pred_labels)
    imbalance = N / P  # minority to majority, 1: X
    # ACC = (TP + TN) / (P + N) #Bad for imbalanced data
    PPV = TP / (TP + FP)  # = precision
    F1 = (2 * TP) / (2 * TP + FP + FN)
    GMEAN = torch.sqrt(TPR * TNR)  # geometric mean

    std_mcc, std_bal_acc = calc_errs(TP, FP, TN, FN)

    TP, TN, FP, FN = (
        TP.to(torch.float32),
        TN.to(torch.float32),
        FP.to(torch.float32),
        FN.to(torch.float32),
    )

    return {
        f"{prefix}TP":TP,f"{prefix}TN":TN,f"{prefix}FP":FP,f"{prefix}FN":FN, #mostly irrelevant
        # f"{prefix}P":P,f"{prefix}N":N,
        # f"{prefix}ACC":ACC, #bad for imbalanced
        f"{prefix}imbalance_factor": imbalance,
        f"{prefix}precision": PPV,
        f"{prefix}TPR": TPR,
        f"{prefix}TNR": TNR,
        f"{prefix}F1": F1,
        f"{prefix}GMEAN": GMEAN,
        f"{prefix}MCC": MCC,
        f"{prefix}MCC_std": std_mcc,
        f"{prefix}BACC": BACC,
        f"{prefix}BACC_std": std_bal_acc
        }


def getMetrics(preds, gt_labels: torch.tensor = None, prefix="", is_harmonic_mean=True):

    if not isinstance(preds, torch.Tensor) and gt_labels is None:
        probabilities, labels = preds
        probabilities = torch.softmax(torch.Tensor(probabilities), dim=1)[:,1]
        gt_labels = torch.Tensor(labels).to(torch.int64) # truncate not round
    else:
        probabilities = torch.Tensor(preds)
        gt_labels = torch.Tensor(gt_labels).to(torch.int64) # truncate not round

    # Curves (requires float probabilities)
    precision, recall, PRC_thresholds = pr_curve(probabilities, gt_labels)
    if is_harmonic_mean:
        # Calculate the harmonic mean of precision and recall
        F1_scores = 2 * precision * recall / (precision + recall)
        # if nan values are present, set them to 0
        F1_scores[torch.isnan(F1_scores)] = 0
        # choose as threshold the one that maximizes the F1 score
        threshold_index = torch.argmax(F1_scores)
    else:
        # Calculate the Euclidean distance to (1,1) in PR space
        distances = np.sqrt((1 - precision) ** 2 + (1 - recall) ** 2)
        # Choose the threshold that minimizes the distance
        threshold_index = torch.argmin(torch.tensor(distances))

    if torch.numel(PRC_thresholds) > 0:
        threshold = PRC_thresholds[threshold_index].item()
        thr_eps = 0.00001
        if threshold == 0.0:
            threshold += thr_eps
        elif threshold == 1.0:
            threshold -= thr_eps
    else:
        print("No threshold found, using 0.5")
        threshold = 0.5

    #print(f"Threshold: {threshold}")
    # Get predicted labels and set type
    pred_labels = probabilities >= threshold
    gt_labels = torch.Tensor(gt_labels).to(torch.int64)
    pred_labels = torch.Tensor(pred_labels).to(torch.int64)

    # Compute all metrics computable on boolean values
    metricsDict = getMetricsFromPreds(pred_labels, gt_labels, prefix=prefix)

    # Curves (requires float probabilities)
    precision, recall, PRC_thresholds = pr_curve(probabilities, gt_labels)
    fpr, tpr, ROC_thresholds = roc_curve(probabilities, gt_labels) 
    prc = (precision.tolist(), recall.tolist(), PRC_thresholds.tolist())
    roc = (fpr.tolist(), tpr.tolist(), ROC_thresholds.tolist())  # TODO simplyfy with BACC above!

    #print(f"N:\t {len(gt_labels)}, {len(probabilities)}")
    #print(f"metricsDict:\n{pformat(metricsDict)}")

    # Area under curves
    auc.update(recall, precision),  # Area under PRecision Recall curve
    AUPRC = auc.compute()[0].tolist()
    auc.reset()
    #print(f"AUPRC:{AUPRC}")


    auc.update(fpr, tpr)  # Area under Reciever-operator curve)
    AUROC = auc.compute()[0].tolist()
    auc.reset()
    #print(f"AUROC:{AUROC}")

    metricsDict.update( {        
        f"{prefix}prc": prc,
        f"{prefix}AUPRC": AUPRC,
        f"{prefix}roc": roc,
        f"{prefix}AUROC": AUROC,  # potentially problematic for imbalanced data
    } )

    return metricsDict

def logMetrics(metricsDict, indentation=4):
    # Log stats
    N = metricsDict["TP"] + metricsDict["FP"] + metricsDict["TN"] + metricsDict["FN"]
    log(f"N:\t {N}", indentation=indentation)
    for key in metricsDict:
        if key in [
            "TP",
            "TN",
            "FP",
            "FN",
            "TPR",
            "TNR",
            "AUPRC",
            "AUROC",
        ]:  # dont log the curves data
            log(f"{key}:\t {float(metricsDict[key]):.4f}", indentation=indentation)
        if key in ["BACC", "MCC"]:
            log(
                f"{key}:\t {float(metricsDict[key]):.4f} (±{float(metricsDict[key+'_std']):.4f})",
                indentation=indentation,
            )


def evaluateModel(model, datasetDict, params):

    if params["model_name"] in ["Linear", "RandomForest", "XGBoost", "Random_SK"]: #sklearn models
        #Get metrics for training set
        preds = model.predict_proba(datasetDict["X_train"])
        preds_pos_train = preds[:, [1]]
        Y_train = np.expand_dims(datasetDict["Y_train"],1)
        prefix = "train_"
        metrics_train = getMetrics(preds_pos_train, Y_train, prefix=prefix)

        #Get metrics for valdiation set
        preds = model.predict_proba(datasetDict["X_val"]) # get predicted class probabilities
        preds_pos_val = preds[:, [1]] # get predicted probabilities for class 1 (pos), shape (N, 1)
        Y_val = np.expand_dims(datasetDict["Y_val"],1) # shape: (N) -> (N,1)
        prefix = "val_"
        metrics_val = getMetrics(preds_pos_val, Y_val, prefix=prefix)

        return {"train_metrics":metrics_train, "train_probs": preds_pos_train, "train_labels": Y_train,
                "val_metrics":metrics_val, "val_probs": preds_pos_val, "val_labels": Y_val}
    
    elif params["model_name"] in ["Peng", "Peng6", "Linear_pytorch"]:
        raise NotImplementedError("Manual evaluation for pytorch models not implemented yet.")
        # This should not be nessecary anyways as for pytroch a lightningModule can be used.
        # So instead of calling this method fix your model definition!!
    elif params["model_name"] in ["Lora"]:
        #Evaluation results already exist in the model state dict, just need renaming
        metrics_train = {}
        metrics_val = {}
        for entry in model.state.log_history:
            for key in entry.keys():
                if "eval_" in key and "prc" not in key and "roc" not in key:
                    metrics_val[key.replace("eval_", "val_")] = entry[key]
                if "train_" in key and "prc" not in key and "roc" not in key:
                    metrics_train[key] = entry[key]
        return {"train_metrics":metrics_train, "val_metrics":metrics_val}
        #TODO: add probabilities and labels to the return dict

    else:
        raise NotImplementedError(f"Model \"{params['model_name']}\" not implemented for manual evaluation")


def manualLogging(metrics_train, metrics_val, params, logger_WB, logger_TB):
    #Log hyperparamaters
    logger_WB.log_hyperparams(params)
    logger_TB.log_hyperparams(params)

    #Log metrics for training set
    prefix = "train_"
    for key in metrics_train.keys():
        d = {key: metrics_train[key]}
        if key in [f"{prefix}prc", f"{prefix}roc"]:
            pass
        else: # is scalar
            logger_WB.log_metrics(d)
            logger_TB.log_metrics(d)

    #Log metrics for validation set
    prefix = "val_"
    for key in metrics_val.keys():
        d = {key: metrics_val[key]}
        #log curves/images
        if key in [f"{prefix}prc", f"{prefix}roc"]:
            #logger_WB.log_graph(key, metrics[key])
            #logger_TB.log_graph(key, metrics[key])
            pass
            #TODO: how to log curves?!
            #prc = (precision, recall, PRC_thresholds)
            #roc = (fpr, tpr, ROC_thresholds) 
        else: # is scalar
            logger_WB.log_metrics(d)
            logger_TB.log_metrics(d)

    
def plotCurves(metricsDict, fileNamePrefix):
    plotPRC(metricsDict["prc"], fileName=f"{fileNamePrefix}PRC.png")
    plotROC(metricsDict["roc"], fileName=f"{fileNamePrefix}ROC.png")

def analyzeProteinAnnotationTypes(
    probabilities, labels, dataSet_df, idx, indentation=4
):
    # certainty analysis (how close is probability to 0 or 1) d in [0,0.5]
    labelDistances = torch.zeros_like(probabilities)
    for i, (p, label) in enumerate(zip(probabilities, labels)):
        if p > 0.5:
            d = 1 - p
        else:
            d = p
        labelDistances[i] = d

    # get where the sequence was truncated
    data = {
        "both": {"p": [], "label": [], "dist": []},
        "IDR-only": {"p": [], "label": [], "dist": []},
        "RBD-only": {"p": [], "label": [], "dist": []},
        "others-only": {"p": [], "label": [], "dist": []},
    }
    for i, id in enumerate(tqdm(idx)):
        sub_df = dataSet_df.iloc[id.item()]
        d = dict(sub_df)  # sample as dict (columns=keys, cells=values)
        annotations = d["annotations"]
        p = probabilities[i]
        label = labels[i]
        dist = labelDistances[i]

        # get protein category
        hasIDR = False
        hasRBD = False

        # if we do not have annotation information about this protein
        if type(annotations) != str:  # -> annotations is none (type=float).
            # log(f"[{id}]: annotations are nan",indentation=indentation)
            continue

        for t in eval(annotations):
            (fr, to, ty, name, sName) = t
            if ty == 1:
                hasRBD = True
            elif ty == 2:
                hasIDR = True

        if hasIDR and hasRBD:
            df = data["both"]
        elif hasIDR == True and hasRBD == False:
            df = data["IDR-only"]
        elif hasIDR == False and hasRBD == True:
            df = data["RBD-only"]
        else:
            df = data["others-only"]

        # aggregate data required for statisticalanalysis
        df["p"].append(float(p))
        df["label"].append(bool(label))
        df["dist"].append(float(dist))

    metricsTable = {}
    for category in data.keys():
        df = data[category]
        N = len(df["p"])
        log(f"{category} (N={N})", indentation=indentation)
        if N < 10:
            log(f"Skipping due to sample size", indentation=indentation + 1)
            continue

        metricsDict = getMetrics(
            probabilities=torch.tensor(df["p"]), gt_labels=torch.tensor(df["label"])
        )
        # logMetrics(metricsDict, indentation=5) #will be done for all in a seperate table
        # plotCurves(metricsDict, fileNamePrefix = f"{modelName}__{instanceName}__{ckptName}__")
        metricsDict.pop("prc")
        metricsDict.pop("roc")
        metricsTable[category] = metricsDict

    #Logging / Output

    #print header:
    s_header = f"\t\t\t"+ "\t".join([f"{key}" for key in metricsTable.keys()])
    log(s_header, indentation=indentation+1)
    for metricKey in metricsDict.keys():
        s = f"{metricKey:16}"
        for category in metricsTable.keys():
            value = metricsTable[category][metricKey]
            s += f"\t{value:.4f}"
        log(s, indentation=indentation+1)

    scatterCategories(
        data, targetKey="p", fileName="Scatter_Categories_VS_p.png", violin=True
    )
    # scatterCategories(data, targetKey="label", fileName="Scatter_Categories_VS_label.png", violin=True)
    scatterCategories(
        data, targetKey="dist", fileName="Scatter_Categories_VS_dist.png", violin=True
    )

def analyzeProteinTruncationEffect(
    probabilities, labels, dataSet_df, embeddingFolder, idx, indentation=4
):
    # get where the sequence was truncated
    data = {
        "truncated": {"p": [], "label": [], "trunc_p": []},
        "non-truncated": {
            "p": [],
            "label": [],
            "trunc_p": [],
        },  # does not need truncation percentage as it is not truncated (always 100%)
    }
    for i, id in enumerate(tqdm(idx)):
        sub_df = dataSet_df.iloc[id.item()]
        d = dict(sub_df)  # sample as dict (columns=keys, cells=values)
        annotations = d["annotations"]
        p = probabilities[i]
        label = labels[i]
        y_ref = bool(d["positive"])

        # just sanity check if that really is the correct sample
        assert (
            y_ref == label
        ), f"For idx={id}, the dataset label and the table labe missmatch ( {label} != {y_ref})"

        # get embedding
        embeddingPath = embeddingFolder.joinpath(d["Gene_ID"])
        try:
            embedding = torch.load(embeddingPath, map_location=torch.device("cpu"))
        except EOFError as e:
            log(
                f"EOFError while loading embedding: {embeddingPath}\n\t[DELETING RESPECTIVE FILE]"
            )
            embeddingPath.unlink()
            raise e

        # chekc if truncated or not
        emb_len = embedding.shape[0] - 2  # without start and end tokens
        seq_len = len(d["sequence"])

        if seq_len != emb_len:  # was truncated
            # log(f"{seq_len}\t{emb_len}") #This does not #aggregate data required for statisticalanalysis
            data["truncated"]["p"].append(float(p))
            data["truncated"]["label"].append(bool(label))
            data["truncated"]["trunc_p"].append(float(emb_len / seq_len))
        else:  # not truncated
            data["non-truncated"]["p"].append(float(p))
            data["non-truncated"]["label"].append(bool(label))
            data["truncated"]["trunc_p"].append(float(emb_len / seq_len))

    # Create table of metrics.
    metricsTable = {}
    for category in data.keys():
        df = data[category]
        N = len(df["p"])
        log(f"{category} (N={N})", indentation=indentation)
        if N < 10:
            log(f"Skipping due to sample size", indentation=indentation + 1)
            continue

        metricsDict = getMetrics(
            probabilities=torch.tensor(df["p"]), gt_labels=torch.tensor(df["label"])
        )
        # logMetrics(metricsDict, indentation=5) #will be done for all in a seperate table
        # plotCurves(metricsDict, fileNamePrefix = f"{modelName}__{instanceName}__{ckptName}__")
        metricsDict.pop("prc")
        metricsDict.pop("roc")
        metricsTable[category] = metricsDict

    #Logging / Output
    s_header = f"\t\t\t"+ "\t".join([f"{key}" for key in metricsTable.keys()])
    log(s_header, indentation=indentation+1)
    for metricKey in metricsDict.keys():
        s = f"{metricKey:16}"
        for category in metricsTable.keys():
            value = metricsTable[category][metricKey]
            s += f"\t{value:.4f}"
        log(s, indentation=indentation+1)


    #iterate trough every row of the table and log the metrics

#Unit tests
if __name__ == "__main__":
    from pprint import pformat, pprint

    ## INDIVIDUAL FUNCTION TESTS ##
    if(False):
        # Test calc_errs
        std_mcc, std_bal_acc = calc_errs(tp=100, fp=10, tn=100, fn=10, iterations=1000)
        print(f"calc_errs(tp=100, fp=10, tn=100, fn=10, iterations=1000):")
        print(f"\t-> std_mcc: {std_mcc}, std_bal_acc: {std_bal_acc}")

        # Test getMetricsFromPreds
        gt_labels =     torch.tensor( [1, 0, 1, 0, 1, 0, 1, 0] )
        pred_labels =   torch.tensor( [1, 1, 1, 1, 0, 0, 0, 0] )
        # -> TP=2, TN=2, FP=2, FN=2
        metricsDict = getMetricsFromPreds(pred_labels=pred_labels, gt_labels=gt_labels, )
        print(f"getMetricsFromPreds(pred_labels={pred_labels}, gt_labels={gt_labels}):")
        print(f"\t-> {pformat(metricsDict)}")


    ## TESTING REAL MODEL ##

    # Initialize global environment and import useful utility functions
    import sys
    from pathlib import Path
    sys.path.append(str(Path(".").absolute()))
    from scripts.initialize import *
    initialize(__file__)


    # Get model
    modelName = "Peng"
    ckptPath = MODELS.joinpath(f"{modelName}/lightning_logs/LM=esm1b_t33_650M_UR50S-E=bressin19-S=2023-E=30-BS=1024/checkpoints/epoch=29-step=180-val_loss=0.1973.ckpt")
    #ckptPath = MODELS.joinpath("Peng/lightning_logs_pre2023-10-17/LM=esm1b_t33_650M_UR50S-E=bressin19-S=2023-E=100-BS=512-ckpt=epoch=99-step=300-val_loss=0.1900.ckpt/checkpoints/epoch=199-step=1000-val_loss=0.0819.ckpt")
    #modelName = "XGBoost" #TODO: create checkpoints
    #ckptPath = MODELS.joinpath(f"{modelName}/lightning_logs/LM=esm1b_t33_650M_UR50S-E=bressin19-S=2023-E=30-BS=1024/checkpoints/epoch=29-step=180-val_loss=0.1973.ckpt")
    #modelName = "Lora" #TODO
    #modelName = "Linear" #TODO: create checkpoints

    from utils import getModelFromCkpt
    print(f"Loading model at: {ckptPath}")
    model = getModelFromCkpt(params={"model_name":modelName, "checkpoint_path":ckptPath})
    params = model.params
    #pprint(params)
    params["devices"] = [1] if isinstance(params["devices"], str) or isinstance(params["devices"], int) else params["devices"]

    # Get Dataset
    print(f"Get Train & Val Dataset")
    train_set, val_set = getTrainingDatasets(params=params, indentation=3)

    # Run classifier
    print(f"Run classifier")
    probabilities, labels, batch_idx = runClassifier(model, val_set, params)

    #print(f"Probabilities:\n\t{pformat(probabilities)}")
    #print(f"Labels:\n\t{pformat(labels)}")

    #Get Metrics
    metricsDict = getMetrics(probabilities=probabilities, gt_labels=labels)
    #remove roc & prc from dict (to not spam console output)
    metricsDict.pop("roc")
    metricsDict.pop("prc")
    print(f"Metrics:\n{pformat(metricsDict)}")

    