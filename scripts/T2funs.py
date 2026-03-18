import os
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_curve,roc_curve,auc,roc_auc_score

#compute certrainty of MCC and BACC
def calc_errs(tp, fp, tn, fn): #This is a modified version of a function I have from Henrik!
    """
    Calculates the Error estimates for a variety of performance measurements from the given confusion matrix counts
    :param tp: True Positives
    :param fp: False Positives
    :param tn: True Negatives
    :param fn: False Negatives
    :return: MCC-error, BACC-error
    """
    iterations = 1000
    data = ["tp"]*tp + ["fp"]*fp + ["tn"]*tn + ["fn"]*fn

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
            import random as rnd
            pick = rnd.randint(0, tp + fp + tn + fn - 1)
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
        mccs.append( ((loc_tp * loc_tn) - (loc_fp * loc_fn))/np.sqrt((loc_tp + loc_fp)*(loc_tp + loc_fn)*(loc_tn + loc_fp)*(loc_tn + loc_fn)) )
        bal_accs.append( 0.5*( loc_tp/(loc_tp+loc_fn) +  loc_tn/(loc_tn+loc_fp) ) )
                        
    # calculate standard deviation of the performance measurements
    std_mcc = np.std(mccs)
    std_bal_acc = np.std(bal_accs)
                        
    return std_mcc, std_bal_acc


#compute statistics
def getStats(all_labels, all_softMaxed, TP, FP, TN, FN):
    precisions, recalls, PRC_thresholds = precision_recall_curve(all_labels,all_softMaxed)
    fpr, tpr, ROC_thresholds = roc_curve(all_labels,all_softMaxed)

    stats = {
        "TP":TP,
        "TN":TN,
        "FP":FP,
        "FN":FN,
        "BACC": 0.5*( TP/(TP+FN) +  TN/(TN+FP) ),
        "MCC":((TP * TN) - (FP * FN))/np.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)),
        "PRC":(precisions, recalls, PRC_thresholds),
        "AUPRC": auc(recalls, precisions), #Area under PRecision Recall curve
        "ROC":(fpr, tpr, ROC_thresholds),
        "AUROC": auc(fpr, tpr) #Area under Reciever-operator curve
    }
    return stats

#plot precision recall curve
def plotPRC(precisions, recalls):
    plt.figure()
    plt.title("PRC")
    plt.plot(recalls, precisions)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()

#Plot receiver operating characteristic
def plotROC(FPR, TPR):
    plt.figure()
    plt.title("ROC")
    plt.plot(FPR, TPR)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()


#### OTHER STUFF ####

#fixing previous mistakes
def bugFix(encodingFolder):

    encodings = []

    for filename in tqdm(os.listdir(encodingFolder)):

        t = torch.load(encodingFolder+filename)
        label = t["label"]
        

        f_mean = t["mean_representation"]

        if(len(f_mean.shape)>1):
            print(label)
            f_mean = torch.mean(f_mean,axis=0)

        d = {
            "label":label,
            "representation":t["representation"],
            "mean_representation":torch.tensor(f_mean)
        }

        #torch.save(d, encodingFolder+filename)

#bugFix("ml4rg_g8_2/data/emb_tbr_tstl/")
               
def readEncoding(encodingFolder):
    encodings = []

    for filename in tqdm(os.listdir(encodingFolder)): #6,4
        t = torch.load(encodingFolder+filename)
        name = t["label"]
        print(name)
        encoding = t["representations"]
        
        print(encoding.shape)
        encodings.append([name,encoding])
    return encodings

#encodings=readEncoding(encodingFolder) # we do not want to do this as the encodings are rather big

