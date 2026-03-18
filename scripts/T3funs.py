import os
import torch
from tqdm import tqdm
import pandas as pd

from model_v1 import dataset_v1, model_v1, model_v1_oon #import model and its dataset structure
from torch.utils.data import DataLoader
import time
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_curve,roc_curve,auc,roc_auc_score

import datetime
from pathlib import Path
import matplotlib.pyplot as plt


#### 1. Pre Training ####
soft_max=torch.nn.Softmax(dim=1)

#one run of a dataset on the model (either training or just evaluating)
def runOn(model, data_loader, criterion_model, dev, train=False, optimizer=None, lr_scheduler=None, device="cuda:0"): #optimizer and scheduler are only necesarry for training

    #t = time.time()

    setSize = len(data_loader.dataset)
    bs = data_loader.batch_size
    
    losses=torch.zeros((setSize),device=device)
    t0=time.time()
    
    all_logits = torch.zeros((setSize,2),device=device)
    all_softMaxed = torch.zeros((setSize,2),device=device)
    all_labels = torch.zeros((setSize),device=device)
    all_preds = torch.zeros((setSize),device=device)
    all_IDs = []
        
    TP, FP, TN, FN = 0,0,0,0
    
    if(train):
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    #print("\t\t\ta: "+str(time.time()-t))
    #t = time.time()
    
    #t2 = time.time()
    for i, (IDs, labels, embs) in enumerate(data_loader): #this takes very long for training set!
        items = len(labels)

        #print("\t\t\t\t0: "+str(time.time()-t2))
        #t2 = time.time()
        embs, labels = embs.to(device), labels.to(device)

        ##print("\t\t\t\t1: "+str(time.time()-t2))
        ##t2 = time.time()

        if(train):
            optimizer.zero_grad() #Zero the gradient

        ##print("\t\t\t\t2: "+str(time.time()-t2))
        ##t2 = time.time()

        logits=model(embs) #run model

        ##print("\t\t\t\t3: "+str(time.time()-t2))
        ##t2 = time.time()

        #Get loss
        loss=criterion_model(logits,labels)
        losses[i*bs:i*bs+items] = loss.item()
        
        ##print("\t\t\t\t4: "+str(time.time()-t2))
        ##t2 = time.time()
        if(train):
            loss.backward()

            #optimize
            optimizer.step()
            lr_scheduler.step(loss.item()) #we want to reduce the lr based on the epochs
        
        ##print("\t\t\t\t5: "+str(time.time()-t2))
        ##t2 = time.time()

        all_IDs.extend(IDs)
        all_logits[i*bs:i*bs+items] = logits #TODO: here they do some wierd shit in the original (division by 0.61 for humans)
        softMaxed = soft_max(logits)
        all_softMaxed[i*bs:i*bs+items] = softMaxed
        all_labels[i*bs:i*bs+items] = labels
        pred = softMaxed[:,0]<softMaxed[:,1]
        all_preds[i*bs:i*bs+items] = pred
        
        TP += sum(torch.logical_and(labels,pred))
        TN += sum(torch.logical_and(torch.logical_not(labels),torch.logical_not(pred)))
        FP += sum(torch.logical_and(torch.logical_not(labels),pred))
        FN += sum(torch.logical_and(labels,torch.logical_not(pred)))

        #print("\t\t\t\t6: "+str(time.time()-t2))

    #print("\t\t\tb: "+str(time.time()-t))
    #t = time.time()

    torch.set_grad_enabled(False)
    
    precisions, recalls, PRC_thresholds = precision_recall_curve(all_labels.cpu(),all_softMaxed.cpu()[:,1])
    fpr, tpr, ROC_thresholds = roc_curve(all_labels.cpu(),all_softMaxed.cpu()[:,1])
    TP, FP, TN, FN = TP.cpu(), FP.cpu(), TN.cpu(), FN.cpu()

    stats = {
        "IDs":all_IDs,
        "labels":all_labels.cpu(),
        "logits":all_logits.cpu(),
        "softMaxed":all_softMaxed.cpu(),
        "pred":all_preds.cpu(),
        "TP":TP,
        "TN":TN,
        "FP":FP,
        "FN":FN,
        "loss": torch.mean(losses).cpu(), #loss
        "BACC": 0.5*( TP/(TP+FN) +  TN/(TN+FP) ),
        #"BACC2": balanced_accuracy_score(all_labels.cpu(), all_preds.cpu()), #that is synonemous to the maunal computation
        "MCC":((TP * TN) - (FP * FN))/torch.sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)),
        "PRC":(precisions, recalls, PRC_thresholds),
        "AUPRC": auc(recalls, precisions), #Area under PRecision Recall curve
        "ROC":(fpr, tpr, ROC_thresholds),
        "AUROC": auc(fpr, tpr) #Area under Reciever-operator curve
    }
    
    #print("\t\t\tc: "+str(time.time()-t))

    return stats

def trainModel(params, model=None, oon_version=False):

    #Get datloader
    print("Setting up data loader...")
    #t = time.time()
    train_set = dataset_v1(params["encodingFolder"], params["dataFrame"], settypes=["train"], residual=False)
    train_loader = DataLoader(train_set, batch_size=params["batchsize"], shuffle=True,
        prefetch_factor=params["prefetch_factor"], num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],persistent_workers=params["persistent_workers"],
        pin_memory_device=params["pin_memory_device"])
    
    val_set = dataset_v1(params["encodingFolder"], params["dataFrame"], settypes=["val"], residual=False)
    val_loader = DataLoader(val_set, batch_size=params["batchsize"], shuffle=True,
        prefetch_factor=params["prefetch_factor"], num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],persistent_workers=params["persistent_workers"],
        pin_memory_device=params["pin_memory_device"])
    #print("\tA: "+str(time.time()-t))
    #t = time.time()

    #define model and optimizers
    print("Setting up model, optimizer and lr scheduler...")
    if(model == None):
        if(oon_version): #"one output neuron" version of the calssifier model.
            model = model_v1_oon(hiddem_dim=params["hidden_size"], embedding_dim=params["embedding_dim"]).to(params["device"])
        else:
            model = model_v1(hiddem_dim=params["hidden_size"], embedding_dim=params["embedding_dim"]).to(params["device"])

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=params["patience"], factor=params["factor"], verbose=True)

    #evaluation criterions (requires amount of positives and negatives)
    loss_weight_pos = (train_set.positives+train_set.negatives) / (2.0 * train_set.positives) 
    loss_weight_neg = (train_set.positives+train_set.negatives) / (2.0 * train_set.negatives) 
    crit_weight= torch.tensor([loss_weight_neg,loss_weight_pos]).to(params["device"])
    criterion_model = torch.nn.CrossEntropyLoss(weight=crit_weight,reduction='mean') 
    #print("\tB: "+str(time.time()-t))
    #t = time.time()

    print("Do training and evaluation...")
    all_stats = {"train":[],"val":[]}

    start = time.time()
    for epoch in range(params["epochs"]):
        
        #t2 = time.time()
        #Training set run
        train_stats = runOn(model, train_loader, criterion_model, params["device"], train=True, optimizer=optimizer, lr_scheduler=lr_scheduler)
        #print("\t\trun: "+str(time.time()-t2))
        #t2 = time.time()

        #Validation set run
        val_stats = runOn(model, val_loader, criterion_model, params["device"], train=False)

        all_stats["train"].append(train_stats)
        all_stats["val"].append(val_stats)

        #final prints
        timePerEpoch = (time.time() - start)/(epoch+1)
        ETA = (params["epochs"]-epoch)*timePerEpoch
        #print("\trest: "+str(time.time()-t2))
        
        print(f'EPOCH {epoch} (TPE={timePerEpoch:.2f}s ETA={ETA/60:.2f}min)\t Validation: loss={val_stats["loss"]:.4f} MCC={val_stats["MCC"]:.4f} BACC={val_stats["BACC"]:.4f} AUPRC={val_stats["AUPRC"]:.4f}')

    #print("\tC: "+str(time.time()-t))

    return model, all_stats

def importModel(modelPath, device="cuda:0"):

    f = torch.load(modelPath,map_location=torch.device(device))
        
    stat_dict = f["model"]
    params = f["params"]
    all_stats = f["stats"]

    model = model_v1(hiddem_dim=params["hidden_size"], embedding_dim=params["embedding_dim"]).to(device)
    model.load_state_dict(stat_dict)
    model.eval()

    return model, all_stats, params

#"model" can eitehr be an finished model or a path to a .pt file
#   if you do not load a model from file, please provide stats and params dict
def testModel(model, all_stats, params):
    if(type(model) == str): #need to load model
        model, all_stats, params = importModel(model)

    print("Tesing model...")

    test_set = dataset_v1(params["encodingFolder"], params["dataFrame"], settypes=["test"], residual=False)

    test_loader = DataLoader(test_set, batch_size=params["batchsize"], shuffle=True,
        prefetch_factor=params["prefetch_factor"], num_workers=params["num_workers"],
        pin_memory=params["pin_memory"],persistent_workers=params["persistent_workers"],
        pin_memory_device=params["pin_memory_device"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=params["patience"], factor=params["factor"], verbose=True)

    #evaluation criterions (requires amount of positives and negatives)
    loss_weight_pos = (test_set.positives+test_set.negatives) / (2.0 * test_set.positives) 
    loss_weight_neg = (test_set.positives+test_set.negatives) / (2.0 * test_set.negatives)

    if(type(model)==model_v1_oon):
        crit_weight= torch.tensor([loss_weight_neg,loss_weight_pos]).to(params["device"])
        criterion_model = torch.nn.CrossEntropyLoss(weight=crit_weight,reduction='mean')
    elif(type(model)==model_v1):
        crit_weight= torch.tensor([loss_weight_neg,loss_weight_pos]).to(params["device"])
        criterion_model = torch.nn.CrossEntropyLoss(weight=crit_weight,reduction='mean')
    else:
        raise RuntimeError(f"Unknown model type {type(model)} -> What loss shall we use?!")
    
    t = time.time()

    #just do one evaluatuion run (no epoch needed)
    with torch.no_grad():
        epoch_stats = runOn(model, test_loader, criterion_model, params["device"], train=False)

        all_stats["test"] = epoch_stats
  
    print(f'Test set (t={time.time()-t:.2f}s): loss={epoch_stats["loss"]:.4f} MCC={epoch_stats["MCC"]:.4f} BACC={epoch_stats["BACC"]:.4f} AUPRC={epoch_stats["AUPRC"]:.4f}')

    return model, all_stats
    

#plot Training and Validation Loss curve
def plotTrainVal(stats, modelFolder, name="TrainVal",dpi=200):
    train_loss_curve = [ d["loss"] for d in stats["train"]]
    val_loss_curve = [ d["loss"] for d in stats["val"]]

    plt.figure()
    plt.title("Loss")
    plt.yscale("log")
    plt.plot(train_loss_curve, label="training")
    plt.plot(val_loss_curve, label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(modelFolder+name+".png",dpi=dpi)
    plt.show()

#plot MCC, BACC and AUPRC
def plotMCC_BACC_AUPR(stats, modelFolder, name="MCC_BACC_AUPRC",dpi=200):
    train_BACC_curve = [ d["BACC"] for d in stats["train"]]
    val_BACC_curve = [ d["BACC"].cpu() for d in stats["val"]]
    train_AUPRC_curve = [ d["AUPRC"] for d in stats["train"]]
    val_AUPRC_curve = [ d["AUPRC"] for d in stats["val"]]
    train_MCC_curve = [ d["MCC"] for d in stats["train"]]
    val_MCC_curve = [ d["MCC"] for d in stats["val"]]

    plt.figure()
    plt.title("MCC, BACC, AUPRC")
    plt.plot(train_BACC_curve, label="BACC (training)")
    plt.plot(val_BACC_curve, label="BACC (validation)")
    plt.plot(train_AUPRC_curve, label="AUPRC (training)")
    plt.plot(val_AUPRC_curve, label="AUPRC (validation)")
    plt.plot(train_MCC_curve, label="MCC (training)")
    plt.plot(val_MCC_curve, label="MCC (validation)")
    plt.xlabel("Epoch")
    #plt.ylabel("Loss")
    plt.legend()
    plt.savefig(modelFolder+name+".png",dpi=dpi)
    plt.show()


#plot precision recall curve
def plotPRC(stats, modelFolder, name="PRC",dpi=200):
    p, r, t = stats["val"][-1]["PRC"]
    plt.figure()
    plt.title("PRC")
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    #plt.legend()
    plt.savefig(modelFolder+name+".png",dpi=dpi)
    plt.show()

#Plot receiver operating characteristic
def plotROC(stats, modelFolder, name="ROC",dpi=200):
    FPR, TPR, t = stats["val"][-1]["ROC"]
    plt.figure()
    plt.title("ROC")
    plt.plot(FPR, TPR)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    #plt.legend()
    plt.savefig(modelFolder+name+".png",dpi=dpi)
    plt.show()

#creates directory with current timestamp
def createOutputDict(parentFolder,suffix=""):
    now = datetime.datetime.now()
    outputFolder = parentFolder+now.strftime("%Y-%m-%d_%H:%M:%S")+suffix+"/"
    Path(outputFolder).mkdir(parents=True, exist_ok=True)
    return outputFolder

#expoer model with statistics and parameters
def exportModel(dataFolder, model, stats, params, name="model"):
    modelFolder = createOutputDict(dataFolder)
    modelPath = modelFolder+name+".pt"

    print("Saving to: "+modelPath)

    torch.save({"model":model.state_dict(),"stats":stats,'params':params},modelPath)

    return modelFolder



#### OTHER STUFF ####

#fixing previous mistakes


#encodings=readEncoding(params["encodingFolder"]) # we do not want to do this as the encodings are rather big
