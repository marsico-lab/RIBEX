import os
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle

from model_v1 import dataset_v1, model_v1 #import model and its dataset structure
from torch.utils.data import DataLoader
import time
from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_curve,roc_curve,auc,roc_auc_score

import datetime
from pathlib import Path
import matplotlib
#matplotlib.rcParams['figure.dpi'] = 200

import matplotlib.pyplot as plt
from difflib import SequenceMatcher
import Levenshtein
import scipy
import scipy.signal

import sklearn.manifold #for T-SNE


from T3funs import importModel, soft_max

#### 1. AA subsequences ####
def addColumnsProbChange(model, params, dataFrame, dataSet, lastLayerOutputShape=(80)):
    column_delta_ps = [[]] * len(dataFrame)
    column_delta_logits = [[]] * len(dataFrame)
    column_LL_output = torch.zeros((len(dataFrame),lastLayerOutputShape))
    
    IDlist = list(dataFrame.id.values)
    with torch.no_grad():

        #for getting the last layer output
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        model.block2[5].register_forward_hook(get_activation('block2.5')) #last layer befor output
        
        for i, (ID, label, emb) in tqdm(enumerate(dataSet),total=len(dataSet)):

            #generate mean representations leaving out exactly one AA every time
            mean_representations = torch.zeros((len(emb)+1,emb.shape[1]))  #last element is full

            for row_index in range(emb.shape[0]):
                mean_representations[row_index] = torch.mean(emb[torch.arange(0, emb.shape[0]) != row_index], axis=0)
            mean_representations[-1] = torch.mean(emb, axis=0)
                
            mean_representations = mean_representations.to(params["device"])

            #do inference / run model
            logits=model(mean_representations)
            softMaxed = soft_max(logits)

            #set values
            d = dataFrame[dataFrame.id == ID]
            d_i = IDlist.index(ID)

            #last layer output
            column_LL_output[d_i] = activation['block2.5'][-1] #output for whole sequence at the last layer before classification

            #logits
            default_logit = logits[-1][1].to("cpu")
            delta_logit = (default_logit - logits[:-1,1]).to("cpu")
            dataFrame.loc[dataFrame.id==str(ID),["ft_logit"]] = float(default_logit)
            column_delta_logits[d_i] = delta_logit
            
            #softmax
            default_softMax = softMaxed[-1][1].to("cpu")
            delta_p = (default_softMax - softMaxed[:-1,1]).to("cpu")
            dataFrame.loc[dataFrame.id==str(ID),["ft_softMaxed"]] = float(default_softMax)
            column_delta_ps[d_i] = delta_p
    
    dataFrame["ft_delta_p"] = column_delta_ps
    dataFrame["ft_delta_logit"] = column_delta_logits
    dataFrame["ft_LL_output"] = list(column_LL_output.cpu().numpy())

#matrix plot of intensities
def matrixPlotProbabilities(dataFrame, dpi=None):
    
    #min_value = float(min([min(d) for d in subset.ft_delta_logit.values]))
    max_len = max([len(entry) for entry in dataFrame.ft_delta_p.values])
    m = np.zeros((len(dataFrame),max_len))

    for i, e in enumerate(dataFrame.ft_delta_logit.values):
        m[i][0:len(e)] = e#+min(e)

    if( dpi != None):
        matplotlib.rcParams['figure.dpi'] = dpi
    #plt.figure()
    plt.matshow(m,cmap = "afmhot_r")
    #plt.show()

#get subsequence from seq at position with a specific width
# retruns; subsequence-string and delta_p
def getSS(values, seq, position, width):
    off = int((width-1)/2)
    i1 = position-off if position-off >= 0 else 0
    i2 = position+off
    
    vs = values[i1:i2]
    p_mean = np.mean(vs)
    
    SS = seq[i1:i2]
    
    return SS, p_mean

def getSSfromTo(values, seq, fromTo):
    i1 = fromTo[0]
    i2 = fromTo[1]
    
    vs = values[i1:i2]
    p_mean = np.mean(vs)
    
    SS = seq[i1:i2]
    
    return SS, p_mean

#get all subsequences with a specific width at a specific positin
def getSSs(dataFrame, SSwidth, peak_width, window_size, pol_deg, pf):
    
    #SSs = {} #Sub-sequences: <seq>: ( <AA substring>, <probability delta> )
    SS_id = []
    SS_string = []
    SS_delta = []
    SS_binding = []
    SS_ppos = []
    SS_pneg = []

    keys = dataFrame.keys()
    for row in tqdm(dataFrame.values):
        
        d = dict(zip(keys,row))
        
        v = d["ft_delta_p"]
        v = scipy.signal.savgol_filter(v, window_size, pol_deg) # window size 51, polynomial order 3

        p = len(v)*pf[1]+pf[0] #prominence for this sequence length

        #SSs[d["id"]] = []
        pos = v.copy()
        pos[pos<0]=0
        peaks_pos = scipy.signal.find_peaks(
            x=pos,
            prominence=p,
            width=peak_width
        )
        
        for i,pos in enumerate(peaks_pos[0]):
            #SS, delta_p = getSS(v,d["seq"],pos,SSwidth)
            fromTo = (peaks_pos[1]["left_bases"][i], peaks_pos[1]["right_bases"][i])
            SS, delta_p = getSSfromTo(v,d["seq"],fromTo)

            #format 1
            #SSs[d["id"]].append((SS, delta_p))
            #print((SS, delta_p))
            
            #format 2
            SS_id.append(d["id"])
            SS_string.append(SS)
            SS_delta.append(delta_p)
            SS_binding.append( d["label"]==1 )

        neg = -v.copy()
        neg[neg<0]=0
        peaks_neg = scipy.signal.find_peaks(
            x=neg,
            prominence=p,
            width=peak_width
        )
        
        for i,pos in enumerate(peaks_neg[0]):
            #SS, delta_p = getSS(v,d["seq"],pos,SSwidth)
            fromTo = (peaks_neg[1]["left_bases"][i], peaks_neg[1]["right_bases"][i])
            SS, delta_p = getSSfromTo(v,d["seq"],fromTo)
            
            #format 1
            #SSs[d["id"]].append((SS, delta_p))
            #print((SS, delta_p))
            
            #format 2
            SS_id.append(d["id"])
            SS_string.append(SS)
            SS_delta.append(delta_p)
            SS_binding.append( d["label"]==1 )

        SS_ppos.append(peaks_pos)
        SS_pneg.append(peaks_neg)

    return (
        np.array(SS_id),
        np.array(SS_string),
        np.array(SS_delta),
        np.array(SS_binding),
        np.array(SS_ppos,dtype=object),
        np.array(SS_pneg,dtype=object)
        )

#get index pairs of sequences over a certain threshold
# returns: first index, second index, similarity
def getPairsOver(similiarity_thr, Y, L):
    row = 0
    row_start = 0
    
    output = []

    for i in np.where(Y>=similiarity_thr)[0]:
        curr_row_length = L-row-1
        diagonal_offset_in_row = i-row_start
        #print("curr_row_length: "+str(curr_row_length))
        #print("i: "+str(i))
        while(diagonal_offset_in_row >= curr_row_length):
            diagonal_offset_in_row -= curr_row_length
            row += 1
            row_start += curr_row_length
            curr_row_length -= 1

        #print("diagonal_offset_in_row: "+str(diagonal_offset_in_row))
        #print("row: "+str(row))
        #print("row_start: "+str(row_start))

        column = row+1+diagonal_offset_in_row

        #print("column: "+str(column))

        #print((row,column))
        #print((SSs[row],SSs[column]))
        #print()
        output.append( ((row,column), Y[i]) )
        
    return output

#create linear function from two datapoints
def getLinear(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    gradient = (y2-y1)/(x2-x1)
    t = y1-(x1*gradient)
    return gradient, t