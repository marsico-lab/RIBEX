from Bio import SeqIO #to read fasta file
from bioservices import QuickGO
from bioservices import UniProt #to read other stuff?
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#### TASK 1 & 2 ####

#generates a list of length "length" with labels "train", "val", "test" according to the
# percentual split [split]
# (this function is meant to be used to create a new column in a pandas dataframe)
def getSplitLabels(length, split, shuffle=True):
    lenTest = int(length*split[2])
    lenVal = int(length*split[1])
    lenTrain = length-(lenTest+lenVal)

    output = ["train"]*lenTrain+["val"]*lenVal+["test"]*lenTest

    if(shuffle):
        np.random.shuffle(output)
    
    return output

#function to manually read the fasta files from bressin et al. (the existing methods from pandas I did not like)
def readFasta(filePath,seqRange=(None,None)):
    fastaFile = SeqIO.parse(open(filePath),'fasta')
    
    d = {
            #"database": [],
            "id": [],
            #"shortDescription": [],
            #"description": [],
            "seq": []
        }
    
    for fasta in list(fastaFile):
        description, sequence = fasta.description, str(fasta.seq)
        attrib = description.split("|")
        
        seqLen = len(str(sequence))
        
        if(seqRange[0]!=None and seqRange[0]>seqLen): #skip to short
            continue
        elif(seqRange[1]!=None and seqRange[1]<seqLen): #skip to long
            continue

        #d["database"].append(attrib[0])
        d["id"].append(attrib[1])
        #d["shortDescription"].append(attrib[2])
        #d["description"].append(attrib[3])
        d["seq"].append(str(sequence))
        
    #Alternative ways to read fasta:

    #fasta_sequences = SeqIO.parse(open(fastaFile),'fasta')
    #fastaList = list(fasta_sequences) #as list

    #fasta_sequences = SeqIO.parse(open(fastaFile),'fasta')
    #fastaDict = SeqIO.to_dict(fasta_sequences) #as dict
      
    return d       

#reads the Bressin et al. dataset from the positives and negatives file  
def getData(fileNameRBP, fileNameNRBP, shuffle=True, split=(0.8, 0.1, 0.1), seqRange=(None,None)):
    
    #read and split RBP data
    RBP = readFasta(fileNameRBP,seqRange=seqRange)
    RBP["label"] = [1]*len(RBP["seq"]) # add RBP label
    RBP["set"] = getSplitLabels(len(RBP["seq"]),split,shuffle=shuffle)
    
    #read and split non RBP data
    NRBP = readFasta(fileNameNRBP,seqRange=seqRange)
    NRBP["label"] = [0]*len(NRBP["seq"]) # add NRBP label
    NRBP["set"] = getSplitLabels(len(NRBP["seq"]),split,shuffle=shuffle)
    
    #make one big dict with all teh values
    pdDict = RBP
    for key in pdDict.keys():
        pdDict[key].extend(NRBP[key])
    
    #make a dataframe out of it and return it
    #index = np.arange(len(pdDict["seq"]))
    #if(shuffle):
    #    index = np.random.permutation(len(index))
    
    return pd.DataFrame(data=pdDict)#, index=index)


#### TASK 3 ####
import os
import torch
from tqdm import tqdm

#export relevant (pandas df) data to one FASTA file
# return which sequence IDs need still encoding
def generateEncodingInput(dataPreEncoding, fastaPath, encodingFolder):
    
    #filter out already exisiting encodings    
    if not os.path.exists(encodingFolder):
        print("Creating folder "+str(encodingFolder))
        os.makedirs(encodingFolder)
    filenames = os.listdir(encodingFolder)
    existingIDs = [ name.split(".")[0] for name in filenames]

    print("Encodings required: "+str(len(dataPreEncoding)))
    print("Already encoded: "+str(len(existingIDs)))
    
    #manually export to fasta
    outputIDs = []
    with open(fastaPath, "w") as f:
        for name, seq in dataPreEncoding:
            if name in existingIDs:
                #print(name)
                existingIDs.remove(name)
                continue
            outputIDs.append(name) 
            f.write(">"+name+"\n")
            f.write(seq+"\n")
    
    #print statistics
    print("Encodings left to generate: "+str(len(outputIDs)))            
            
    return outputIDs

#### TASK 4 ####
import esm


#### OTHER STUFF ####

#reformat dicts in folder (this is just a helper function I invoke manually if required)
def reformatDatastructure(targetFolder):
    fileNames = os.listdir(targetFolder)
    for fileName in tqdm(fileNames):
        filePath = targetFolder+fileName
        t = torch.load(filePath)
        t2 = {
            "label":t["label"],
            "mean_representation":t["mean_representations"]
            }

        torch.save(t2,filePath)

#read h5 file from hendrik and format to our datastructure
import h5py
def h5toPt(h5FilePath_residual, h5FilePath_mean, ptFolder):
    """
    Loads the given embeddings into a dict using the protein ids as key
    :param filepath: filepath to the h5 file containing the embeddings
    :return: dict containing the embeddings with the protein id as key
    """
    with h5py.File(h5FilePath_residual, 'r') as f_res:
        with h5py.File(h5FilePath_mean, 'r') as f_mean:
            for i_res, i_mean in tqdm(zip(f_res.keys(),f_mean.keys())):
                label_res = f_res[i_res].attrs["original_id"]
                label_mean = f_mean[i_mean].attrs["original_id"]
                if(label_res!=label_mean):
                    print("missmatch:"+str(label_res)+" vs "+str(label_mean))

                d = {
                    "label":label_mean,
                    "representation":torch.tensor(f_res[i_res]),
                    "mean_representation":torch.tensor(f_mean[i_res])
                }

                torch.save(d, ptFolder+label_mean+".pt")

#meanEmbeddingFilePath = "ml4rg_g8_2/data/hendrik/reduced_embeddings_file_hendrik.h5"
#residualEmbeddingFilePath = "ml4rg_g8_2/data/hendrik/embeddings_file_hendrik.h5"              
#h5toPt(residualEmbeddingFilePath, meanEmbeddingFilePath, "ml4rg_g8_2/data/emb_tbr_tstl/" )


#reduce dataset (remove residual)

def reduceData(folder1="emb_esm1b_trunc_pretrain/", folder2 = "emb_esm1b_trunc_pretrain_reduced/"):
    for filename in tqdm(os.listdir(folder1)):
        t = torch.load(folder1+filename)
        torch.save({"label":t["label"],"mean_representation":t["mean_representation"]},folder2+filename)