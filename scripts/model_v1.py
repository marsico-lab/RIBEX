import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils

#The dataloader
import os
from torch.utils.data import Dataset

class dataset_v1(Dataset):


    def __init__(self, ptFolder, df, settypes ,residual=False):
        self.ptFolder = ptFolder
        self.filenames = []
        self.residual = residual
        self.labels = []
        self.positives = 0 # num of positiv samples
        self.negatives = 0 # num of negative samples

        #get Filenames of relevant set
        dfSub = df.loc[ [True if t in settypes else False for t in df.set], ["id","label"]]
        print("Reading filenames for "+str(len(dfSub))+" "+str(settypes)+" samples...")

        filenames = os.listdir(ptFolder)
        for ID, label in dfSub.values:
            name = ID+".pt"
            if( name in filenames ):
                self.filenames.append(name)
                self.labels.append(label)
                if(label == 1):
                    self.positives += 1
                else:
                    self.negatives += 1
            else:
                print("Missing file: "+str(name))
        
        print("Found "+str(len(self.filenames))+" "+str(settypes)+" samples.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filePath = self.ptFolder+self.filenames[idx]

        label = self.labels[idx]

        t = torch.load(filePath)
        ID = t["label"]

        if(self.residual):
            emb = t["representation"]
        else:
            emb = t["mean_representation"]

        if(self.ptFolder.split("/")[-2][:7]=="emb_tbr"):
            emb = emb.float()

        return ID, label, emb

#The Model
class model_v1(nn.Module): #the big model
    def __init__(self, hiddem_dim=320, embedding_dim=1280, num_GRU_layers=6, L=1):
        super().__init__()
        self.hidden_dim = hiddem_dim
        self.GRU_layers = num_GRU_layers
        self.embedding_dim = embedding_dim
        self.L = L
        
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.GRU_layers,
            batch_first=True,
            bidirectional=True, dropout=0.05
            )
        
        self.block1=nn.Sequential( 
            nn.Linear(2*self.hidden_dim+2*self.GRU_layers*self.hidden_dim, self.embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim,int(embedding_dim/2**1)),
            nn.BatchNorm1d(int(embedding_dim/2**1)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim/2**1),int(embedding_dim/2**2)),
            )

        self.block2=nn.Sequential(
            nn.BatchNorm1d(int(embedding_dim/2**2)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim/2**2),int(embedding_dim/2**3)),
            nn.BatchNorm1d(int(embedding_dim/2**3)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim/2**3),int(embedding_dim/2**4)),
            nn.BatchNorm1d(int(embedding_dim/2**4)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim/2**4),2) # 80 -> 2 instead of 64->2
            )
        
    def forward(self, x):
        # N = batch size
        # L = sequence length
        # D = 2 (if bidirectional), 1 otherwise
        # H_in = input_size
        # H_out = hidden_size

        #print("Network input shape:"+str(x.shape))
        x=x.view( x.shape[0], self.L, x.shape[1]) # (N, L) -> (N,L,H_in)
        #print("GRU input shape:"+str(x.shape))

        output, hn = self.gru(x) # (N,L,H_in) -> (N,L,D*H_out) , (D*num_layers, N, H)

        #print("GRU output shape:"+str(output.shape))
        #print("GRU hidden shape:"+str(hn.shape))

        #reformat hidden state
        hn=hn.permute(1,0,2) # (D*num_layers, N, H_out) -> (N, D*num_layers, H_out)
        hn=hn.reshape(hn.shape[0],-1) # (N, D*num_layers, H_out) -> (N, num_layers*D*H_out)
            # -> (12, 3840)
        #print("Flattend GRU hidden shape:"+str(hn.shape))

        #reformat output
        output=output.reshape(output.shape[0],-1) # (N,L,D*H_out) -> (N, L*D*H_out)
            #-> (N, 640)
        #print("Flattend GRU output shape:"+str(output.shape))

        # append flattend hidden state to the output
        output=torch.cat([output,hn],1) # (N, L*D*H_out + num_layers*D*H_out)
        #2*self.hidden_dim+2*self.GRU_layers*self.hidden_dim, self.embedding_dim
        #print("Block1 input shape:"+str(output.shape)) 

        output=self.block1(output)
        #print("Block2 input shape:"+str(output.shape)) 

        return self.block2(output)

#Identical Model but just one output neuron (positiv RBP)
class model_v1_oon(nn.Module): #the big model
    def __init__(self, hiddem_dim=320, embedding_dim=1280, num_GRU_layers=6, L=1):
        super().__init__()
        self.hidden_dim = hiddem_dim
        self.GRU_layers = num_GRU_layers
        self.embedding_dim = embedding_dim
        self.L = L
        
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.GRU_layers,
            batch_first=True,
            bidirectional=True, dropout=0.05
            )
        
        self.block1=nn.Sequential( 
            nn.Linear(2*self.hidden_dim+2*self.GRU_layers*self.hidden_dim, self.embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim,int(embedding_dim/2**1)),
            nn.BatchNorm1d(int(embedding_dim/2**1)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim/2**1),int(embedding_dim/2**2)),
            )

        self.block2=nn.Sequential(
            nn.BatchNorm1d(int(embedding_dim/2**2)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim/2**2),int(embedding_dim/2**3)),
            nn.BatchNorm1d(int(embedding_dim/2**3)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim/2**3),int(embedding_dim/2**4)),
            nn.BatchNorm1d(int(embedding_dim/2**4)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim/2**4),1) # 80 -> 1 instead of 64->1 #HERE is the onlly diference to model_v1
            )
        
    def forward(self, x):
        # N = batch size
        # L = sequence length
        # D = 2 (if bidirectional), 1 otherwise
        # H_in = input_size
        # H_out = hidden_size

        #print("Network input shape:"+str(x.shape))
        x=x.view( x.shape[0], self.L, x.shape[1]) # (N, L) -> (N,L,H_in)
        #print("GRU input shape:"+str(x.shape))

        output, hn = self.gru(x) # (N,L,H_in) -> (N,L,D*H_out) , (D*num_layers, N, H)

        #print("GRU output shape:"+str(output.shape))
        #print("GRU hidden shape:"+str(hn.shape))

        #reformat hidden state
        hn=hn.permute(1,0,2) # (D*num_layers, N, H_out) -> (N, D*num_layers, H)
        hn=hn.reshape(hn.shape[0],-1) # (D*num_layers, N*H_out)
            #-> (12, 3840)
        #print("Flattend GRU hidden shape:"+str(hn.shape))

        #reformat output
        output=output.reshape(output.shape[0],-1) # (N,L,D*H_out) -> (N, D*H_out)
            #-> (N, 640)
        #print("Flattend GRU output shape:"+str(output.shape))

        # append flattend hidden state to the output
        output=torch.cat([output,hn],1) # (*H_out+D*H_out)
        #2*self.hidden_dim+2*self.GRU_layers*self.hidden_dim, self.embedding_dim
        #print("Block1 input shape:"+str(output.shape)) 

        output=self.block1(output)
        #print("Block2 input shape:"+str(output.shape)) 

        return self.block2(output)