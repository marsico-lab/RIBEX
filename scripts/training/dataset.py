
from torch.utils.data import Dataset
import pickle
from pathlib import Path
import torch
from datasets import DatasetDict
from sklearn.model_selection import train_test_split
import sys
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import * # to get log function
import numpy as np

class DataSet(Dataset):

    def __init__(self, dataSetPath, embeddingFolder, device="cpu"):

        #Read dataset table
        with open(dataSetPath, 'rb') as f:
            self.dataSet_df = pickle.load(f)
            self.dataSet_df = self.dataSet_df.reset_index(drop=True)
        # TODO: check why so many elements are filtered from the original TSV file
        #Data set columns/keys: "Gene_ID", "Gene_Name", "taxon_ID", "canonical", "positive", "annotations", "sequence", "cluster"
        # With annontaion tuples: (fr, to, ty, name, sName) where ty: 0=other,1=RBD,2=IDp
        self.embeddingFolder = embeddingFolder
        self.device = device

        #sanity check if all the required embeddings actually exist
        required = set(self.dataSet_df["Gene_ID"])
        existing = set([ p.name for p in self.embeddingFolder.iterdir()]) # Note: this might be more than "required" bit not all are relevant
        missing = required-existing
        if(len(missing) > 0):
            #raise RuntimeError(f"Not all Gene_IDs in the dataset have a embedding! Missing { (len(missing)/len(required))*100 }% of entries ({len(missing)})")
            # instead of raising an error, we will just log/print a warning and run on the existing files
            log(f"{'*' * 100}")
            log(f"WARNING: Not all Gene_IDs in the dataset have a embedding! Missing { (len(missing)/len(required))*100 }% of entries ({len(missing)})")
            log(f"WARNING: Working now on { (1-len(missing)/len(required))*100 }% ({len(existing)}) of the requested entries")
            log(f"{'*' * 100}")
            
            #remove missing entries from dataframe
            self.dataSet_df = self.dataSet_df[~self.dataSet_df["Gene_ID"].isin(missing)] #remove rows with missing embeddings

    def __getitem__(self, idx, includeEmbedding=True):

        #get sample from sample df
        sub_df = self.dataSet_df.iloc[idx]
        d = dict(sub_df) #sample as dict (columns=keys, cells=values)
        y = int(d["positive"])
    
        if(includeEmbedding): #Get embedding
            embeddingPath = self.embeddingFolder.joinpath(d['Gene_ID'])
            try:
                embedding = torch.load(embeddingPath, map_location=self.device) #attention: this loads from the device it was saved from
            except EOFError as e:
                log(f"EOFError while loading embedding: {embeddingPath}\n\t[DELETING RESPECTIVE FILE]")
                embeddingPath.unlink()
                raise e
            
            x = embedding
            return y, x, idx
        else:
            return y, idx
    
    def __len__(self):
        return len(self.dataSet_df)
    
class DataSet_Residual(DataSet):
    """Takes a normal DataSet but returns the mean embedding instead of the complete embedding matrix
    """

    def __init__(self, dataSetPath, embeddingFolder, device="cpu"):
        super().__init__(dataSetPath, embeddingFolder, device=device)

    def __getitem__(self, idx, includeEmbedding=True):
        if(includeEmbedding):
            y,x,idx = super().__getitem__(idx, includeEmbedding)
            return y, torch.mean(x,axis=0), idx
        else:
            return super().__getitem__(idx, includeEmbedding)
        
class LoraDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, embedding_folder=None, max_length=128):
        with open(dataset_path, 'rb') as f:
            self.dataSet_df = pickle.load(f).reset_index(drop=True)
        
        # --- Filtering logic matching generate_splits.py and DataSet class ---
        if embedding_folder and embedding_folder.exists():
            required = set(self.dataSet_df["Gene_ID"])
            # This can be slow for large folders, but necessary for correctness validation
            existing = set([p.name for p in embedding_folder.iterdir()])
            missing = required - existing
            
            if len(missing) > 0:
                print(f"[{self.__class__.__name__}] WARNING: Not all Gene_IDs have an embedding! Missing {len(missing)} entries.")
                # Remove missing entries
                self.dataSet_df = self.dataSet_df[~self.dataSet_df["Gene_ID"].isin(missing)]
                # Crucial reset to align with split generation indices
                self.dataSet_df = self.dataSet_df.reset_index(drop=True)
                print(f"[{self.__class__.__name__}] Filtered DataFrame shape: {self.dataSet_df.shape}")
        
        self.sequences = self.dataSet_df['sequence'].values
        self.labels = self.dataSet_df['positive'].astype(int).values
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.positional_encodings = None
        self.return_pe = False
        
    def set_positional_encodings(self, pe_tensor, return_pe: bool = True):
        assert len(pe_tensor) == len(self.dataSet_df), "Positional encoding tensor length does not match dataset length."
        self.positional_encodings = pe_tensor
        self.return_pe = return_pe
        

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(sequence, padding='max_length', truncation=True, max_length=self.max_length, return_tensors="pt")
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label)
        }
        if self.return_pe:
            item['positional_encoding'] = self.positional_encodings[idx]
        return item

if __name__ == "__main__": #test case
    # Initialize global environment and import useful utility functions
    import sys
    from pathlib import Path
    sys.path.append(str(Path(".").absolute()))
    from scripts.initialize import *
    initialize(__file__)

    params = {
        "LM_name": "esm1b_t33_650M_UR50S",
        "embeddingSubfolder": "bressin19",
        "data_set_name": "bressin19_human_pre-training.pkl"
    }

    embeddingFolder = DATA.joinpath(params["LM_name"]).joinpath(params["embeddingSubfolder"])
    dataSetPath = DATA_SETS.joinpath(params["data_set_name"])

    ds = DataSet(dataSetPath, embeddingFolder)

    #log(ds[10]) #this only works on GPU because the tensors were generated on GPU

class DataSet_PE(DataSet_Residual):
    """
    Like DataSet_Residual (returns pooled embeddings) but also returns positional encodings.
    """
    def __init__(self, dataSetPath, embeddingFolder, device="cpu"):
        super().__init__(dataSetPath, embeddingFolder, device=device)
        self.positional_encodings = None
        self.return_pe = False

    def set_positional_encodings(self, pe_tensor, return_pe: bool = True):
        assert len(pe_tensor) == len(self.dataSet_df), "Positional encoding tensor length does not match dataset length."
        self.positional_encodings = pe_tensor
        self.return_pe = return_pe

    def __getitem__(self, idx, includeEmbedding=True):
        # DataSet_Residual returns (y, x_pooled, idx)
        item = super().__getitem__(idx, includeEmbedding)
        
        if includeEmbedding:
            y, x, idx = item
            pe = torch.tensor([0.0]) # Dummy tensor for collation
            if self.return_pe and self.positional_encodings is not None:
                pe = self.positional_encodings[idx]
            return y, x, pe, idx
        else:
            return item

