# Initialize global environment and import useful utility functions
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *
initialize(__file__)

from scripts.integrated_gradients.utils import parseArguments, setupFolders, get_classifier_fun, getIntegratedGradients
from scripts.training.utils import getDataset, getModelFromCkpt
from scripts.embeddings.utils import getModel

from tqdm import tqdm
import pickle
from captum.attr import IntegratedGradients
import scipy
import matplotlib.pyplot as plt

params = parseArguments()

#Dataset parameters
params["data_set_name"] = f"{params['embeddingSubfolder']}.pkl" #The dataset filename that should be used. See $DATA/data_sets for options

# Print/Log paramaters
log("Parameters:")
for key in params.keys():
    log(f"\t{key}: {params[key]}")

# Seed everything (numpy, torch, lightning, even workers)
#TODO: requires?

modelFolder, attributionsFolder, dataSetPath, embeddingFolder, tokenEmbeddingsFolder = setupFolders(params)
folder_Attributions = attributionsFolder.joinpath("attr_raw")
folder_Attributions.mkdir(exist_ok=True, parents=True)

#Read dataset table
with open(dataSetPath, 'rb') as f:
    dataSet_df = pickle.load(f)
#Data set columns/keys: "Gene_ID", "Gene_Name", "taxon_ID", "canonical", "positive", "annotations", "sequence", "cluster"
# With annotation tuples: (fr, to, ty, name, sName) where ty: 0=other,1=RBD,2=IDp

#sanity check if all the required embeddings actually exist

#required = set(dataSet_df["Gene_ID"])
GeneIDs_positive = set(dataSet_df.loc[dataSet_df["positive"] == True]["Gene_ID"]) #only process positives
required = GeneIDs_positive

existing = set([ p.name for p in folder_Attributions.iterdir()]) # Note: this might be more than "required" bit not all are relevant
missing = required-existing  # todo = required - everyFile
done = set.intersection(required,existing)

log(f"\t\t{len(done)} of {len(required)} ({ (len(done)/len(required))*100 :.06f} %) required files exist in {folder_Attributions}")

# Load model
model = getModelFromCkpt(params)

log(f"\tLoaded model from {params['checkpoint_path']}")
log(f"\t\tLoaded Params:")
for key in model.params.keys():
    log(f"\t\t\t{key}: {model.params[key]}")

classifier_fun = get_classifier_fun(model,params)


#get embedding
#missing = ["Q8TB72"]#,"P35637"] #DEBUGGING
plot = False #DEBUGGING

LM, alphabet, repr_layer, input_emb_dim = getModel(params["LM_name"], params["device"])
batch_converter = alphabet.get_batch_converter()
ig = IntegratedGradients(classifier_fun, multiply_by_inputs=True) #multiply_by_inputs=True is default btw.
# multiply_by_inputs=False will result in a constant (horizontal) attribution line

for gene_id in tqdm(missing):
    #print(f"Processing {gene_id}")

    if False: #debugging code for classifier_fun
        embeddingPath = embeddingFolder.joinpath(gene_id)
        seq_emb = torch.load(embeddingPath, map_location="cpu")#, map_location=params["device"])
        seq_embs = torch.unsqueeze(seq_emb,0)
        print(seq_embs.shape)
        ps = classifier_fun(seq_embs)
        print(f"p:{ps}")

    #Get sequence from dataSet_df where gene_id matches
    seq = dataSet_df.loc[dataSet_df["Gene_ID"]==gene_id]["sequence"].values[0]

    #cut longer sequences to LM input length
    if(len(seq)+2 > input_emb_dim):
        seq = seq[:input_emb_dim-2] 

    if( params["useToken"] != None or params["scalar"] != None): #Do interated gradients for attributions
        attribs_mean, IG_delta, p_base, p_seq = getIntegratedGradients(
            ig, classifier_fun,
            gene_id,seq,embeddingFolder, 
            tokenEmbeddingsFolder, #where the masked embeddings are chached
            batch_converter,LM,alphabet,repr_layer,input_emb_dim, #ESM-2 model stuff
            device = params["device"],
            scalar = params["scalar"], #scaling factor for the IG attributions
            useToken = params["useToken"], #use AA seq of token as baseline not 0-vector, options: <unk>, <cls>, <pad>, <mask>, <eos>
        )
        #print(f"IG_attribs_mean: {IG_attribs_mean}")
        #print(f"IG_delta: {IG_delta}")

        ## Saving
        d = {
                "IG_attribs_mean": attribs_mean, 
                "IG_delta": IG_delta,
                "p_base": p_base,
                "p_seq": p_seq
            }
        
    elif params["maskN"] != None or params["zeroN"] != None: #Do attribution by masking / zeroing each AA-tuple
        key = "maskN" if params["maskN"] != None else "zeroN"
        
        #sliding window of masks
        N = params[key] 
        assert len(seq) >= N, f"Sequence too short for {key}={N} (len(seq)={len(seq)})"

        len_out = len(seq)-N+1 # how many values do we get from the sliding window
        
        #get original sequence prediction
        seq_emb = torch.load(embeddingFolder.joinpath(gene_id), map_location=params["device"])
        seq_emb = torch.unsqueeze(seq_emb,0)
        p_base = classifier_fun(seq_emb)


        #get position predictions
        if( key == "maskN"):
            mask = "<mask>"*N    
            sequences = [seq[:i]+mask+seq[i+N:] for i in range(len_out) ]

            # get embeddings - process in batches
            seq_embs = []
            bs = 2 #how much can ESM process at once?
            #print("Getting embeddings")
            for i in range(0,len_out,bs):
                batch = [ (str(i),seq) for seq in sequences[i:i+bs] ] #because batch converter requires format:[(id,seq),...]
                batch_labels, batch_strs, batch_tokens = batch_converter(batch)
                batch_tokens = batch_tokens.to(params["device"])
                #batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
                with torch.no_grad():
                    #print(f"to {device}")
                    #print(batch_tokens.device)
                    results_base = LM(batch_tokens, repr_layers=[repr_layer], return_contacts=True)
                batch_seq_embs = results_base["representations"][repr_layer]
                seq_embs.extend(batch_seq_embs)
            seq_embs = torch.stack(seq_embs)
        
        elif( key == "zeroN"):
            #Add sequence where zero
            seq_embs = []
            for i in range(len_out):
                sem_emb_mod = seq_emb[0].clone()
                sem_emb_mod[i:i+N,:] = 0 #set position dimension to zero
                seq_embs.append(sem_emb_mod)
            seq_embs = torch.stack(seq_embs)
                

        # get predictions - process in batches
        #print("Getting predictions")
        bs = 2 #how much can our classifier process at once?
        seq_ps = torch.zeros(len_out)
        for i in range(0,len_out,bs):
            batch = seq_embs[i:i+bs]
            ps = classifier_fun(batch)
            seq_ps[i:i+bs] = ps

        #compute deltas -> attributions
        p_base = p_base.cpu().detach().numpy()[0]
        seq_ps = seq_ps.cpu().detach().numpy()
        #print(f"p_base: {p_base}")
        #print(f"seq_ps: {seq_ps}")
        delta_ps = seq_ps - p_base # how does p change if we mask the AA
        attribs_mean = -delta_ps # if AAs were important for binding, delta_p<0 -> attribution>0

        d = {
            "attribs_mean": attribs_mean, 
            "p_base": p_base,
            "ps_seq": seq_ps,
            "delta_ps": delta_ps, # = -(ps_seq - p_base)
        }

    ## Plotting (DEBUGGING; proper plotting is done in the analysis scripts!)
    if(plot):
        figSize=[6.4*2, 4.8*2]
        window_size=31
        pol_deg=5
        v_original = attribs_mean

        plt.figure(figsize=figSize)
        plt.plot(v_original, alpha=0.3, color="grey")
        v = scipy.signal.savgol_filter(v_original, window_size, pol_deg) # window size 51, polynomial order 3
        plt.plot(v, color="black")
        plt.title(f"{gene_id}\n(IG_delta = {IG_delta:.04f}")
        plt.ylabel("Attribution")
        plt.xlabel("AA")
        folderTemp = ATTRIBUTIONS.joinpath(f"debugging/{gene_id}/{params['model_name']}_{params['embeddingSubfolder']}")
        folderTemp.mkdir(exist_ok=True, parents=True)
        filePath = folderTemp.joinpath(f"attrib_{params['scalar']:.2f}.png")
        plt.savefig(filePath)
        print(f"Saved plot to {filePath}")
        

    attributionsPath = folder_Attributions.joinpath(gene_id)
    #torch.save(IG_attribs_mean, attributionsPath) # torch save
    with open(attributionsPath, 'wb') as f: #pickle save
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)


