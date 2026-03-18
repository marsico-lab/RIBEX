#based on "Integradted_gradients_fun.py"

import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *
from captum.attr import IntegratedGradients
import argparse
from natsort import natsorted


##### 3.1) Attribution values - from our classififer #####

def parseArguments():
    """Parse commandline arguments for IG generation script and return them as "params" dictionary"""

    parser = argparse.ArgumentParser(
                        prog='generate.py',
                        description='Generate IG values given a classifier model')

    #parser.add_argument('-D', '--device', dest='device', action='store',
    #                    help='LM device, either "cuda" for GPU & VRAM or "cpu" for CPU & RAM',
    #                    default="cpu")
    parser.add_argument(
            "-D", "--device", dest="device", action="store", type=str, help='Torch device', default="cuda:3" #"auto" uses all
        )
    parser.add_argument(
            "-M", "--modelName", dest="model_name", action="store", help="Model to be trained. See scripts/models for options", default="Peng"
        )
    parser.add_argument(
        "-cpt",
        "--checkpoint-folder",
        dest="checkpoint_folder",
        action="store",
        help="Name of model instance folder where the latest checkpoint shall be used."
    )
    parser.add_argument(
            "-lm",
            "--languageModel",
            dest="LM_name",
            action="store",
            help="Language Model. Options: esm1b_t33_650M_UR50S, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D, esm2_t48_15B_UR50D",
            default="esm1b_t33_650M_UR50S",
        )
    parser.add_argument(
            "-ef",
            "--embeddingSubfolder",
            dest="embeddingSubfolder",
            action="store",
            help="Embedding subfolder. Options: bressin19, RIC, InterPro",
            default="bressin19",
        )
    parser.add_argument("-S", "--seed", dest="seed", type=int, default=2023, help="Training seed")
    parser.add_argument("--useToken", dest="useToken", action="store", default=None, help="Token from which to create baseline AA sequence for integrated gradients. Options: mask, pad, unk, cls, eof. If not provided, scalar is used")
    parser.add_argument("--scalar", dest="scalar", type=float, default=None, help="Scaling factor for the IG attributions (None if not used)")
    parser.add_argument("--maskN", dest="maskN", type=int, default=None, help="Calculate attribution by masking N-tuples of AAs (mutually exclusive with integrated gradients i.e. useToken and scalar)")
    parser.add_argument("--zeroN", dest="zeroN", type=int, default=None, help="Calculate attribution by setting N-tuples of AAs to zero (mutually exclusive with maskN integrated gradients i.e. useToken and scalar)")

    args = parser.parse_args() #parse arguments

    # write commandline arguments to params dict
    params = {}
    for key, value in args._get_kwargs():
        params[key] = value

    assert params["checkpoint_folder"] is not None, f"Please provide a concrete model checkpoint folder (of type {params['model_name']})"
    assert not (params["scalar"] is not None and params["useToken"] != None), "Scalar and useToken are mutually exclusive"
    assert not (params["maskN"] is not None and (params["scalar"] is not None or params["useToken"] != None)), "maskN and scalar/useToken are mutually exclusive"
    assert not (params["zeroN"] is not None and (params["scalar"] is not None or params["useToken"] != None)), "zeroN and scalar/useToken are mutually exclusive"

    return params

def setupFolders(params):
    if( params["useToken"] is not None):
        suffix = params["useToken"]
    elif( params["scalar"] is not None):
        suffix = f"scalar_{params['scalar']}"
    elif( params["maskN"] is not None):
        suffix = f"maskN_{params['maskN']}"
    elif( params["zeroN"] is not None):
        suffix = f"zeroN_{params['zeroN']}"

    # Attributions
    attributionsFolder = ATTRIBUTIONS.joinpath(params["LM_name"]).joinpath(params["embeddingSubfolder"])
    attributionsFolder = attributionsFolder.joinpath(suffix)
    attributionsFolder.mkdir(exist_ok=True, parents=True) # create
    dataSetPath = DATA_SETS.joinpath(params["data_set_name"])

    # IG token embeddings cache
    tokenEmbeddingsFolder = CACHE.joinpath(f"embeddings_{params['useToken']}").joinpath(params["LM_name"]).joinpath(params["embeddingSubfolder"])
    tokenEmbeddingsFolder.mkdir(exist_ok=True, parents=True) # create

    # Classifier Model
    modelFolder = MODELS.joinpath(params["model_name"])
    instanceFolder = modelFolder.joinpath("lightning_logs").joinpath(params["checkpoint_folder"])#
    checkpointFolder = instanceFolder.joinpath("checkpoints")
    if( not checkpointFolder.exists() == True):
            raise RuntimeError(f"No ckpt folder {checkpointFolder}")

    ckptName = natsorted( [path.name for path in checkpointFolder.iterdir()] )[-1] #get last checkpoint
    log(f"\tLatest checkpoint: {ckptName}",indentation=2)

    params['checkpoint_path'] = checkpointFolder.joinpath(ckptName) # Create NEW key value pair
    if params["checkpoint_path"].exists() == False:
        raise RuntimeError(f'Checkpoint does not exist: \"{params["checkpoint_path"]}\"')
    
    #Embeddings
    embeddingFolder = EMBEDDINGS.joinpath(params["LM_name"]).joinpath(params["embeddingSubfolder"])

    return modelFolder, attributionsFolder, dataSetPath, embeddingFolder, tokenEmbeddingsFolder

#modified classifier for IG
# Input: embedding matrix
# Output: binding probability
class IGwrapper_Peng(nn.Module):
    def __init__(self, preModel):
        super(IGwrapper_Peng, self).__init__()
        
        preModel.eval() #needs to be not-train otherwise:
        # ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 1280])
        # which results from the BatchNorm layer when using just one input but having training on
        
        torch.backends.cudnn.enabled = False # needs to be off otherwise:        
        # RuntimeError: cudnn RNN backward can only be called in training mode
        # which result from applying backward() in the GRU (RNN) while training is off (requiremtn from batchNorm) 

        #pretrained classifier
        self.preModel = preModel
        
        #softMax layer
        self.softMax = nn.Softmax(dim=1)
        
    def forward(self, x):
        # x shape : (N, L, E) on params["device"]
        # with:
        # N: batchsize, which is 1 for IG
        # L: sequence Length
        # E: embedding length (1280 for ESM-2)
        x = x.mean(axis = 1) #shape: (N, L, E) -> (N, E)
        
        logits = self.preModel(x) #shape: (N, 2)
        bindingProb = self.softMax(logits) #shape: (N, 2)
        #bindingPred = bindingProb[:,0]<bindingProb[:,1] #bindingPrediction (binary)
        y = bindingProb[:,1] #shape: (N, 2) -> (N)
        
        #print(y)
        #print(y.shape)
        return y

def get_classifier_fun(model, params):
    """takes a model and returns a functions that executes teh model and returns N probability values (for pos class)"""

    if params["model_name"] in ["Linear", "RandomForest", "XGBoost", "Random_SK"]: #sklearn models
        def classifier_fun(seq_embs):  # Scipy models
            # shape(seq_embs) = (N, seq_len, emb_dim)
            residual_emb = torch.mean(seq_embs,axis=1) # get residual embedding -> shape: (N, emb_dim)
            #print(residual_emb.shape)
            preds = model.predict_proba(residual_emb) # shape: (N, 2)
            preds_pos = preds[:, 1] #to make list of values put 1 in brackets
            return preds_pos
        return classifier_fun

    elif params["model_name"] in ["Peng", "Peng6"]:
        wrappedModel = IGwrapper_Peng(model)
        return wrappedModel.forward

    elif params["model_name"] in ["Linear_pytorch"]:
        return model.forward #should result in a probability value already
    
    elif params["model_name"] in ["Lora"]:
        #TODO
        raise NotImplementedError(f"Model \"{params['model_name']}\" not implemented for manual evaluation yet. TODO!")
    
    else:
        raise NotImplementedError(f"Model \"{params['model_name']}\" not implemented for manual evaluation")


#DEPCRECIATED way to get sequences.
# uses the original "data" Dataframe and an additional "lookup" table for sequences/protins that are not in "data"
def doLookup():
    
    seq_AA_lookup = [
        ("Q14671","PUM1_HUMAN","MSVACVLKRKAVLWQDSFSPHLKHHPQEPANPNMPVVLTSGTGSQAQPQPAANQALAAGTHSSPVPGSIGVAGRSQDDAMVDYFFQRQHGEQLGGGGSGGGGYNNSKHRWPTGDNIHAEHQVRSMDELNHDFQALALEGRAMGEQLLPGKKFWETDESSKDGPKGIFLGDQWRDSAWGTSDHSVSQPIMVQRRPGQSFHVNSEVNSVLSPRSESGGLGVSMVEYVLSSSPGDSCLRKGGFGPRDADSDENDKGEKKNKGTFDGDKLGDLKEEGDVMDKTNGLPVQNGIDADVKDFSRTPGNCQNSANEVDLLGPNQNGSEGLAQLTSTNGAKPVEDFSNMESQSVPLDPMEHVGMEPLQFDYSGTQVPVDSAAATVGLFDYNSQQQLFQRPNALAVQQLTAAQQQQYALAAAHQPHIGLAPAAFVPNPYIISAAPPGTDPYTAGLAAAATLGPAVVPHQYYGVTPWGVYPASLFQQQAAAAAAATNSANQQTTPQAQQGQQQVLRGGASQRPLTPNQNQQGQQTDPLVAAAAVNSALAFGQGLAAGMPGYPVLAPAAYYDQTGALVVNAGARNGLGAPVRLVAPAPVIISSSAAQAAVAAAAASANGAAGGLAGTTNGPFRPLGTQQPQPQPQQQPNNNLASSSFYGNNSLNSNSQSSSLFSQGSAQPANTSLGFGSSSSLGATLGSALGGFGTAVANSNTGSGSRRDSLTGSSDLYKRTSSSLTPIGHSFYNGLSFSSSPGPVGMPLPSQGPGHSQTPPPSLSSHGSSSSLNLGGLTNGSGRYISAAPGAEAKYRSASSASSLFSPSSTLFSSSRLRYGMSDVMPSGRSRLLEDFRNNRYPNLQLREIAGHIMEFSQDQHGSRFIQLKLERATPAERQLVFNEILQAAYQLMVDVFGNYVIQKFFEFGSLEQKLALAERIRGHVLSLALQMYGCRVIQKALEFIPSDQQNEMVRELDGHVLKCVKDQNGNHVVQKCIECVQPQSLQFIIDAFKGQVFALSTHPYGCRVIQRILEHCLPDQTLPILEELHQHTEQLVQDQYGNYVIQHVLEHGRPEDKSKIVAEIRGNVLVLSQHKFASNVVEKCVTHASRTERAVLIDEVCTMNDGPHSALYTMMKDQYANYVVQKMIDVAEPGQRKIVMHKIRPHIATLRKYTYGKHILAKLEKYYMKNGVDLGPICGPPNGII")
        ,("Q15717","ELAV1_HUMAN","MSNGYEDHMAEDCRGDIGRTNLIVNYLPQNMTQDELRSLFSSIGEVESAKLIRDKVAGHSLGYGFVNYVTAKDAERAINTLNGLRLQSKTIKVSYARPSSEVIKDANLYISGLPRTMTQKDVEDMFSRFGRIINSRVLVDQTTGLSRGVAFIRFDKRSEAEEAITSFNGHKPPGSSEPITVKFAANPNQNKNVALLSQLYHSPARRFGGPVHHQAQRFRFSPMGVDHMSGLSGVNVPGNASSGWCIFIYNLGQDADEGILWQMFGPFGAVTNVKVIRDFNTNKCKGFGFVTMTNYEEAAMAIASLNGYRLGDKILQVSFKTNKSHK")
        ,("P98175","RBM10_HUMAN","EYERRGGRGDRTGRYGATDRSQDDGGENRSRDHDYRDMDYRSYPREYGSQEGKHDYDDSSEEQSAEDSYEASPGSETQRRRRRRHRHSPTGPPGFPRDGDYRDQDYRTEQGEEEEEEEDEEEEEKASNIVMLRMLPQAATEDDIRGQLQSHGVQAREVRLMRNKSSGQSRGFAFVEFSHLQDATRWMEANQHSLNILGQKVSMHYSDPKPKINEDWLCNKCGVQNFKRREKCFKCGVPKSEAEQKLPLGTRLDQQTLPLGGRELSQGLLPLPQPYQAQGVLASQALSQGSEPSSENANDTIILRNLNPHSTMDSILGALAPYAVLSSSNVRVIKDKQTQLNRGFAFIQLSTIVEAAQLLQILQALHPPLTIDGKTINVEFAKGSKRDMASNEGSRISAASVASTAIAAAQWAISQASQGGEGTWATSEEPPVDYSYYQQDEGYGNSQGTESSLYAHGYLKGTKGPGITGTKGDPTGAGPEASLEPGADSVSMQAFSRAQPGAAPGIYQQSAEASSSQGTAANSQSYTIMSPAVLKSELQSPTHPSSALPPATSPTAQESYSQYPVPDVSTYQYDETSGYYYDPQTGLYYDPNSQYYYNAQSQQYLYWDGERRTYVPALEQSADGHKETGAPSKEGKEKKEKHKTKTAQQIAKDMERWARSLNKQKENFKNSFQPISSLRDDERRESATADAGYAILEKKGALAERQHTSMDLPKLASDDRPSPPRGLVAAYSGESDSEEEQERGGPEREEKLTDWQKLACLLCRRQFPSKEALIRHQQLSGLHKQNLEIHRRAHLSENELEALEKNDMEQMKYRDRAAERREKYGIPEPPEPKRRKYGGISTASVDFEQPTRDGLGSDNIGSRMLQAMGWKEGSGLGRKKQGIVTPIEAQTRVRGSGLGARGSSYGVTSTESYKETLHKTMVTRFNEAQ")
        ,("P35637","FUS_HUMAN","MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNSSSGGGGGGGGGGNYGQDQSSMSSGGGSGGGYGNQDQSGGGGSGGYGQQDRGGRGRGGSGGGGGGGGGGYNRSSGGYEPRGRGGGRGGRGGMGGSDRGGFNKFGGPRDQGSRHDSEQDNSDNNTIFVQGLGENVTIESVADYFKQIGIIKTNKKTGQPMINLYTDRETGKLKGEATVSFDDPPSAKAAIDWFDGKEFSGNPIKVSFATRRADFNRGGGNGRGGRGRGGPMGRGGYGGGGSGGGGRGGFPSGGGGGGGQQRAGDWKCPNPTCENMNFSWRNECNQCKAPKPDGPGGGPGGSHMGGNYGDDRRGGRGGYDRGGYRGRGGDRGGFRGGRGGGDRGGFGPGKMDSRGEHRQDRRERPY")

    ]
    #which sequence/protein

    #seq_id = "Q8TB72" #UniProtKB|Q8TB72|PUM2|Pumilio homolog 2
    seq_id = "Q14671" #UniProtKB|Q14671|PUM1|Pumilio homolog 1
    #seq_id = "Q15717" # ELAV1_HUMAN has RRM
    #seq_id = "P98175" # RBM10_HUMAN 
    #seq_id = "P35637" # FUS_HUMAN 1x RRM1

    seq_AA = None

    #search in data
    r = data.loc[data["id"]=="Q14671"]["seq"]
    if len(r) > 0:
        seq_AA = r.item()
    else: #otherwise search in lookup
        for AA in seq_AA_lookup:
            if AA[0] == seq_id:
                seq_AA = AA[2]
                break

    if seq_AA == None:
        print("SEQUENCE NOT FOUND. Please add manaually to lookup")
    else:
        print(f"ID:{seq_id}\nseq:{seq_AA}\nseq_len:{len(seq_AA)}")

#Create baseline embedding 
def getBaselineEmbedding(
    seq,
    batch_converter,
    LM,alphabet,repr_layer,
    tokenEmbeddingsFolder, #if the embedding exists in this folder we do not need to generate it again (speedup!)
    device,
    useToken = "", #if not "", make sequence out of token repititions
                #if "": use 0s-vector
    ):
    
    if(useToken is not None): 

        maskLen = len(seq)
        filePath = tokenEmbeddingsFolder.joinpath(f"{maskLen}x{useToken}.pt")

        if( os.path.isfile(filePath) ): #embedding already exists
            emb = torch.load(filePath) # just load embedding
        else: #otherwise generate embedding 
            token = f"<{useToken}>" #i.e. "mask" -> "<mask>"
            base_seq = token*maskLen
            base = [(useToken,base_seq)] #required format for batch_converter: (protein name, sequence)
            #print("BASE:",base)

            batch_labels, batch_strs, batch_tokens = batch_converter(base)
            batch_tokens = batch_tokens.to(device)
            #batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            with torch.no_grad():
                #print(f"to {device}")
                #print(batch_tokens.device)
                results_base = LM(batch_tokens, repr_layers=[repr_layer], return_contacts=True)
            emb = results_base["representations"][repr_layer]
            torch.save(emb, filePath) #save to folder
        return emb
    
    else:
        return torch.zeros(len(seq))

#Computes integrated gradients
def getIntegratedGradients(
        ig, classifier_fun,
        gene_id,seq,
        embeddingFolder, #where the embeddings are
        tokenEmbeddingsFolder,
        batch_converter,LM,alphabet,repr_layer,input_emb_dim, #ESM-2 model stuff
        device = "cuda",
        scalar = None, #scaling factor for the IG attributions
        useToken="" # use AA seq of token as baseline not 0-vector, options: <unk>, <cls>, <pad>, <mask>, <eos>
        ):
    #TODO: parallelize
    assert not (scalar is not None and useToken is not None), "scalar and useToken are mutually exclusive" 

    ## Prepare data
    embeddingPath = embeddingFolder.joinpath(gene_id)
    seq_emb = torch.load(embeddingPath, map_location=device)

    #cut longer sequences to LM input length
    if(len(seq_emb) > input_emb_dim):
        seq_emb = seq_emb[:input_emb_dim]

    # verify that sequence and embedding have compatible lengths
    if(len(seq_emb) != len(seq)+2): 
        #print(f"Gene_ID={gene_id}, len(seq_emb)={len(seq_emb)} != len(seq)+2={len(seq)+2}")
        raise RuntimeError(f"Gene_ID={gene_id}, len(seq_emb)={len(seq_emb)} != len(seq)+2={len(seq)+2}")

    seq_embs = torch.unsqueeze(seq_emb,0) # Add batch dimension N = 1
    
    #Get baseline embedding
    if(useToken is not None): #create actual string
        base_embs = getBaselineEmbedding(seq,batch_converter,LM,alphabet,repr_layer,tokenEmbeddingsFolder, device, useToken=useToken)
        base_embs = base_embs.to(device).requires_grad_()
        p_bases = classifier_fun(base_embs)
    elif(scalar == 0): #0 vector
        base_embs = None
        p_bases = classifier_fun(torch.zeros_like(seq_embs))
        #Old version:
        #base_embs = getBaselineEmbedding(seq,batch_converter,LM,alphabet,repr_layer,tokenEmbeddingsFolder, useToken="") # zero vector
    else: # nonzero scalar
        base_embs = scalar
        p_bases = classifier_fun(seq_embs*base_embs)
        #Old version:
        #base_embs=seq_embs*0.99 #the better
    
    ## Get probabilities (for sanity check/analysis later)
    p_seqs = classifier_fun(seq_embs) #sanity check if classifier_fun works

    ## Apply integrated gradients

    #Move to device and make torch
    seq_embs = seq_embs.to(device).requires_grad_()

    attributions, deltas = ig.attribute(    seq_embs,
                                            baselines=base_embs, #can be array or None or scalar
                                            #method='riemann_trapezoid', #default=gausslegendre
                                            #n_steps=100, #default=50
                                            #target=1, #dont really work for us for some reason: AssertionError: Cannot choose target column with output shape torch.Size([50]).
                                            return_convergence_delta=True)
    
    #take from device and make numpy
    attributions = attributions[0].detach().cpu().numpy() 
    IG_delta = deltas[0].detach().cpu().numpy()
    p_base = p_bases[0].detach().cpu().numpy()
    p_seq = p_seqs[0].detach().cpu().numpy()

    IG_attribs_mean = attributions.mean(axis=1,dtype=np.float32)

    return IG_attribs_mean, IG_delta, p_base, p_seq

   
#Plot sum of attribution values versus length of sequences
# -> what is that correlations?! why is that shape emerging?!
def plotAttributionVSLength(
        RBPdomains,
        figureFolder,
        figSize = [6.4*2, 4.8*2],
        ):
    
    sums = []
    colors = []
    lens = []

    for i in RBPdomains.IG_attribs_mean.index:
        
        a = RBPdomains.IG_attribs_mean[i]
        pos = RBPdomains.positive[i]
        
        l = len(RBPdomains.IG_attribs_mean[i])
        lens.append(l)
        colors.append( 1 if pos else 0 )
        s = np.sum(a)
        sums.append(s)
        #print(f"{i}: ({np.min(a):.4f}\t,{np.max(a):.4f})\tsum={s:.4f}\tlen={l}")
            
    plt.figure(figsize=figSize)
    plt.title("sum(Attributions) vs len(Sequences)")
    plt.xlabel("len(Sequence)")
    plt.ylabel("sum(Attribution)")
    plt.scatter(lens,sums,c=colors)
    plt.savefig(figureFolder+f"scatter_sum(Attributions)_len(Sequences)_N={len(lens)}.svg")
    plt.show()

#Attribution value range histograms
def plotAttributionsRangeHistogram(
        RBPdomains,
        figureFolder,
        figSize = [6.4*2, 4.8*2],
        ):
        
    mins = []
    maxs = []
        
    #get min/max values
    for attrib in list(RBPdomains.IG_attribs_mean):
        if len(attrib)==0:
            continue
            
        #get min/max
        mins.append(np.min(attrib))
        maxs.append(np.max(attrib))
        
    #print total range
    print(f"\ttotal range = ({np.min(mins):.2e}, {np.max(maxs):.2e})")
        
    #plot hists
    #m = np.mean(scores)
    #std = np.std(scores)
    
    plt.figure(figsize=figSize)
    plt.hist(mins, bins=50,fc=(0, 1, 1, 0.5),label="minima")#, range=(-3*std+m,3*std+m) )
    plt.hist(maxs, bins=50,fc=(1, 0, 1, 0.5),label="maxima")
    
    plt.title(f"Attribution value ranges")
    plt.xlabel("Min/Max value")
    plt.ylabel("Amount")
    #plt.axvline(x=1.0,color="black",label="x=1")#,ymin=0,ymax=max(h[1]))
    #print(f"{dist} mean  difference: {100*(m-1):.2f} %")

    #plt.axvline(x=m,color="orange",label="mean")

    plt.legend()
    plt.savefig(figureFolder+f"hist_ranges.svg")
        
    plt.show()





