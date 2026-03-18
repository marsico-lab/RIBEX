import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
from model_v1 import dataset_v1, model_v1 #import model and its dataset structure
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from esm.pretrained import esm2_t33_650M_UR50D #the ESM2 model
#dir(esm.pretrained) #print available pretrained ESM models

import matplotlib.pyplot as plt
import scipy.signal # for savgol filter
from tqdm import tqdm
import pandas as pd
import json #if we want to save default dicts nicely as JSON
from natsort import natsorted #for nice sorting

##### 3.1) Attribution values - from our classififer #####

#modified classifier for IG
# Input: embedding matrix
# Output: binding probability
class Classifier(nn.Module):
    def __init__(self, preModel):
        super(Classifier, self).__init__()
        
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
    esm2,
    alphabet,
    maskedEmbeddingsFolder, #if the embedding exists in this folder we do not need to generate it again (speedup!)
    masked = False #True = completly masked sequence, False = just zeros,
    ):
    
    if(masked == True): 

        maskLen = len(seq)
        filePath = maskedEmbeddingsFolder+f"{maskLen}x<mask>.pt"

        if( os.path.isfile(filePath) ): #embedding allready exists
            emb = torch.load(filePath) # just load embedding
        else: #otherwise generate embedding 
            mask = "<mask>"
            base_seq = mask*maskLen
            base = [("mask",base_seq)] #required format for batch_converter
            
            batch_labels, batch_strs, batch_tokens = batch_converter(base)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            with torch.no_grad():
                results_base = esm2(batch_tokens, repr_layers=[33], return_contacts=True)

            emb = results_base["representations"][33]

            torch.save(emb, filePath) #save to folder

        return emb
    
    else:
        return torch.zeros_like(seq)


#Computes integraded gradients
def getIntegratedGradients(
        RBPdomains,
        filePath_IG, #where the output dataframe is located/shall be saved
        classifier, #the classififer model
        embeddingFolder, #where the esm-2 embeddings are
        maskedEmbeddingsFolder, #where esm-2 embedding of mask vectors are
        batch_converter,esm2,alphabet, #ESM-2 model stuff
        device = "cuda",
        lengthLimit = 1000,
        masked = True, #use masked AA seq as baseline not 0-vector
        onlyCanonical=True, #only work on canonical insoforms and no others (faster but less samples or course)
        force_generate_ig=False #force regeneration of Integrated gradients. otherwise the filePath_IG will be use if it exists
        ):

    if(force_generate_ig or (os.path.isfile(filePath_IG)==False)): #either force regeneration or gnerate if not existant
    
        print(f"Working on {device}")
        ig = IntegratedGradients(classifier, multiply_by_inputs=True)#multiply_by_inputs=True is default btw.

        #if(not "IG_attribs" in RBPdomains.keys()): #add columns if not existant
        #    RBPdomains["IG_attribs"] = [[]]*len(RBPdomains.index)
        if(not "IG_attribs_mean" in RBPdomains.keys()): #add columns if not existant
            RBPdomains["IG_attribs_mean"] = [None]*len(RBPdomains.index)

        if(not "IG_approx_err" in RBPdomains.keys()):
            RBPdomains["IG_approx_err"] = [None]*len(RBPdomains.index)

        existingFiles = os.listdir(embeddingFolder)

        for i in tqdm(RBPdomains.index):


            #check if canonical
            if(onlyCanonical):
                if list(RBPdomains.canonical)[i]!=True:
                    #print(f"[{i}]: Not canonical")
                    continue
            
            #check if sequence exists
            seq = RBPdomains.Protein_seq[i]
            if(seq == None):
                #print(f"[{i}]: None")
                continue

            #check length Limit
            if(lengthLimit < len(seq)):
                #print(f"[{i}]: len(seq)={len(seq)} > {lengthLimit}")
                continue

            #check if embedding exists
            RBP_Name = RBPdomains.RBP_Name[i]
            Protein_ID = RBPdomains.Protein_ID[i]
            Gene_ID = RBPdomains.Gene_ID[i]

            #filename = f"{Protein_ID}.pt"
            filename = f"{Gene_ID}.pt"
            if(not filename in existingFiles):
                #print(f"No embedding file for protein {RBP_Name}\t with Gene_ID {Gene_ID}. Skipping")
                print(f"[{i}]: No embedding")
                continue

            #read embedding
            seq_emb = torch.load(embeddingFolder+filename)
            
            if(len(seq_emb) != len(seq)+2):
                print(f"RBP_Name={RBP_Name}, Gene_ID={Gene_ID}, len(seq_emb)={len(seq_emb)} != len(seq)={len(seq)}")
            base_emb = getBaselineEmbedding(seq,batch_converter,esm2,alphabet,maskedEmbeddingsFolder, masked=masked)

            #Apply integrated gradients
            #print(f"Processing {filename} \t( {RBP_Name} )")

            if(len(seq_emb.shape) == 2): #add "batch" dimension
                seq_emb = seq_emb[None,:]
                #print(f"New Input shape: {seq_emb.shape}")
            model_input = seq_emb.to(device).requires_grad_()
            model_baseline = base_emb.to(device).requires_grad_()
            #model_baseline=model_input*0.9 #the better


            attributions, deltas = ig.attribute(    model_input,
                                                    baselines=model_baseline,
                                                    method='gausslegendre',
                                                    return_convergence_delta=True)

            attributions = attributions[0].detach().cpu()
            #RBPdomains.at[i,"IG_attribs"] = attributions.type(torch.float32).numpy() #use 32bit float, but is still big!
            attributions = attributions.numpy() 
            RBPdomains.at[i,"IG_attribs_mean"] = attributions.mean(axis=1,dtype=np.float32)

            delta = deltas[0].detach().cpu()
            RBPdomains.at[i,"IG_delta"] = delta.numpy()

        #save
        print(f"Saving IG results to {filePath_IG}")
        RBPdomains.to_pickle(filePath_IG)
        print("done.")
    else:
        print(f"Loading IG results from {filePath_IG}")
        RBPdomains = pd.read_pickle(filePath_IG)

    return RBPdomains
    
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


##### 3.2) Scoring known regions #####


#filter out proteins with:
# - unknown binding domains for evaluation
def removeUnknownBindingDomains(RBPdomains,seqMinLen):
    filtered = [] #list of rows/proteins that fullfill all critera
    for i in RBPdomains.index:
        ok = True
        
        #search for unknwon binding regions
        domains = RBPdomains.domains[i]
        for domain in domains:
            fr, to, ty = domain
            if(ty == "UNKNOWN"):
                ok=False
                break
                
        
            
            
    return RBPdomains[filtered]
            
#evaluate the score of each motif based on the attribution values in this motif region vs outside of the motifs
def getScores(
        RBPdomains,
        figureFolder,
        filePath_eval,
        embeddingFolder,
        force_generate_score=False, #force eval regeneration or just load filePath_eval if it exists
        plot = True,
        figSize=[6.4*2, 4.8*2],

        #Curve Smoothing
        window_size=31,
        pol_deg=5,

        # Attribution generation
        normalizeFeatures = False, #normalization on feature level (based on non-domain region)
        perAxis = True, #Only relevant if normalizeFeatures==True
            #False: mean and std computaion over complete (flat) set of domain.
            #This ignores channels/features and performes significantly worse

        #Attribution Postprocessing
        normalize = False, #for better interpretability how much the signal (domain) deviates from the (noisy) baseline (outside of domain)
        absolute_values = False,
        cutoff_before_compressed = True, #cuts away all attributions below 0 before feture dimension is compressed
        cutoff_after_compressed = False, #cuts away all attributions below 0 after feture dimension is compressed
        flip = False, #mirrows around x axis (+ -> - and - -> +)

        #not reasonable
        sign_corrected = False #change attribution value sign when input was negative -> does not seems to make much diffference
        ):
    
    #For best results set true either:
    # 0V baseline - normalize
    # 0V baseline - normalizeFeatures, perAxis
    # mask basline - cutoff_before_compressed

    if(plot):
        try:
            os.mkdir(figureFolder)
        except FileExistsError:
            pass

    #if(normalizeFeatures):
    #    RBPdomains[f"IG_attribs_norm"] = [None]*len(RBPdomains.index)
    #    #RBPdomains[f"IG_delta_norm"] = [None]*len(RBPdomains.index) #how to compute that properly?

    if(force_generate_score or (os.path.isfile(filePath_eval) == False)):
        #add columns if not existant
        RBPdomains[f"motif_scores"] = [None]*len(RBPdomains.index)       
        
        #if(not f"domain_mask" in RBPdomains.keys()): #we need this one later
        #    RBPdomains[f"domain_mask"] = [None]*len(RBPdomains.index) 
        RBPdomains[f"signalMask_2"] = [None]*len(RBPdomains.index) 
        RBPdomains[f"signalMask_1"] = [None]*len(RBPdomains.index) 
        RBPdomains[f"signalMask_0"] = [None]*len(RBPdomains.index) 
        RBPdomains[f"signalMask_1and2"] = [None]*len(RBPdomains.index) 
        

        for i in tqdm(RBPdomains.index):

            RBP_Name = RBPdomains.RBP_Name[i]
            #Protein_ID = RBPdomains.Protein_ID[i]
            Gene_ID = RBPdomains.Gene_ID[i]
            #Protein_seq = RBPdomains.Protein_seq[i]
            IG_attribs = RBPdomains.IG_attribs_mean[i]
            if(flip):
                IG_attribs *= -1
            #IG_delta = RBPdomains.IG_delta[i]
            domains = RBPdomains.domains[i]
            #pos = RBPdomains.positive[i]
            
            #get masks of annotations
            annotations = {
                "signalMask":{
                    "0":np.zeros(len(IG_attribs)), #other
                    "1":np.zeros(len(IG_attribs)), #RBD
                    "2":np.zeros(len(IG_attribs)) #IDP
                }
            }
            for domain in domains: #print(domain)
                fr,to, ty, name, sName = domain

                if(ty==0): #neither RBD nor IDR
                    annotations["signalMask"]["0"][int(fr):int(to)+1] = 1
                elif(ty==1): #RBD
                    annotations["signalMask"]["1"][int(fr):int(to)+1] = 1
                elif(ty==2): # IDR
                    annotations["signalMask"]["2"][int(fr):int(to)+1] = 1
                else:
                    raise RuntimeError("Unknwon domain type: {ty}")

            annotations["signalMask"]["1and2"] = np.logical_and(annotations["signalMask"]["2"],annotations["signalMask"]["1"])
            annotations["signalMask"]["1or2"] = np.logical_or(annotations["signalMask"]["2"],annotations["signalMask"]["1"])
            annotations["signalMask"]["any"] = np.logical_or(annotations["signalMask"]["0"], annotations["signalMask"]["1or2"])

            RBPdomains.at[i,f"signalMask_2"] = annotations["signalMask"]["2"]
            RBPdomains.at[i,f"signalMask_1"] = annotations["signalMask"]["1"]
            RBPdomains.at[i,f"signalMask_0"] = annotations["signalMask"]["0"]
            RBPdomains.at[i,f"signalMask_1and2"] = annotations["signalMask"]["1and2"]
            RBPdomains.at[i,f"signalMask_1or2"] = annotations["signalMask"]["1or2"]
            RBPdomains.at[i,f"signalMask_any"] = annotations["signalMask"]["any"]
            
            #normalize value distribution based on outside values
            #if(normalizeFeatures): #all numpy operations of nature: matrix - vector will be executed on the tailing dimension (1280)
                #if(perAxis):
                #    mean = baselineValues.mean(axis=0)
                #    std = baselineValues.std(axis=0) #shape (L,1280) -> (1280)
                    #baselineValues = (baselineValues-mean)/std
                #    IG_attribs = (IG_attribs-mean)/std
                #else:
                #    mean = np.mean(baselineValues.flatten())
                #    std = np.sqrt(np.var(baselineValues.flatten())) #shape (L,1280) -> ()
                #    IG_attribs = (IG_attribs-mean)/std
                    
                #RBPdomains.at[i,"IG_attribs_norm"] = IG_attribs
                #print(IG_approx_err)
                #IG_approx_err = (IG_approx_err)/std
                #this does not work becuase what is the single value std if we normalize over all the positions?
                #RBPdomains.at[i,"IG_approx_err_norm"] = IG_approx_err
                
            #change attribution sign for negative inputs
            if(sign_corrected):
                filename = f"{Gene_ID}.pt"
                seq_emb = torch.load(embeddingFolder+filename).numpy()

                IG_attribs = (IG_attribs * np.sign(seq_emb))[0]
            
            #absolute values
            if (absolute_values):
                IG_attribs = np.abs(IG_attribs)
            
            #set 0 what is below 0
            if(cutoff_before_compressed == True):
                IG_attribs[IG_attribs < 0] = 0
                

            #print(f"{Gene_ID} ({RBP_Name})")

                    
            #if(normalize): # outside level used for normalization
            #    outsideValues = v_original[domainMask==0]
            #    v_original = (v_original-np.mean(outsideValues))/np.std(outsideValues)
            v_original = IG_attribs

            if(cutoff_after_compressed == True): # all values < 0 get to be 0
                v_original[v_original < 0] = 0
                
            if plot: #need to create figure here
                plt.figure(figsize=figSize)
                plt.plot(v_original, alpha=0.3, color="grey") #plot non-smooth signal
                    

            ## Protein region analysis ##
            annotations["baseline"] = {}
            annotations["domainValues"] = {}
            annotations["domainMean"] = {}
            annotations["score"] = {"rel":{},"abs":{}}

            #define baseline mask: where neither IDR nor RBD are
            #baselineMask = ~annotations["signalMask"]["1or2"]
            baselineMask = ~annotations["signalMask"]["any"]
            baselineValues = v_original[baselineMask]
            baselineMean = np.mean(baselineValues)
            if plot:
                plt.hlines(y=baselineMean,xmin=0,xmax=len(v_original),color="blue")
            annotations["baseline"]["mask"] = baselineMask
            annotations["baseline"]["values"] = baselineValues
            annotations["baseline"]["mean"] = baselineMean
            RBPdomains.at[i, f"baseline_mask"] = annotations["baseline"]["mask"]
            RBPdomains.at[i, f"baseline_mean"] = annotations["baseline"]["mean"]

            for maskName in annotations["signalMask"].keys(): 

                #complete domain
                domainValues = v_original[annotations["signalMask"][maskName]==1]
                if(len(domainValues) == 0):
                    RBPdomains.at[i,f"score_rel_{maskName}"] = None
                    annotations["score"]["rel"][maskName] = None
                    RBPdomains.at[i,f"score_abs_{maskName}"] = None
                    annotations["score"]["abs"][maskName] = None
                    continue

                domainMean = sum(domainValues)/len(domainValues)
                if plot:
                    if(maskName == "2"):
                        plt.hlines(y=domainMean,xmin=0,xmax=len(v_original),color="tomato")
                    elif(maskName == "1"):
                        plt.hlines(y=domainMean,xmin=0,xmax=len(v_original),color="springgreen")
                    elif(maskName == "1or2"):
                        plt.hlines(y=domainMean,xmin=0,xmax=len(v_original),color="olivedrab")
                    else:
                        pass #dont plot mean for otehr types
                    
                annotations["domainValues"][maskName] = domainValues
                annotations["domainMean"][maskName] = domainMean

                #Percent how many values go into different areas
                # PROBLEM: small VS big areas bias percentage!
                #p_score = sum(domainValues)/sum(baselineValues) #percent domain vs non-domain
                #RBPdomains.at[i,f"score_rel"] = p_score
                
                #Percent how many values go into different areas NORMALIZED for area length (sum/length = mean)
                # PROBLEM: sign of attention values negative peaks and postive peaks are present in dataset!
                if(baselineMean==0):
                    score_rel = None
                else:
                    score_rel = domainMean/baselineMean #- 1 #change in percent 
                RBPdomains.at[i,f"score_rel_{maskName}"] = score_rel
                annotations["score"]["rel"][maskName] = score_rel
                
                abs_score = domainMean-baselineMean #change absolute
                RBPdomains.at[i,f"score_abs_{maskName}"] = abs_score
                annotations["score"]["abs"][maskName] = abs_score

            #motif specific scores
            l = []
            for domain in domains:
                fr, to, ty, name, sName = domain
                
                motifValues = v_original[int(fr):int(to)+1]
                if(len(motifValues) <= 0):
                    print(f"[{i}]: WARNING: {Gene_ID} ({RBP_Name}) has domain {sName} ({name}) from {fr} to {to} (distance = {to-fr}, len(seq)={len(v_original)})")
                    continue
                motifMean = sum(motifValues)/len(motifValues)
                if(baselineMean == 0):
                    score_rel_motif = None
                else:
                    score_rel_motif = motifMean/baselineMean
                abs_score_motif = motifMean-baselineMean
                #print(f"({ty}, {score_rel_motif}, {abs_score_motif})")
                l.append((ty, name, sName, score_rel_motif, abs_score_motif))
                    
            RBPdomains.at[i, f"motif_scores"] = l
                
            #plot black smoothed line
            if plot:
                v = scipy.signal.savgol_filter(v_original, window_size, pol_deg) # window size 51, polynomial order 3
                plt.plot(v, color="black")
                score_rel = annotations["score"]["rel"]["1or2"] #which score is used for naming the file and reporting in figure?
                plt.title(f"{Gene_ID} ({RBP_Name}) \nscore either rel="+(f"{score_rel:.04}" if score_rel != None else "None")+f"\nscore abs={abs_score:.2e}")
                #\n(IG_delta = {IG_delta:.04f})
                plt.ylabel("Attribution")
            
                ## motif specific plot ##
                v_min = np.min(v_original)
                v_max = np.max(v_original)
                for domain in domains:
                    fr, to, ty, name, sName = domain
                    fr = int(fr)
                    to = int(to)

                    if(ty == 0):
                        color="lavender"
                    elif(ty == 1):
                        color="springgreen"
                    elif(ty == 2):
                        color="tomato"

                    motifValues = v_original[fr:to+1]
                    motifMean = sum(motifValues)/len(motifValues)
                    #draw annotation
                    plt.hlines(y=motifMean,xmin=fr,xmax=to,color=color,lw=10,alpha=0.8) #vertical box
                    plt.vlines(x=[fr,to],ymin=v_min, ymax=v_max,color=color, linestyles="dashed")

                    plt.text(x=(fr+to)/2, y=motifMean, s=f"{name}\n({sName})", ha='center', va='center', #text
                                fontsize=8, color="black", fontweight="bold" )

                plt.savefig(
                    figureFolder+f"rel="+(f"{score_rel:.4f}" if score_rel != None else "None")+f"_abs={abs_score:.2e}_{Gene_ID}_{RBP_Name}.svg")
                #plt.show()
                plt.clf()
                plt.close()
                

        #save
        print("Saving Evaluation results")
        RBPdomains.to_pickle(filePath_eval)
    else:
        print("Loading Evaluation results")
        RBPdomains = pd.read_pickle(filePath_eval)
        
    return RBPdomains

#plot score distribution
def plotScoreDistribution(
        RBPdomains,
        figureFolder,
        figSize = [6.4*2, 4.8*2],
        scoreTypes = ["rel","abs"], #proper values: "abs","rel","p"
        regionTypes = ["1","2","1or2","1and2","0"]
        ):
    
    for scoreType in scoreTypes:
        for regionType in regionTypes:
            
            #get non-Null scores
            key = f'score_{scoreType}_{regionType}'
            rows = np.logical_and( pd.isnull(RBPdomains[key]) == False, pd.isnull(RBPdomains[key]) != None)            

            scores_pos = list(RBPdomains.loc[np.logical_and(rows,RBPdomains.positive==True)][key])
            scores_neg = list(RBPdomains.loc[np.logical_and(rows,RBPdomains.positive==False)][key])
            
            #get µ, var and std
            m_pos, m_neg = np.mean(scores_pos), np.mean(scores_neg) 
            #var = np.var(scores)
            std_pos, std_neg = np.std(scores_pos), np.std(scores_neg) 
            
            
            if(scores_pos == []):
                print(f"scores_pos: No data for scoreType {scoreType} and regionType {regionType} !")
                #continue
            else:
                r = [m_pos-3*std_pos, m_pos+3*std_pos]
                
            if(scores_neg == []):
                print(f"scores_neg: No data for scoreType {scoreType} and regionType {regionType} !")
                #continue
            else:
                if m_neg-3*std_neg < r[0]:
                    r[0] = m_neg-3*std_neg
                if m_neg+3*std_neg > r[1]:
                    r[1] = m_neg+3*std_neg
                

            print(f"\ttotal range = ({r[0]:.2e}, {r[1]:.2e})")
        
            #plot histograms
            plt.figure(figsize=figSize)
            if(scores_neg != []):
                plt.hist(scores_neg, bins=100,fc=(0, 1, 0, 0.5),label="non RBP",range=r,
                            weights=100 * (np.ones(len(scores_neg))/len(scores_neg)))
            if(scores_pos != []):
                plt.hist(scores_pos, bins=100,fc=(1, 0, 0, 0.5),label="RBP",range=r,
                            weights=100 * (np.ones(len(scores_pos))/len(scores_pos)))
            plt.title(f"{scoreType} {regionType} distribution\nµ_neg={m_neg:.2f}  (std={std_neg:.2f})\nµ_pos={m_pos:.2f} (std={std_pos:.2f})")
                
            if(scoreType == "rel"):
                plt.xlabel("Score (relative)")
                plt.axvline(x=1.0,color="black",label="x=1")#,ymin=0,ymax=max(h[1]))
                #print(f"{scoreType} mean difference: {100*(m-1):.2f} %")
            else:
                plt.xlabel("Score (absolute)")
                plt.axvline(x=0.0,color="black",label="x=0")
                #print(f"{scoreType} mean difference: {m:.4e}")

            plt.axvline(x=m_neg,color="green",label="mean(non RBP)")
            plt.axvline(x=m_pos,color="red",label="mean(RBP)")

            plt.ylabel("Proteins (%)")
            plt.legend()
            plt.savefig(figureFolder+f"hist_{scoreType}_{regionType}.svg")

    plt.show()

#plot score vs domain type
def plotScoreVSLenth(
        RBPdomains,
        figureFolder,
        figSize = [6.4*2, 4.8*2],
        scoreTypes = ["rel","abs"], #proper values: "abs","rel"
        regionTypes = ["1","2","1or2","1and2","0"] # proper values: ["1","2","1or2","1and2","0"]
        ):
    
    for scoreType in scoreTypes:
        for regionType in regionTypes:
            relevantColumns = RBPdomains[np.invert(pd.isnull(RBPdomains[f'score_{scoreType}_{regionType}']))]

            raw_scores = list(relevantColumns[f'score_{scoreType}_{regionType}'])
            scores=[]
            for score in raw_scores:
                if score != None and np.isnan(score) == False:
                    scores.append(score)
            if(scores == []):
                print(f"No data for scoreType {scoreType}, mode and regionType {regionType} !")
                continue
            lens = [len(seq) for seq in relevantColumns["Protein_seq"]]

            p = np.corrcoef(lens,scores)[0,1]

            plt.figure(figsize=figSize)
            h = plt.scatter(lens, scores) 
            plt.title(f" {scoreType} {regionType} score vs len(seq)\n pearson correlation coeff.: {p:.04f}")

            plt.xlabel("Sequence length")
            plt.ylabel("Score")
            plt.savefig(figureFolder+f"scatter_lenVSscore_{regionType}.svg")

    plt.show()

#plot score vs domain type
# takes 1-4 min depending on the amount of modes
def plotScoreVSMotif(
        RBPdomains,
        figureFolder,
        figSize = [6.4*2, 4.8*2],
        scoreTypes = ["rel"],
        sortByMeasure = "median", #sort motif display by either: "mean", "median", "var", "std"
        specific = False, #use specific domain names not theri families
        minEntryCount = 5, #only draw boxplot if more than this amount of samples are used in that plot
        scatter = False # also draw scatterplot (not only boxplot. ATTENTSION: this migth take super long and generate huge files
    ):

    for scoreType in scoreTypes:
        relevantColumns = RBPdomains[np.invert(pd.isnull(RBPdomains[f'motif_scores']))]
        relevantCells = relevantColumns[f'motif_scores']

        #reformat tuples per protein to dataframe with: motif, socre_rel, score_abs
        scores_dict = {
            "type":[],  
            "name":[], 
            "sName":[],
            "score_rel":[],
            "score_abs": []
        }
        motif_stats = {} #type, mean, median, etc. per motif

        #get score entries
        for cell in relevantCells:
            for entry in cell:
                ty, name, sName, s_rel, s_abs = entry
                if(s_rel == None or s_abs == None):
                    continue
                scores_dict["type"].append(ty)
                scores_dict["name"].append(name)
                scores_dict["sName"].append(sName)
                scores_dict["score_rel"].append(s_rel)
                scores_dict["score_abs"].append(s_abs)
                
                if(specific):
                    if not sName in motif_stats.keys():
                        motif_stats[sName] = {"type":ty}
                else:
                    if not name in motif_stats.keys():
                        motif_stats[name] = {"type":ty}
        scores = pd.DataFrame(scores_dict)

        #get scores per motif
        notEvaluated = []
        #print(f"\t\tIgnoring: name, type, entryCount")
        nameColumn = scores.sName if specific else scores.name
        for name in natsorted(nameColumn.unique()):
            motifValues = np.array( scores[nameColumn == name][f"score_{scoreType}"] )
            entryCount = len(motifValues)
            if entryCount >= minEntryCount:
                motif_stats[name]["mean"]=np.mean(motifValues)
                motif_stats[name]["median"]=np.median(motifValues)
                motif_stats[name]["var"]=np.var(motifValues)
                motif_stats[name]["std"]=np.std(motifValues)
                motif_stats[name]["values"]=motifValues
            else:
                notEvaluated.append( name )
                del motif_stats[name]
        #print(f"\t\tIgnoring (occurance < {minEntryCount}): {notEvaluated}")

        #get sorting order
        sortByValues = np.array([ motif_stats[motif][sortByMeasure] for motif in motif_stats.keys() ]) #get motif measure
        sortOrder = np.argsort(sortByValues)
        keysSorted = np.array([list(motif_stats.keys())[index] for index in sortOrder]) #keys by sorting order
        sortedValues = np.array( [motif_stats[motif]["values"] for motif in keysSorted], dtype=object) #sorted values
        
        #filter for minimal Box count
        motifCount = np.array([ len(valueList) for valueList in sortedValues ])
        print(motifCount)
        keysSorted = keysSorted[motifCount >= minEntryCount]
        sortedValues = sortedValues[motifCount >= minEntryCount]
        
        figSize = ( (figSize[0]/40)*len(keysSorted), figSize[1])
        
        #scatterplot all motifs
        if(scatter):
            plt.figure(figsize=figSize)
            xs, ys = [], []
            for i,motif in enumerate(keysSorted): #draw means sorted by sortByMeasure
                measure = motif_stats[motif][sortByMeasure]
                xs.append(motif)
                ys.append(measure)
                print(f"{motif}\tN={len(sortedValues[i])}\t{sortByMeasure}={measure:.2e}")

            plt.scatter(xs, ys , marker="_", color="red")
            plt.scatter(scores.name,scores[f"score_{scoreType}"],s=5,marker="_") #draw individual values

            plt.title(f" {scoreType} motif type vs score\nOrdered by {sortByMeasure}")
            plt.xlabel("Motif")
            plt.ylabel("Scores")
            plt.xticks(rotation=60)
            plt.savefig(figureFolder+f"scatter_motifVSscore_{scoreType}.jpg")
            #plt.savefig(figureFolder+f"scatter_motifVSscore_.svg") #potentially takes much time   

        #limit y range
        minimum = np.min([ np.min(values) for values in sortedValues])
        maximum = np.max([ np.max(values) for values in sortedValues])
        cap = 4
        if(maximum > cap):
            print(f"Maximum ({maximum}) is bigger than {cap} (capping visualization there)")
            maximum = cap
            
        #also make boxplot
        plt.figure(figsize=figSize)
        p = plt.boxplot(sortedValues,labels=keysSorted, widths=0.75)
        plt.ylim((minimum-(maximum-minimum)*0.10,maximum))
        plt.axhline(1, c='black')
        plt.title(f" {scoreType} motif type vs score\nOrdered by {sortByMeasure}")
        plt.xlabel("Motif")
        plt.ylabel("Scores")
        plt.xticks(rotation=90)


        #add Ns at top and color names
        y_pos = minimum-(maximum-minimum)*0.035
        
        xticklabels = plt.gca().get_xticklabels() # for coloring

        print(f"{'Motif name':<30}\tty\tN\t{sortByMeasure}")
        for i,motif in enumerate(keysSorted):
            #N
            N=len(sortedValues[i])
            plt.text(x=i+1, y=y_pos, s=f"{N}", rotation="vertical", fontsize=8, ha='center', fontweight="bold") #text
                                #rotation="vertical", fontsize=8, color="black",  va='center', )
            #type coloring
            ty = motif_stats[motif]["type"]
            if(ty == 0):
                color="steelblue"
            elif(ty == 1):
                color="green"
            elif(ty == 2):
                color="red"
            xticklabels[i].set_color(color)

            print(f"{motif:<30}\t{ty}\t{N}\t{motif_stats[motif][sortByMeasure]:.4f}")

        plt.tight_layout()
        if(specific):
            plt.savefig(figureFolder+f"boxplot_motifVSscore_{scoreType}_minEntries={minEntryCount}_specific.jpg")
        else:
            plt.savefig(figureFolder+f"boxplot_motifVSscore_{scoreType}_minEntries={minEntryCount}.jpg") 
        #using .svg  #potentially takes much time  
        
        plt.show()


##### 3.3) Binding Domain - inferred #####

#Flank detector that returns tuples of positive intervals with (from_index, to_index_exclusive)
# expects a binary input vector
def getPositiveRegions(vs):
    flanks = []
    lastState = 0
    for i,v in enumerate(vs):
        if v != lastState:
            lastState = v
            flanks.append(i)
    if(lastState == 1): #add last falling flank
        flanks.append(len(vs))
    
    return np.array(flanks).reshape(-1,2) # shapes: (2*N) -> (N, 2)

#inferre domains/motifs from attribution values
def inferMotifs(
        RBPdomains,
        figureFolder,
        filePath_inf,
        thr_dom = 0.7,
        force_generate_inf=False, #force eval regeneration or just load filePath_eval if it exists
        plot=True,
        figSize=[6.4*2, 4.8*2],

        #Curve Smoothing
        window_size=31,
        pol_deg=5,
        ):

    if(force_generate_inf or (os.path.isfile(filePath_inf) == False)):
        #add new columns
        if(not f"inf_mask" in RBPdomains.keys()):
            RBPdomains[f"inf_mask"] = [None]*len(RBPdomains.index) 
        if(not f"inf_motifs" in RBPdomains.keys()):
            RBPdomains[f"inf_motifs"] = [None]*len(RBPdomains.index)
        if(not f"inf_motifs_score" in RBPdomains.keys()): #scores of inferred motifs
            RBPdomains[f"inf_motifs_score"] = [None]*len(RBPdomains.index)

        if plot:
            try:
                os.mkdir(figureFolder+f"inf/")
            except FileExistsError:
                pass

        for i in tqdm(RBPdomains.index):
            RBP_Name = RBPdomains.RBP_Name[i]
            Protein_ID = RBPdomains.Protein_ID[i]
            Protein_seq = RBPdomains.Protein_seq[i]
            IG_attribs = RBPdomains.IG_attribs_mean[i]

            domains = RBPdomains.domains[i]

            
            v_original = IG_attribs

            if plot: #need to create figure here
                plt.figure(figsize=figSize)
                plt.plot(v_original, alpha=0.3, color="grey") #plot non-smooth signal

            ##complete not-domain
            baselineMean = RBPdomains[f"baseline_mean"][i]
            if plot:
                plt.hlines(y=baselineMean,xmin=0,xmax=len(v_original),color="blue")

            #apply filter
            v = scipy.signal.savgol_filter(v_original, window_size, pol_deg) # window size 51, polynomial order 3
            v_min = np.min(v[:-5]) # TODO: why is the last value so low? why do we have to compensate this here?
            v_max = np.max(v)

            #get domains
            v_norm = (v-v_min)/(v_max-v_min) #normalized signal into [0;1]
            inf_mask = v_norm > thr_dom #inferred motif mask
            RBPdomains.at[i,f"inf_mask"] = inf_mask

            motif_regions = getPositiveRegions(inf_mask)
            motifs = [] #actual motif sequences
            motif_scores = []
            
            for fr, to in motif_regions:
                motifs.append(Protein_seq[fr:to]) #get actual motif

                #get motif score
                motifValues = v_original[fr:to] #does not need +1 because we built that region ourselfs (and properly)
                motifMean = sum(motifValues)/len(motifValues)
                motifScore = motifMean/baselineMean
                motif_scores.append(motifScore)

            RBPdomains.at[i,f"inf_motifs"] = motifs
            RBPdomains.at[i,f"inf_motifs_score"] = motif_scores
            
        
            #plot stuff
            if plot:
                #get scores for plotting
                score_rel = RBPdomains[f"score_rel_1or2"][i]
                score_abs = RBPdomains[f"score_abs_1or2"][i]

                #plot basic signal
                plt.plot(v, color="black")
                plt.title(f"{Protein_ID} ({RBP_Name}) \nscore rel="+(f"{score_rel:.04}" if score_rel != None else "None")+f"\nscore abs={score_abs:.2e}")
                #\n(IG_delta = {IG_delta:.04f})
                plt.ylabel("Attribution")

                # plot known motifs
                v_min = np.min(v_original)
                v_max = np.max(v_original)
                for domain in domains:
                    fr, to, ty, name, sName = domain
                    fr = int(fr)
                    to = int(to)

                    if(ty == 0):
                        color="lavender"
                    elif(ty == 1):
                        color="springgreen"
                    elif(ty == 2):
                        color="tomato"

                    motifValues = v_original[fr:to+1]
                    motifMean = sum(motifValues)/len(motifValues)
                    #draw motifs
                    plt.hlines(y=motifMean,xmin=fr,xmax=to,color=color,lw=10,alpha=0.8) #vertical box
                    plt.vlines(x=[fr,to],ymin=v_min, ymax=v_max,color=color, linestyles="dashed")

                    plt.text(x=(fr+to)/2, y=motifMean, s=f"{name}\n({sName})", ha='center', va='center', #text
                                                fontsize=8, color="black", fontweight="bold" )

                #plot new motifs
                for i, (fr, to) in enumerate(motif_regions):
                    motifValues = v_original[fr:to]
                    motifMean = sum(motifValues)/len(motifValues)
                    motif = motifs[i]

                    #draw motifs
                    plt.hlines(y=motifMean,xmin=fr,xmax=to,color="aqua",lw=10,alpha=0.8) #vertical box
                    plt.vlines(x=[fr,to],ymin=v_min, ymax=v_max,color="aqua", linestyles="dashed")

                    plt.text(x=(fr+to)/2, y=motifMean, s=motif, ha='center', va='center', #text
                                    fontsize=1200/len(Protein_seq), color="black", fontweight="bold")


                plt.savefig(
                        figureFolder+f"inf/"+f"rel="+(f"{score_rel:.4f}" if score_rel != None else "None")+f"_abs={score_abs:.2e}_{Protein_ID}_{RBP_Name}.svg")
                #plt.show()
                plt.clf()
                plt.close()
                    
        #save
        print("Saving Inference results")
        RBPdomains.to_pickle(filePath_inf)
    else:
        print("Loading Inference results")
        RBPdomains = pd.read_pickle(filePath_inf)

    return RBPdomains


##### 3.4) K-mers - inferred #####

def getKmers(
        RBPdomains, #inferred motifs (high attribution regions)
        figureFolder,
        filePath_kmers,
        k=3, #must be odd!
        thr_kmer=0.7,
        force_generate_kmers=False, #force kmers regeneration or just load filePath_kmers if it exists
        plot=True,
        figSize=[6.4*2, 4.8*2],
        fromMotifs=False,

        #Curve Smoothing
        window_size=31,
        pol_deg=5,
        ):


    if(force_generate_kmers or (os.path.isfile(filePath_kmers) == False)):

        #create figure folder
        if plot:
            try:
                os.mkdir(figureFolder+f"kmers/")
            except FileExistsError:
                pass
                
        #create output structures
        kmers_dict = {} #key = unique kmer, values: list of scores -> len of list = kmer count
            
        k_off = int(k/2)

        for i in tqdm(RBPdomains.index):
            RBP_Name = RBPdomains.RBP_Name[i]
            Protein_ID = RBPdomains.Protein_ID[i]
            Protein_seq = RBPdomains.Protein_seq[i]
            IG_attribs = RBPdomains.IG_attribs_mean[i]

            #IG_delta = RBPdomains.IG_delta[i]
            domains = RBPdomains.domains[i]

            fs = 1200/len(Protein_seq) #fontsize for kmers
            v_original = IG_attribs
            

            if plot: #need to create figure here
                plt.figure(figsize=figSize)
                plt.plot(v_original, alpha=0.3, color="grey") #plot non-smooth signal

            ##complete not-domain
            baselineMean = RBPdomains[f"baseline_mean"][i]
            if plot:
                plt.hlines(y=baselineMean,xmin=0,xmax=len(v_original),color="blue")

            #apply filter
            v = scipy.signal.savgol_filter(v_original, window_size, pol_deg) # window size 51, polynomial order 3
            v_min = np.min(v[:-5]) # TODO: why is the last value so low? why do we have to compensate this here?
            v_max = np.max(v)
            
            #plot stuff
            if plot:
                #get scores for plotting
                score_rel = RBPdomains[f"score_rel_1or2"][i]
                score_abs = RBPdomains[f"score_abs_1or2"][i]

                #plot basic signal
                plt.plot(v, color="black")
                plt.title(f"{Protein_ID} ({RBP_Name}) \nscore rel="+(f"{score_rel:.04}" if score_rel != None else "None")+f"\nscore abs={score_abs:.2e}")
                #\n(IG_delta = {IG_delta:.04f})
                plt.ylabel("Attribution")

                # plot known motifs
                for domain in domains:
                    fr, to, ty, name, sName = domain
                    fr = int(fr)
                    to = int(to)

                    motifValues = v_original[fr:to+1]
                    motifMean = sum(motifValues)/len(motifValues)
                    #draw motifs
                    plt.hlines(y=motifMean,xmin=fr,xmax=to,color="springgreen",lw=10,alpha=0.8) #vertical box
                    plt.vlines(x=[fr,to],ymin=v_min, ymax=v_max,color="springgreen", linestyles="dashed")

                    plt.text(x=(fr+to)/2, y=motifMean, s=name, ha='center', va='center', #text
                                                    fontsize=8, color="black", fontweight="bold" )

            ### get kmers ###
            if(fromMotifs): #get all kmers out of inferred motifs (using score of motif for kmer)
                pMotifs_inf = RBPdomains.inf_motifs_mean[i]
                pMotifs_inf_scores = RBPdomains.inf_motifs_mean[i]

                #get all k-mers and respective scores
                for motif, score in tqdm(zip(pMotifs_inf, pMotifs_inf_scores)):
                    if len(motif) < k: # if the motif is smaller than the kmer, we cannot build the kmer
                        continue

                    for i in range(len(motif)-(k-1)): #extract kmers
                        kmer_name = motif[i:i+k]

                        #add to kmers
                        if(not kmer_name in kmers_dict.keys()):
                            kmers_dict[kmer_name] = [score]
                        else:
                            kmers_dict[kmer_name].append(score)

            else:
                thr = v_min+(v_max-v_min)*thr_kmer #concerts percent of range to value
                for i, v in enumerate(v_original):
                    if v >= thr and i > 0 and i < len(v_original)+k_off: #create a kmer around this value
                        fr, to = i-k_off, i+k_off
                        kmer_name = Protein_seq[fr: to+1]
                        kmer_mean = np.mean(v_original[fr: to+1])
                        kmer_score = np.mean(kmer_mean/baselineMean)
                            
                        #add to unique kmers
                        if(not kmer_name in kmers_dict.keys()):
                            kmers_dict[kmer_name] = [kmer_score]
                        else:
                            kmers_dict[kmer_name].append(kmer_score)
                
                        #plot kmer
                        if plot:
                            plt.hlines(y=kmer_mean,xmin=fr,xmax=to,color="aqua",lw=10,alpha=0.8) #vertical box
                            #plt.vlines(x=[fr,to],ymin=v_min, ymax=v_max,color="aqua", linestyles="dashed")

                            plt.text(x=(fr+to)/2, y=kmer_mean, s=kmer_name, ha='center', va='center', #text
                                            fontsize=fs, color="black", fontweight="bold")

                if plot:
                    plt.savefig(figureFolder+f"kmers/"+f"rel="+(f"{score_rel:.4f}" if score_rel != None else "None")+f"_abs={score_abs:.2e}_{Protein_ID}_{RBP_Name}.svg")
                    #plt.show()
                    plt.clf()
                    plt.close()

            #save
        print("Saving Kmers results")
        with open(filePath_kmers, 'w') as f:
            json.dump(kmers_dict, f)

    else:
        print("Loading Kmers results")
        with open(filePath_kmers, 'r') as f:
            kmers_dict = json.load(f)
                    
    return kmers_dict

#what k-mers are most common (occurance)?
def plotKmersByOccurance(kmers_dict, sessionFolder, figureFolder, k=3, plotTopN=200, printTopN = 20, figureSuffix="kmer_frequency_occurance_ranked"):
    kmer_occurance = [len(scores) for scores in kmers_dict.values()]
    sortIndices = np.argsort(kmer_occurance)[::-1]

    x,y = [], [] # keys, occurance
    all_keys = list(kmers_dict.keys())

    #write complete list (not only top) to file
    path = sessionFolder+"topKmers_occurance.txt"
    with open(path, "a+") as f:
        for i, index in enumerate(sortIndices):
            key = all_keys[index]
            value = kmer_occurance[index]
            #print(f"{i}: {key} -> {value}")
            f.write(f"{key}\t{value}\n")
            x.append(key)
            y.append(value)

    #plot and print
    plt.figure()
    plt.plot(y[:plotTopN])
    plt.title(f"{k}-mer ranking by occurance")
    #plt.yscale("log")
    plt.ylabel("Occurance")
    plt.xlabel("Rank")
    plt.savefig(figureFolder+figureSuffix+f"_topN={plotTopN}.svg")
    plt.show()

    for i in range(printTopN):
        print(f"{i}: {x[i]} -> {y[i]}")
        
    return x

#what k-mers have highest mean score?
def plotKmersByScore(kmers_dict, sessionFolder, figureFolder, k=3, minOccurance=10, plotTopN=200, printTopN = 20, figureSuffix="kmer_frequency_score_ranked"):
    kmer_score = [np.mean(scores) for scores in kmers_dict.values()]
    kmer_occurance = [len(scores) for scores in kmers_dict.values()]
    sortIndices = np.argsort(kmer_score)[::-1]

    x,y,o = [], [], [] # keys, occurance
    all_keys = list(kmers_dict.keys())

    #write complete list (not only top) to file
    path = sessionFolder+"topKmers_mean_score.txt"
    with open(path, "a+") as f:
        for i, index in enumerate(sortIndices):
            key = all_keys[index]
            mean_score = kmer_score[index]
            occurance = kmer_occurance[index]
            #print(f"{i}: {key} -> {value}")
            f.write(f"{key} ({occurance}x)\t{mean_score}\n")

            if(occurance >= minOccurance): #we only want to report kmers with a minimal occurance
                x.append(key)
                y.append(mean_score)
                o.append(occurance)

    #plot and print
    plt.figure()
    plt.plot(y[:plotTopN])
    #plt.yscale("log")
    plt.title(f"{k}-mer ranking by score (occ >= {minOccurance})")
    plt.ylabel(f"Mean  {k}-mer score")
    plt.xlabel("Rank")
    plt.savefig(figureFolder+figureSuffix+f"_topN={plotTopN}.svg")
    plt.show()

    for i in range(printTopN):
        print(f"{i}: {x[i]} ({o[i]}x) -> {y[i]}")
        
    return x

#kmer scores vs therir occurance. Colored by bressin importance
def plotKmersScoreVSoccurance(
        kmers_dict,
        bressinSorted,
        figureFolder,
        minOccurance = 15,
        clipWeights=0.5,
        s = 30,
        k=3,
        figureSuffix = "kmers_scoreVSoccurance_bressinColored"
    ):

    #plot bressin availables
    occs_b, scores_b, c_b  = [],[], [], #occurance, score, color value
    occs, scores = [],[] #occurance, score

    for key in kmers_dict.keys():
        values = kmers_dict[key]
        mean = np.mean(values)
        occurance = len(values)

        if(occurance < minOccurance):
            continue

        if(key in bressinSorted.keys()):
            bressinWeight = bressinSorted[key]

            if(bressinWeight>clipWeights):
                bressinWeight = clipWeights
            if(bressinWeight<-clipWeights):
                bressinWeight = -clipWeights

            scores_b.append(mean)
            occs_b.append(occurance)
            c_b.append(bressinWeight)
        else:
            scores.append(mean)
            occs.append(occurance)

    #plot and print
    plt.figure(figsize=[12,6])
    plt.title(f"{k}-mer occurance VS scores (inferred kmers) (occ >= {minOccurance})")
    plt.ylabel(f"Mean kmer score")
    plt.xlabel(f"Kmer occurance")

    plt.scatter(occs_b,scores_b,c=c_b,
                cmap="RdYlGn", #"bwr" #viridis
                s=s
               )
    plt.colorbar()

    plt.scatter(occs,scores,c="grey",s=s)

    #plt.yscale("log")
    plt.savefig(figureFolder+figureSuffix+f"_minOcc={minOccurance}.svg")
    plt.show()

#comparison aginast Bressin top N (on human)
def compairKmersBressin(
        oursSorted, #list of (contribution) sorted kmers
        bressinSorted, #path to bressin top occurances file (requires deletion of comments in file to be parse-able!)
        figureFolder,
        percent=True,
        k=3,
        topN_start=50, #if None: from 1
        topN_end=200, #if None: all available values
        oursBias = 0, #use this amount more from our set vs theirs
        figureSuffix="kmer_ours_vs_bressin"
        ):
    global AA_alphabet

    total_ours = len(oursSorted)
    total_theirs = len(bressinSorted)
          
    overlap_theirs = [] # (topN, value)
    overlap_ours = [] # (topN, value)
    overlap_baseline = [] # (topN, value) for random sets

    if(topN_start==None):
        topN_start=1

    if(topN_end==None):
        topN_end=len(oursSorted)

    bressinSortedKmers = list(bressinSorted.keys())

    for topN in tqdm(range(topN_start,topN_end)):
        #generate sets for intersections
        theirs = set(bressinSortedKmers[:topN])
        ours = set(oursSorted[:topN+oursBias])
        #create intersection
        overlap = ours&theirs

        if(percent):#percentages
            v_theirs = len(overlap)/len(theirs)
            v_ours = len(overlap)/len(ours)
        else:
            v_theirs = len(overlap)
            v_ours = len(overlap)
        overlap_theirs.append( (topN, v_theirs) )
        overlap_ours.append( (topN, v_ours) )

        #baseline (two random sets)
        expected_baseline = ((topN/total_theirs)*(topN/total_ours))*total_ours
        if(percent):
            v_baseline = expected_baseline/len(ours)
        else:
            v_baseline = expected_baseline
        overlap_baseline.append( (topN, v_baseline) )

    #make a nice figure
    #xlim = (topN_start,topN_end)
    plt.figure()
    plt.title("Top set overlap: Bressin kmers vs our kmers")
    x = np.array(overlap_theirs).transpose()[0]
    y = np.array(overlap_theirs).transpose()[1]
    plt.plot(x,y*100 if percent else y,label="theirs")
    x = np.array(overlap_ours).transpose()[0]
    y = np.array(overlap_ours).transpose()[1]
    plt.plot(x,y*100 if percent else y,label="ours")
    x = np.array(overlap_baseline).transpose()[0]
    y = np.array(overlap_baseline).transpose()[1]
    plt.plot(x,y*100 if percent else y,label="baseline")

    #plt.xlim(xlim)
    plt.ylim((0,None))
    if(oursBias>0):
        plt.xlabel(f"top N ranked {k}mers of Bressin vs ours (bias={oursBias})")
    else:
        plt.xlabel(f"top N ranked {k}mers of Bressin vs ours")

    plt.ylabel("% of set" if percent else "Overlapping elements")
    plt.legend()
    plt.savefig(figureFolder+figureSuffix+f"_fromTo=({topN_start},{topN_end})_p={percent}.svg")
    plt.show()
        
    #print most interesting points
    m = int((topN_start+topN_end)/2)
    if(percent):
        print(f"Overlap at their N={topN_start}: {overlap_theirs[0][1]*100}%")
        print(f"Overlap at their N={m}: {overlap_theirs[m-topN_start][1]*100}%")
        print(f"Overlap at their N={topN_end}: {overlap_theirs[-1][1]*100}%")
    else:
        print(f"Overlap at their N={topN_start}: {overlap_theirs[0][1]}")
        print(f"Overlap at their N={m}: {overlap_theirs[m-topN_start][1]}")
        print(f"Overlap at their N={topN_end}: {overlap_theirs[-1][1]}")

    alphabetSize = len(AA_alphabet)
    print(f"All found 3mers N = {len(oursSorted)} of {alphabetSize**k}") #20*20*20 = 8000

    return (overlap_ours, overlap_theirs)

##### 3.5) Residue analysis - inferred #####
AA_alphabet = ['A',  'C','D','E','F','G','H','I',  'K','L','M','N',  'P','Q','R','S','T',  'V','W',  'Y']


#what residues are most/least common in TOTAL?
def residueOccuranceTotal(RBPdomains, figureFolder, figureSuffix="residue_frequency_total"):
    global AA_alphabet
    
    #compute residue occurance
    AA_count_total = dict.fromkeys(AA_alphabet, 0)
    for i in tqdm(RBPdomains.loc[pd.isnull(RBPdomains.inf_motifs_mean) == False].index):
        seq = RBPdomains.Protein_seq[i]
        for letter in AA_alphabet:
            AA_count_total[letter] += seq.count(letter)
        
    #compute percentage of occurance
    countAll = sum(AA_count_total.values())
    AA_p_total = dict.fromkeys(AA_alphabet, 0)
    for AA in AA_count_total.keys():
        AA_count = AA_count_total[AA]
        p = AA_count/countAll
        AA_p_total[AA] = p

    #print/plot most common ones
    sortIndices = np.argsort(list(AA_p_total.values()))[::-1]
    keys = list(AA_p_total.keys())

    x,y,o = [],[],[] #AA, percentage (occurance), occurance
    for i in sortIndices:
        key = keys[i]
        x.append(key)
        y.append(AA_p_total[key]*100 )
        occ = AA_count_total[key]
        o.append(occ)

    plt.figure()
    plt.title("Total residue frequency")
    plt.scatter(x,y)
    plt.ylabel("%")
    plt.xlabel("Residue")

    #mark residues that are most common in RBPs (according to Bressin 19):
    # "K), arginine (R) and glycine (G) are found to have the largest SVM weight"
    plt.gca().get_xticklabels()[x.index("K")].set_color('green')
    plt.gca().get_xticklabels()[x.index("R")].set_color('green')
    plt.gca().get_xticklabels()[x.index("G")].set_color('green')
    # "E and L are the residues most absent from RNAbinding sites in human cells (11)"
    plt.gca().get_xticklabels()[x.index("E")].set_color('red')
    plt.gca().get_xticklabels()[x.index("L")].set_color('red')

    plt.savefig(figureFolder+figureSuffix+f".svg")
    plt.show()
    
    for key, p, occ in zip(x, y, o):
        print(f"{key}: {occ}x\tp={p:.2f}%")
    
    return AA_count_total, AA_p_total

#what residues are most/least common in our inferred kmers (potentially relative to something)?
def residueOccuranceInKmers(kmers_dict,k, figureFolder,
                            reltiveToCount=None, #is that is a dict of residue occurances, the occurance will be computed respect to them
                            figureSuffix="residue_frequency_kmers"):
    global AA_alphabet
    
    #count residue occurnace in kmers
    AA_count_kmers = dict.fromkeys(AA_alphabet, 0)
    for kmer in tqdm(kmers_dict.keys()):
        occurance = len(kmers_dict[kmer])
        for letter in kmer:
            AA_count_kmers[letter] += occurance

    #compute (relative) percentage of occurance
    AA_p_kmers = dict.fromkeys(AA_alphabet, 0)
    countAll = sum(AA_count_kmers.values())
    
    if reltiveToCount != None:
        for AA in AA_count_kmers.keys():
            AA_count = AA_count_kmers[AA]
            AA_expected = countAll*reltiveToCount[AA]
            delta_p = 1-(AA_count/AA_expected)
            AA_p_kmers[AA] = delta_p
    else:
        for AA in AA_count_kmers.keys():
            AA_count = AA_count_kmers[AA]
            p = AA_count/countAll
            AA_p_kmers[AA] = p

    #print/plot most common ones      
    sortIndices = np.argsort(list(AA_p_kmers.values()))[::-1]
    keys = list(AA_p_kmers.keys())

    x, y, o = [], [], [] # AA, (relative) occurance (percentage)
    for i in sortIndices:
        key = keys[i]
        p = AA_p_kmers[key]*100
        x.append(key)
        y.append( p )
        occ = AA_count_kmers[key]
        o.append(occ)

    plt.figure()
    plt.scatter(x,y)
    if(reltiveToCount == None):
        plt.title(f"In-{k}-mer residue frequency")
        plt.ylabel("%")
    else:
        plt.title(f"In-{k}-mer residue frequency (relative)")
        plt.ylabel("% (relative)")
    plt.xlabel("Residue")

    #mark residues that are most common:
    # "K), arginine (R) and glycine (G) are found to have the largest SVM weight"
    plt.gca().get_xticklabels()[x.index("K")].set_color('green')
    plt.gca().get_xticklabels()[x.index("R")].set_color('green')
    plt.gca().get_xticklabels()[x.index("G")].set_color('green')
    # "E and L are the residues most absent from RNAbinding sites in human cells (11)"
    plt.gca().get_xticklabels()[x.index("E")].set_color('red')
    plt.gca().get_xticklabels()[x.index("L")].set_color('red')
    plt.savefig(figureFolder+figureSuffix+f".svg")

    plt.show()

    for key, p, occ in zip(x, y, o):
        if(reltiveToCount == None):
            print(f"{key}: {occ}x\tp={p:.2f}%")
        else:
            print(f"{key}: {occ}x\tdelta_p={p:.2f}%")

    return AA_count_kmers, AA_p_kmers




