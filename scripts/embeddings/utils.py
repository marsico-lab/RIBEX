

# Pretrained model data from https://github.com/facebookresearch/esm#pre-trained-models-

import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *
initialize(__file__)

from esm.pretrained import esm1b_t33_650M_UR50S, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D, esm2_t48_15B_UR50D #the ESM2 model
import torch
torch.hub.set_dir(TORCH_MODEL_CACHE)


# structure= LM_name : ( #layers, #million_params, dataset, embedding_dim)
pretrainedModelInfos = {
    #ESM-2
        "esm2_t48_15B_UR50D":   (48,    15000,  "UR50/D 2021_04", 5120),
        "esm2_t36_3B_UR50D":    (36,    3000,   "UR50/D 2021_04", 2560),
        "esm2_t33_650M_UR50D":  (33,    650,    "UR50/D 2021_04", 1280),
        "esm2_t30_150M_UR50D":  (30,    150,    "UR50/D 2021_04", 640),
        "esm2_t12_35M_UR50D":   (12,    35,     "UR50/D 2021_04", 480),
        "esm2_t6_8M_UR50D":     (6,     8,      "UR50/D 2021_04", 320),
    #ProtT5
        "protT5_xl_uniref50":   (24,    3000,   "UniRef50",        1024),
    #ESM-1b
        "esm1b_t33_650M_UR50S": (33,    650,    "UR50/S 2018_03", 1280),
    #ESM-1
        "esm1_t34_670M_UR50S":  (34,    670,    "UR50/S 2018_03", 1280),
        "esm1_t34_670M_UR50D":  (34,    670,    "UR50/D 2018_03", 1280),
        "esm1_t34_670M_UR100":  (34,    670,    "UR100 2018_03", 1280),
        "esm1_t12_85M_UR50S":   (12,    85,     "UR50/S 2018_03", 768),
        "esm1_t6_43M_UR50S":    (6,     43,     "UR50/S 2018_03", 768),
}


def getModel(LM_name, device):
    #Set model
    match LM_name:
        case "esm1b_t33_650M_UR50S": #ESM1-b t33 0.65B
            LM, alphabet = esm1b_t33_650M_UR50S()
            repr_layer = 33
            input_emb_dim = 1024 # output is 1280
        case "esm2_t33_650M_UR50D": #ESM2 t33 0.65B
            LM, alphabet = esm2_t33_650M_UR50D()
            repr_layer = 33
            input_emb_dim = 200000#1280 #TODO: what is te actual size?
        case "esm2_t36_3B_UR50D":  #ESM2 t36 3B
            LM, alphabet = esm2_t36_3B_UR50D()
            repr_layer = 36
            input_emb_dim = 200000#2560 #TODO: what is te actual size?
        case "esm2_t48_15B_UR50D":  #ESM2 t48 15B
            LM, alphabet = esm2_t48_15B_UR50D()
            repr_layer = 48
            input_emb_dim = 200000#5120 #TODO: what is te actual size?
        case _: #default/fallthrough
            raise RuntimeError(f"Unknown language model \"{LM_name}\"")
    return LM.to(device), alphabet, repr_layer, input_emb_dim

    # NOTE: If the above lines fail due to some downloading error, you migth want to download the models yourself to the
    # hub cache location (TORCH_MODEL_CACHE/checkpoints/)
    # wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1b_t33_650M_UR50S.pt
    # wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm1b_t33_650M_UR50S-contact-regression.pt
    # wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt
    # wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t33_650M_UR50D-contact-regression.pt
    # wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt
    # wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t36_3B_UR50D-contact-regression.pt
    # wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t48_15B_UR50D.pt
    # wget https://dl.fbaipublicfiles.com/fair-esm/regression/esm2_t48_15B_UR50D-contact-regression.pt
