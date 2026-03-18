


import pandas as pd
from tqdm import tqdm
from contextlib import nullcontext #for empty context

# Initialize global environment and import useful utility functions 
import sys
from pathlib import Path
sys.path.append(str(Path(".").absolute()))
from scripts.initialize import *
initialize(__file__)

from scripts.embeddings.utils import getModel, _prepare_prott5_sequence
torch.hub.set_dir(TORCH_MODEL_CACHE)


#Parse arguments
import argparse
import time

parser = argparse.ArgumentParser(
                    prog='generateEmbeddings.py',
                    description='Generate embeddings given a dataset and a language model')

parser.add_argument('-D', '--device', dest='device', action='store',
                    help='LM device, either "cuda" for GPU & VRAM or "cpu" for CPU & RAM',
                    default="cpu")
parser.add_argument('-M', '--languageModel', dest='LM_name', action='store',
                    help='Language Model. Options: esm1b_t33_650M_UR50S, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D, esm2_t48_15B_UR50D, protT5_xl_uniref50',
                    default="esm1b_t33_650M_UR50S")
parser.add_argument('-l', '--maxSeqLen', dest='maxSeqLen', action='store',
                    help='Crop sequences above that length to that length',
                    type=int, default=None)
parser.add_argument('-p', '--precision', dest='precision', action='store',
                    help='Model precision. Either f32, f16 or auto (for automatic mixed precision)',
                    default="auto")
parser.add_argument('-f', '--fsdp', dest='fsdp', action=argparse.BooleanOptionalAction,
                    help='Use Fully Sharded Data Parallel. Use this flag if your model is to big for your GPU and you want to offload also tp CPU',
                    type=bool, default=False)

args = parser.parse_args() #parse arguments

#ESM1-b requires max seq length of 1024
# write commandline arguments to params dict
params = {}
for key, value in args._get_kwargs():
    params[key] = value

# Print/Log paramaters
log("Parameters:")
for key in params.keys():
    log(f"\t{key}: {params[key]}")


# list of datasets (raw) we want to encode
dataSetsRaw = [
    DATA_RAW.joinpath("bressin19.tsv"),
    DATA_RAW.joinpath("InterPro.tsv"),
    DATA_RAW.joinpath("RIC.tsv"),
        # Example issues:
        #[P38898]: sequence is None or empty ('nan')
		#[Q16385]: Gene_ID relates to more than one row
    ]


is_prot_t5 = LM_name.startswith("protT5")
tokenizer = None
alphabet = None

if fsdp == False: #default one-GPU run
    LM, helper, repr_layer, input_emb_dim = getModel(LM_name=LM_name, device=device)
    if is_prot_t5:
        tokenizer = helper
    else:
        alphabet = helper
else:
    if is_prot_t5:
        raise RuntimeError("ProtT5 models are not supported with FSDP.")
    from datetime import timedelta
    # Distributed Data Parallel
    url = "tcp://localhost:23456"
    #torch.distributed.init_process_group(backend="nccl", init_method=url, rank=0, world_size=1)
    storePath = Path("/p/project/hai_ml4rg_rbp/Project_ml4rg/DDP_store")
    #storePath = Path(os.getenv('FASTDATA',"/p/home/jusers/steinbauer1/juwels/"),"DDP_store")
    store = torch.distributed.FileStore(file_name=str(storePath), world_size=1)
    time.sleep(10)
    print("init_process_group", flush=True)
    started = False
    while started == False:
        try:
            torch.distributed.init_process_group(backend="nccl", store=store, rank=0, world_size=1, timeout=timedelta(seconds=5))
            #the timeout is important because otherwise the process group will wait 30min for another process
            started = True
        except RuntimeError as e:
            print(f"init_process_group: runtime error: {e}", flush=True)
            #print(f"\t -> trying again in 5 sec", flush=True)
            #time.sleep(5)
            started=True #if you try again you will only get "runtime error: trying to initialize the default process group twice!""

   
    #torch.distributed.init_process_group(backend="nccl", init_method="tcp://localhost:23456", rank=0, world_size=0)
    print("init_process_group done", flush=True)
    #torch.distributed.init_process_group(backend="mpi", rank=0, world_size=1)

    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.distributed.fsdp.wrap import enable_wrap, wrap

    # initialize the model with FSDP wrapper
    fsdp_params = dict(
       mixed_precision=True,
       device_id ="cuda",
        #flatten_parameters=True, # unexpected keyword argument 'flatten_parameters'
        #state_dict_device=torch.device("cpu"),  # reduce GPU mem usage #unexpected keyword argument 'state_dict_device'
        #cpu_offload=True,  # enable cpu offloading
    )
    print("Wrapping", flush=True)
    with enable_wrap(wrapper_cls=FullyShardedDataParallel, device_id ="cuda:0" ):# **fsdp_params):

        model, alphabet, repr_layer, input_emb_dim = getModel(LM_name=LM_name)
        #model, vocab = esm.pretrained.load_model_and_alphabet_core(
        #    LM_name, model_data, regression_data
        #)
        batch_converter = alphabet.get_batch_converter()
        model.eval()

        # Wrap each layer in FSDP separately
        for name, child in model.named_children():
            if name == "layers":
                for layer_name, layer in child.named_children():
                    #print(f"\twrapping{layer_name}", flush=True)
                    wrapped_layer = wrap(layer)
                    setattr(child, layer_name, wrapped_layer)
        fsdp_model = wrap(model)
    LM = fsdp_model
    tokenizer = None

#Set truncation distance
log(f"\tsequenceTruncationLength: {input_emb_dim} (model default)")

#Set precision
context = nullcontext()
match precision:
    case "f32":
        pass #that is default
    case "f16":
        LM.to(torch.float16)
    case "auto":
        env = torch.amp.autocast(device_type=device, dtype=torch.float32)
    case _:#default/falltrough
        raise RuntimeError(f"Unknown precision \"{precision}\" (valid: f32, f16, auto)")

#generate embeddings for each language model
batch_converter = None
if not is_prot_t5:
    batch_converter = alphabet.get_batch_converter()

if False: # GPU / CPU offloading. taken from: https://github.com/facebookresearch/esm/blob/main/examples/esm2_infer_fairscale_fsdp_cpu_offloading.py

    #Imports
    log("FSDP branch")
    import esm
    from torch.distributed.fsdp import FullyShardedDataParallel, CPUOffload
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy, wrap, enable_wrap
    from torch.nn.parallel import DistributedDataParallel

    # Load model
    log("load model")
    model_name = LM_name
    model_data, regression_data = esm.pretrained._download_model_and_regression_data(model_name)
    model, alphabet = esm.pretrained.load_model_and_alphabet_core(
            model_name, model_data, regression_data
        )    
    batch_converter = alphabet.get_batch_converter()

    # Distributed Data Parallel

    log("setup DDP")
    url = "tcp://localhost:23456"
    #torch.distributed.init_process_group(backend="nccl", init_method=url, rank=0, world_size=1)
    storePath = Path(os.getenv('FASTDATA',"/p/home/jusers/steinbauer1/juwels/"),"DDP_store2")
    store = torch.distributed.FileStore(file_name=str(storePath), world_size=1)
    torch.distributed.init_process_group(backend="nccl", store=store, rank=0, world_size=1)
    #torch.distributed.init_process_group(backend="mpi", rank=0, world_size=1)
    
    # initialize the model with FSDP wrapper
    log("init FSDP model")
    fsdp_params = dict(
        mixed_precision=True,
        flatten_parameters=True,
        state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
        cpu_offload=True,  # enable cpu offloading
    )
    with enable_wrap(wrapper_cls=FullyShardedDataParallel, **fsdp_params):
        model, vocab = esm.pretrained.load_model_and_alphabet_core(
            LM_name, model_data, regression_data
        )
        batch_converter = vocab.get_batch_converter()
        model.eval()

        # Wrap each layer in FSDP separately
        for name, child in model.named_children():
            if name == "layers":
                for layer_name, layer in child.named_children():
                    wrapped_layer = wrap(layer)
                    setattr(child, layer_name, wrapped_layer)
        fsdp_model = wrap(model)

    if False: #does not work?
        model = DistributedDataParallel(model)
        
        # Fully Sharded Data Parallel
        fsdp_model = FullyShardedDataParallel(
            model(),
            mixed_precision=True,
            flatten_parameters=True,
            state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
            fsdp_auto_wrap_policy=transformer_auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=True), # enable cpu offloading
        )

    LM = fsdp_model


for dataSetRaw_Path in dataSetsRaw:
    datsetName = dataSetRaw_Path.stem
    log(f"\tEncoding {datsetName}")

    outputFolder = EMBEDDINGS.joinpath(LM_name).joinpath(datsetName)
    outputFolder.mkdir(exist_ok=True, parents=True) # create

    #get existing files
    existing = set(os.listdir(outputFolder))

    #get required files
    df = pd.read_csv(dataSetRaw_Path, sep="\t")
    Gene_IDs = df.Gene_ID
    required = set(Gene_IDs)

    #get missing file names
    missingNames = required - existing  # todo = required - existing
    log(f"\t\t{len(existing)} of {len(required)} ({ (len(existing)/len(required))*100 :.06f} %) files exist in {outputFolder}")

    #generate embeddings
    for Gene_ID in tqdm(missingNames):
        locator = df.Gene_ID==Gene_ID

        if(sum(locator) > 1):
            log(f"\t\t[{Gene_ID}]: Gene_ID relates to more than one row. rows index={df.loc[locator]['Gene_ID'].index.to_list()}")
            continue
        elif(sum(locator) == 0):
            log(f"\t\t[{Gene_ID}]: Gene_ID has no row associated")
            continue

        seq = df.loc[locator]["sequence"].item()
        if (seq == "" or seq == None or pd.isnull(seq)):
            log(f"\t\t[{Gene_ID}]: sequence is None or empty ('{seq}')")
            continue
        elif(type(seq)==float):
            log(f"\t\t[{Gene_ID}]: sequence type is float. Value: '{seq}'")

        extra_tokens = 0 if is_prot_t5 else 2

        #Skip if sequence too long for user-defined max length
        if(maxSeqLen != None and len(seq)+extra_tokens > maxSeqLen):
            log(f"\t\t[{Gene_ID}]: len(sequence) = {len(seq)} > {maxSeqLen} -> skipping")
            continue

        #Truncate sequence if too long for the LM
        if is_prot_t5:
            if len(seq) > input_emb_dim:
                seq = seq[:input_emb_dim]
        else:
            if( len(seq)+2 > input_emb_dim): #+2 because of start/end tokens
                seq = seq[:input_emb_dim-2]

        # create input data
        filePath = outputFolder.joinpath(Gene_ID)
        if is_prot_t5:
            prepared = _prepare_prott5_sequence(seq)
            if prepared == "":
                log(f"\t\t[{Gene_ID}]: sequence empty after preprocessing -> skipping")
                continue
            tokenizer_max_len = input_emb_dim
            if maxSeqLen is not None:
                tokenizer_max_len = min(tokenizer_max_len, maxSeqLen)
            inputs = tokenizer(
                [prepared],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=tokenizer_max_len,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with context, torch.no_grad():
                outputs = LM(**inputs)
            emb = outputs.last_hidden_state[0]
        else:
            batchData = [( Gene_ID, seq ) ]
            #execute
            with context, torch.no_grad():
                batch_labels, batch_strs, batch_tokens = batch_converter( batchData )
                #batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

                results = LM(batch_tokens.to(device), repr_layers=[repr_layer], return_contacts=True)
            embs = results["representations"][repr_layer]
            emb = embs[0]
            #log(emb.shape)
        
        #saving output
        torch.save(emb, filePath)

log("done.")
#log(f"Entries total: {len(RBPs)}")
#log(f"\tExisiting embeddings: {len(exisitng_indices)}")
#log(f"\tMissing sequence: {len(errorous_entries)}")
#log(f"\tToo big for processing: {len(too_big_indices)}")
#log(f"\tTodo: {len(todo)}")
