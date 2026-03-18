from T3funs import *


from torch.profiler import profile, record_function, ProfilerActivity

dataFolder = "data/"

#Parameter
params = {
    #Model paramaters
    "hidden_size":320,
    "embedding_dim":1280,
    #Datset Parameters
    "dataFrame": pd.read_pickle(dataFolder+"T1_data.pkl"),
    "encodingFolder": dataFolder+"emb_esm1b_trunc/", #keys: ['label', 'layer', 'representation', 'mean_representation']
    #Dataloader paramater
    "prefetch_factor": 2, #2
    "num_workers": 3, #3
    "pin_memory": True,
    "persistent_workers": True,
    "pin_memory_device":"",
    #Training Paramaters
    "batchsize": 320, #for bratchsize 320: prf 2, nw 3
    "epochs": 2,
    "lr": 0.00005,
    "device": torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
}


t0 = time.time()
#with profile(
#    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
#    profile_memory=True,
#    record_shapes=True,
#    with_stack=True) as prof:
#    with record_function("model_inference"):
model, stats = trainModel( params )
print("training took "+str(time.time()-t0))

#print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
#prof.export_chrome_trace("trace.json")

#plotTrainVal(stats)
#plotMCC_BACC_AUPR(stats)
#plotPRC(stats)
#plotROC(stats)

#modelPath = exportModel(dataFolder, model, stats, params, name="naive_prof")