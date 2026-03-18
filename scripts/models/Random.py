import torch
import torch.nn as nn
import pytorch_lightning as pl

from scripts.training.analyze_utils import getMetrics #to compute all our metrics

class RandomClassifier(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params

        self.labelPool = [] # pool of gt labels
        self.rng = torch.Generator() # random number generator
        self.rng.manual_seed(params["seed"]) # set seed
        #self.rng = np.random.default_rng(params["seed"]) # random number generator


    def loss(self, input, target):
        """Implements binary cross entropy loss with *class* weights.
        In contrast; pytroch BCELoss only has elementwise weights.
        Reduction is mean.
        """
        weights = self.params["crit_weight"] # (weight_neg_class, weight_pos_class)
        
        input = torch.clamp(input,min=1e-7,max=1-1e-7)
        bce = - weights[1] * target * torch.log(input) - (1 - target) * weights[0] * torch.log(1 - input)
        return torch.mean(bce)

    def forward(self, x):
        # Get random gt label

        # choose random label from pool
        if(len(self.labelPool) == 0): #pool has no elements yet
            #random value between 0 and 1
            ps = torch.rand((x.shape[0],1), generator=self.rng)
        else:
            rand_index = torch.randint(0, len(self.labelPool), (x.shape[0],), generator=self.rng)
            ps = torch.unsqueeze(torch.Tensor(self.labelPool)[rand_index],dim=-1) #draw random elements

        #move ps to same device as x
        ps = ps.to(x.device)

        return ps
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0, get_preds = False):
        ys, xs, idx = batch
        ys = ys.unsqueeze(-1).float()
        #print(f"ys: {ys}")

        ps = self.forward(xs) # get predictions (probabilities not logits)
        #print(f"ps: {ps}")
       
        return ps, ys, idx

    def training_step(self, batch, batch_idx):
        #Note: for (fast) training, we just compute the loss as metric, not all the other ones.
        # this is why we dont use predict_step() but forward() directly
        ys, xs, idx = batch
        ys = ys.unsqueeze(-1).float()
        #print(f"ys: {ys}")
        ps = self.forward(xs) # get predictions
        #print(f"ps: {ps}")
        loss = self.loss(ps,ys)
        #print(f"loss: {loss} (type: {type(loss)})")
        self.labelPool.extend(ys) #add gt values to pool
        
        #logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        ps, ys, idx = self.predict_step(batch, batch_idx)
        loss = self.loss(ps,ys)

        #Get evaluation metrics
        print(f"ps type: {ps.dtype}, ys type: {ys.dtype}")
        metrics_dict =  getMetrics(preds=ps, gt_labels=ys, prefix="val_")
        prc = metrics_dict.pop("val_prc") # remove/extract PRC
        roc = metrics_dict.pop("val_roc") # remove/extract ROC
        metrics_dict["val_loss"] = loss # add loss
        self.log_dict(metrics_dict, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        ps, ys, idx = self.predict_step(batch, batch_idx)
        loss = self.loss(ps,ys)
        
        #Get evaluation metrics
        metrics_dict = getMetrics(preds=ps, gt_labels=ys, prefix="test_")
        prc = metrics_dict.pop("test_prc") # remove/extract PRC
        roc = metrics_dict.pop("test_roc") # remove/extract ROC
        metrics_dict["test_loss"] = loss # add loss
        self.log_dict(metrics_dict, on_step=False, on_epoch=True, sync_dist=True)

        return loss
    
    def configure_optimizers(self):
        #This is completely superfluous, but required by pytorch-lightning
        return None
    
    def backward(self, loss):
        # Override backward to do nothing
        pass

