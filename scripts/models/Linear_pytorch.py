import torch
import torch.nn as nn
import pytorch_lightning as pl

from scripts.training.analyze_utils import getMetrics #to compute all our metrics

class linearClassififer(pl.LightningModule):
    def __init__(self, params, embedding_dim):
        super().__init__()
        self.params = params
        self.embedding_dim = embedding_dim

        #Architecture
        self.fc = nn.Linear(embedding_dim, 1)  # embedding_dim -> 1 (one output neuron)

        #Training
        #self.loss = torch.nn.BCELoss(weight=self.params['crit_weight']) # Binary Cross Entropy Loss
        # we implemented this ourself to properly support class weights

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
        # N = batch size
        # L = sequence length

        #print(f"Network input shape:{x.shape}")
        output = self.fc(x) # (N, L) -> (N, 1)
        output = torch.sigmoid(output) # keeps dim, just scales to [0,1]
        #output = output.view(-1) # (N, 1) -> (N)
        #print(f"\tOutput shape: {output.shape}")

        return output
    
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
        
        #logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        ps, ys, idx = self.predict_step(batch, batch_idx)
        loss = self.loss(ps,ys)

        #Get evaluation metrics
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

        optimizer = torch.optim.Adam(self.parameters(), lr=self.params["lr"], weight_decay=self.params["weight_decay"], capturable=True)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.params["patience"], factor=self.params["factor"], verbose=True)

        return {"optimizer":optimizer, "lr_scheduler":lr_scheduler, "monitor": "val_loss"}