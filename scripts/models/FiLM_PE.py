import torch
from torch import nn
import pytorch_lightning as pl

from scripts.training.analyze_utils import getMetrics
import pandas as pd
from pathlib import Path


# The Model
class FiLM_PE(pl.LightningModule): 
    def __init__(self,
                params, # all training related parameters
                embedding_dim=1280, 
                pe_dim=128,
                num_labels=2,
                p_pe_drop=0.2
                ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pe_dim = pe_dim
        self.num_labels = num_labels
        self.params = params
        self._val_probs = []  # to store validation probabilities
        self._val_labels = []  # to store validation labels
        self._val_indices = [] # to store validation indices


        # call this to save init args to the checkpoint
        self.save_hyperparameters()

        #Setup training stuff
        self.criterion_model = torch.nn.CrossEntropyLoss(weight=self.params['crit_weight'],reduction='mean') # evaluation criterion
        self.soft_max = torch.nn.Softmax(dim=1)

        # --- PE → FiLM ---
        # Logic from EsmWithPE in utils.py
        H = embedding_dim
        self.norm = nn.LayerNorm(H)

        self.pe_dropout = nn.Dropout(p_pe_drop)
        self.pe_to_gamma = nn.Sequential(nn.Linear(pe_dim, H), nn.Tanh())
        self.pe_to_beta  = nn.Sequential(nn.Linear(pe_dim, H), nn.Tanh())
        self.gamma_dropout = nn.Dropout(0.1)
        self.beta_dropout  = nn.Dropout(0.1)
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # --- head on pooled (B,H)
        self.classifier = nn.Sequential(
            nn.LayerNorm(H), nn.Dropout(0.2), nn.Linear(H, num_labels)
        )

    def forward(self, x, pe):
        # x: (N, H) - already pooled embeddings from dataset
        # pe: (N, pe_dim) - positional encodings
        # Ensure inputs match model precision
        target_dtype = self.norm.weight.dtype
        if x.dtype != target_dtype:
            x = x.to(target_dtype)
        if pe is not None and pe.dtype != target_dtype:
            pe = pe.to(target_dtype)
        h = self.norm(x)

        # FiLM from PE (safe against NaNs/Inf in PE)
        if pe is not None and pe.shape[-1] == self.pe_dim and self.pe_dim > 2:
            pe = torch.nan_to_num(pe, nan=0.0, posinf=0.0, neginf=0.0)
            pe = self.pe_dropout(pe)
            gamma = self.gamma_dropout(self.pe_to_gamma(pe))
            beta  = self.beta_dropout(self.pe_to_beta(pe))
            h = h * (1 + self.alpha * gamma) + self.alpha * beta

        return self.classifier(h)
    
    def on_validation_epoch_start(self):
        self._val_probs.clear()
        self._val_labels.clear()
        self._val_indices.clear()


    def predict_step(self, batch, batch_idx, dataloader_idx=0, get_preds = False):
        # Expects batch to have 4 elements: ys, xs, pe, idx
        ys, xs, pe, idx = batch
        
        logits = self.forward(xs, pe) # get predictions
        softMaxed = self.soft_max(logits)
        
        #normalize probability
        ps = torch.zeros_like(softMaxed[:,1] )
        for i, (np, p) in enumerate(softMaxed):
            if(np+p != 1.0 ):
                p /= np+p #adapt p
            ps[i] = p

        return logits, ps, ys, idx

    def training_step(self, batch, batch_idx):
        ys, xs, pe, idx = batch
        logits = self.forward(xs, pe) # get predictions
        loss = self.criterion_model(logits,ys)
        
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        logits, ps, ys, idx = self.predict_step(batch, batch_idx)
        loss = self.criterion_model(logits,ys)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self._val_probs.append(ps.detach().cpu())
        self._val_labels.append(ys.detach().cpu())
        self._val_indices.append(idx.detach().cpu())


        return loss
    
    def on_validation_epoch_end(self):
        probs = torch.cat(self._val_probs).numpy()
        labels = torch.cat(self._val_labels).numpy()
        metrics = getMetrics(probs, labels, prefix="val_")
        # Log all metrics
        metrics.pop("val_prc", None)
        metrics.pop("val_roc", None)
        self.log_dict(metrics, sync_dist=True)

        # Save predictions
        indices = torch.cat(self._val_indices).numpy()
        
        # Access Dataset to get Gene_IDs
        # Use simple heuristic to find the core DataSet
        try:
            val_dl = self.trainer.val_dataloaders
            if isinstance(val_dl, list):
                val_dl = val_dl[0]
                
            dataset = val_dl.dataset
            # Unwrap Subset(s)
            while hasattr(dataset, "dataset"):
                dataset = dataset.dataset
                
            if hasattr(dataset, "dataSet_df"):
                gene_ids = dataset.dataSet_df.iloc[indices]["Gene_ID"].values
                
                # Create DataFrame
                results = pd.DataFrame({
                    "Gene_ID": gene_ids,
                    "score": probs,
                    "label": labels
                })
                
                # Save to logger dir or params output dir
                # prioritize trainer.log_dir, fallback to '.'
                save_dir = self.trainer.log_dir or "."
                p = Path(save_dir).joinpath(f"predictions_epoch_{self.current_epoch}.tsv")
                results.to_csv(p, sep="\t", index=False)
                # log(f"Saved test predictions to {p}") # Avoiding log import dependency here, maybe use print or integrate
                print(f"Saved predictions to {p}")
        except Exception as e:
            print(f"Failed to save predictions: {e}")

    def test_step(self, batch, batch_idx):
        logits, ps, ys, idx = self.predict_step(batch, batch_idx)
        loss = self.criterion_model(logits,ys)
        
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