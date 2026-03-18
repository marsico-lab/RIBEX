import torch
from torch import nn
import pytorch_lightning as pl

from scripts.training.analyze_utils import getMetrics

# The Model
class peng_parametrized(pl.LightningModule):  # Model from  but parametrized and pytroch Lightning compatible
    def __init__(self,
                params, # all trainign realted parameters
                hiddem_dim=320, # = 1.25 * 256 (Peng et. al 2019)
                embedding_dim=1280, # = 1.25 * 1024 (Peng et. al 2019)
                num_GRU_layers=6, # 6 (Peng et. al 2019)
                L=1 # sequence length (fixed to 1)
                ):
        super().__init__()
        self.hidden_dim = hiddem_dim
        self.GRU_layers = num_GRU_layers
        self.embedding_dim = embedding_dim
        self.L = L
        self.params = params
        self._val_probs = []  # to store validation probabilities
        self._val_labels = []  # to store validation labels

        # call this to save init args to the checkpoint
        self.save_hyperparameters()

        #Setup training stuff
        self.criterion_model = torch.nn.CrossEntropyLoss(weight=self.params['crit_weight'],reduction='mean') # evaluation criterion
        self.soft_max = torch.nn.Softmax(dim=1)

        #Setup network structure
        self.gru = nn.GRU(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.GRU_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.05,
        )

        self.block1 = nn.Sequential(
            nn.Linear(2 * self.hidden_dim + 2 * self.GRU_layers * self.hidden_dim, self.embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, int(embedding_dim / 2**1)),
            nn.BatchNorm1d(int(embedding_dim / 2**1)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim / 2**1), int(embedding_dim / 2**2)),
        )

        self.block2 = nn.Sequential(
            nn.BatchNorm1d(int(embedding_dim / 2**2)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim / 2**2), int(embedding_dim / 2**3)),
            nn.BatchNorm1d(int(embedding_dim / 2**3)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim / 2**3), int(embedding_dim / 2**4)),
            nn.BatchNorm1d(int(embedding_dim / 2**4)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim / 2**4), 2),  # 80 -> 2 instead of 64->2
        )

    def forward(self, x):
        # N = batch size
        # L = sequence length
        # D = 2 (if bidirectional), 1 otherwise
        # H_in = input_size
        # H_out = hidden_size

        #print("Network input shape:"+str(x.shape))
        x = x.view(x.shape[0], self.L, x.shape[1])  # (N, L) -> (N,L,H_in)
        # print("GRU input shape:"+str(x.shape))
        x = x.to(next(self.gru.parameters()).dtype) # match GRU dtype
        output, hn = self.gru(x)  # (N,L,H_in) -> (N,L,D*H_out) , (D*num_layers, N, H)

        # print("GRU output shape:"+str(output.shape))
        # print("GRU hidden shape:"+str(hn.shape))

        # reformat hidden state
        hn = hn.permute(1, 0, 2)  # (D*num_layers, N, H_out) -> (N, D*num_layers, H_out)
        hn = hn.reshape(hn.shape[0], -1)  # (N, D*num_layers, H_out) -> (N, num_layers*D*H_out)
        # -> (12, 3840)
        # print("Flattend GRU hidden shape:"+str(hn.shape))

        # reformat output
        output = output.reshape(output.shape[0], -1)  # (N,L,D*H_out) -> (N, L*D*H_out)
        # -> (N, 640)
        # print("Flattend GRU output shape:"+str(output.shape))

        # append flattend hidden state to the output
        output = torch.cat([output, hn], 1)  # (N, L*D*H_out + num_layers*D*H_out)
        # 2*self.hidden_dim+2*self.GRU_layers*self.hidden_dim, self.embedding_dim
        # print("Block1 input shape:"+str(output.shape))

        output = self.block1(output)
        # print("Block2 input shape:"+str(output.shape))

        return self.block2(output)
    
    def on_validation_epoch_start(self):
        self._val_probs.clear()
        self._val_labels.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0, get_preds = False):

        ys, xs, idx = batch
        #print(f"ys: {ys}")
        logits = self.forward(xs) # get predictions
        #print(f"logits: {logits}")
        softMaxed = self.soft_max(logits)
        #print(f"softMaxed: {softMaxed}")
        #ps = softMaxed[:,1] #positive probability (positive + negative might not be exactly be 1)
        
        #normalize probability
        ps = torch.zeros_like(softMaxed[:,1] )
        for i, (np, p) in enumerate(softMaxed):
            if(np+p != 1.0 ):
                #print(f"\t{np}+{p}={np+p} (but should be 1.0)")
                p /= np+p #adapt p
            ps[i] = p

        #preds = softMaxed[:,0]<softMaxed[:,1]
        #print(f"preds: {preds}")

        return logits, ps, ys, idx

    def training_step(self, batch, batch_idx):
        #Note: for (fast) training, we just compute the loss as metric, not all the other ones.
        # this is why we dont use predict_step() but forward() directly
        ys, xs, idx = batch
        #print(f"ys: {ys}")
        logits = self.forward(xs) # get predictions
        #print(f"logits: {logits}")
        loss = self.criterion_model(logits,ys)
        #print(f"loss: {loss} (type: {type(loss)})")
        
        #logs metrics for each training_step, and the average across the epoch, to the progress bar and logger
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss
        
    def validation_step(self, batch, batch_idx):
        logits, ps, ys, idx = self.predict_step(batch, batch_idx)
        loss = self.criterion_model(logits,ys)
        self.log("val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True)
        self._val_probs.append(ps.detach().cpu())
        self._val_labels.append(ys.detach().cpu())

        return loss
    
    # After the loader finishes, assemble the epoch data and call the same getMetrics LoRA uses
    def on_validation_epoch_end(self):
        probs = torch.cat(self._val_probs).numpy()
        labels = torch.cat(self._val_labels).numpy()
        metrics = getMetrics(probs, labels, prefix="val_")
        # Log all metrics
        metrics.pop("val_prc", None)
        metrics.pop("val_roc", None)
        self.log_dict(metrics, sync_dist=True)

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
        #return {"optimizer":optimizer}