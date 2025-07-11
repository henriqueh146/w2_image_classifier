import torch
import torch.nn.functional as F
import torchvision.models as models

from torch import nn
from pytorch_lightning import LightningModule

from sklearn.metrics import accuracy_score, f1_score


class BinaryImageClassifier(LightningModule):

    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        base_model = models.resnet18(pretrained=True)
        base_model.fc = nn.Linear(base_model.fc.in_features, 1)
        self.model = base_model


    def forward(self, x):
        return torch.sigmoid(self.model(x))
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.binary_cross_entropy(y_hat, y.float())
        self.log("train_loss", loss)
        return loss
    
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()
        loss = F.binary_cross_entropy(y_hat, y.float())
        preds = (y_hat > 0.5).int()
        acc = accuracy_score(y.cpu(), preds.cpu())
        f1 = f1_score(y.cpu(), preds.cpu())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
    

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
