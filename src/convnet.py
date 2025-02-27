import torch
import torch.nn as nn
from lightning import LightningModule
import torch.optim as optim


class MyConvNet(LightningModule):
    def __init__(self, cfg):
        super(MyConvNet, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, stride=self.cfg.stride, dilation=self.cfg.dilation),
            nn.BatchNorm2d(8),
            nn.AvgPool2d(2),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=self.cfg.stride, dilation=self.cfg.dilation),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(2),
            nn.ReLU()
        )

        self.lin1 = nn.Linear(in_features=16*7*7, out_features=100)
        self.act1 = nn.LeakyReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.lin2 = nn.Linear(100, self.cfg.n_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = x.view((x.shape[0], -1))
        x = self.lin1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.lin2(x)

        return x

    def basic_step(self, batch, step):
        img, labels = batch
        outputs = self.forward(img)
        y_probs = self.softmax(outputs)

        calculated_metrics = {}
        calculated_metrics[f'{step}/loss'] = self.criterion(outputs, labels)
        calculated_metrics[f'{step}/accuracy'] = torch.sum(y_probs.argmax(dim=1) == labels) / labels.shape[0]

        self.log_dict(calculated_metrics, prog_bar=True)

        return calculated_metrics

    def training_step(self, batch, batch_idx):
        metrics = self.basic_step(batch, "train")
        return metrics["train/loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self.basic_step(batch, "valid")
        return metrics["valid/loss"]

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.cfg.lr)