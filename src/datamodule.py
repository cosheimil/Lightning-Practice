import torch.utils.data as data
from lightning import LightningDataModule
from dataset import SignLanguageDataset
import pandas as pd


class SignMnistDataModule(LightningDataModule):
    def __init__(self, cfg, transforms):
        super().__init__()

        self.cfg = cfg
        self.train_transform = transforms

    def _prepare_data(self):
        # Подгружаем данные
        self.train = pd.read_csv(str(self.cfg.dataset_path / 'sign_mnist_train.csv'))
        self.test = pd.read_csv(str(self.cfg.dataset_path / 'sign_mnist_test.csv'))

    def setup(self, stage: str):
        # Подгружаем Dataset
        self._prepare_data()
        self.train_dataset = SignLanguageDataset(self.train, transform=self.train_transform)
        self.valid_dataset = SignLanguageDataset(self.test)

    def _return_dataloader(self, dataset: data.Dataset, phase: str):
        # Базовая функция для создания DataLoader
        return data.DataLoader(
            dataset,
            batch_size=self.cfg.batch_size[phase],
            num_workers=2,
            shuffle=True if phase=='train' else False
        )

    def train_dataloader(self):
        self.train_dl = self._return_dataloader(self.train_dataset, 'train')
        return self.train_dl

    def val_dataloader(self):
        self.valid_dl = self._return_dataloader(self.valid_dataset, 'valid')
        return self.valid_dl