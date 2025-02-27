from argparse import ArgumentParser
from pathlib import Path

import torch
from lightning import seed_everything
from lightning.pytorch.loggers import CSVLogger

import torchvision.transforms.v2 as T

from dataclasses import dataclass


from datamodule import SignMnistDataModule
from convnet import MyConvNet
from trainer import get_trainer


# Config
@dataclass
class CFG:
    model_name = "sign_mnist_cnn"
    dataset_path = Path('data/')
    model_path = Path('models/')
    batch_size = {
        'train': 200,
        'valid': 100,
    }

    debug = True

    seed = 2025

    stride = 1
    dilation = 1
    n_classes = 25

    lr = 1e-3
    epochs = 10
    device_count = 1

trainer_params = {
    'accelerator': 'cpu',
    # 'devices': CFG.device_count,
    'max_epochs': CFG.epochs,
    'log_every_n_steps': 10,
    'deterministic': True,

    'logger': CSVLogger("logs")
}

def _print_center(text):
    print("{:=^50}".format(text))

def create_augs():
    return T.Compose([
        T.RandomHorizontalFlip(p=0.1),
        T.RandomApply([
            T.RandomRotation(degrees=(-180, 180))],
            p=0.2
        ),
        # transforms.ToDtype(torch.float32, scale=True),
        # transforms.Normalize([mean], [std]),
    ])

def load_dataset(train_transforms):
    if CFG.debug:
        _print_center("Подгружаем датасет")
    datamodule = SignMnistDataModule(CFG, train_transforms)
    alphabet = [chr(i) for i in range(ord("A"), ord("Z"))]
    letters_dict = {
        idx: elem for idx, elem in enumerate(alphabet)
    }

    return datamodule, letters_dict

def check_model(trainer, model, dm):
    if trainer.fast_dev_run:
        _print_center("Проверяем модель")
        try:
            trainer.fit(model, dm)
            _print_center("Тестовый прогон успешно пройден")
        except:
            raise ValueError("Тестовый прогон завершился с ошибкой")

def get_predictions(model, batch):
    img, labels = batch
    preds = model.forward(img)
    preds = torch.softmax(preds, dim=1)

    return preds, labels

def convert2letter(y, letters_dict):
    return [letters_dict[i.item()] for i in y]

def main():
    parser = ArgumentParser()
    parser.add_argument("--fast-dev-run", action='store_true')
    args = parser.parse_args()
    fast_dev_run_flag = args.fast_dev_run
    print(fast_dev_run_flag)

    seed_everything(CFG.seed)

    train_transforms = create_augs()
    dm, letters_dict = load_dataset(train_transforms)
    model = MyConvNet(CFG)
    trainer = get_trainer(trainer_params)
    check_trainer = get_trainer(trainer_params, fast_dev_run=fast_dev_run_flag)

    check_model(check_trainer, model, dm)
    _print_center("Обучаем модель")
    trainer.fit(model, dm)

    dm.setup(stage='valid')
    test_dl = dm.val_dataloader()
    test_dl = iter(test_dl)

    batch = next(test_dl)

    if CFG.debug:
        _print_center("Inference")

    y_probs, y_true = get_predictions(model, batch)

    if CFG.debug:
        _print_center("Inference end!")
        _print_center("Metrics")
        accuracy = torch.sum(y_probs.argmax(dim=1) == y_true) / y_true.shape[0]
        print(f'Accuracy: {accuracy}')


if __name__ == "__main__":
    main()