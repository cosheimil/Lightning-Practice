from lightning import Trainer


def get_trainer(trainer_params, fast_dev_run=False):
    if fast_dev_run:
        trainer_params['fast_dev_run'] = True

    return Trainer(**trainer_params)