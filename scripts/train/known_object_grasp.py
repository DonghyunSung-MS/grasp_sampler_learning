import os
from typing import List

os.environ["WANDB_API_KEY"] = "6325b41c1a09c4bc612dd4c50f5e3791dbb9eabe"
os.environ["WANDB_START_METHOD"] = "fork" # this is important to compatible with joblib loky backend

from pathlib import Path

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from gsl import ROOT
from gsl.dataset import RotAugCategoryGrasp, read_grasps
from gsl.models import MODEL_ZOO
from joblib.externals.loky.backend.context import get_context


def seeding(seed):
    pl.seed_everything(seed)


def get_dataloader(config, object_names):

    trainsets = []
    evalsets = []
    testsets = []

    for object_name in object_names:
        grasp_path = ROOT / f"data/grasps/{object_name}/grasps.h5"
        trainset, evalset, testset = read_grasps(grasp_path, config.num_train, config.num_test)

        trainsets.append(trainset)
        evalsets.append(evalset)
        testsets.append(testset)

    train_loader = DataLoader(
        RotAugCategoryGrasp(trainsets),
        batch_size=config.batch_size,
        num_workers=8,
        shuffle=True,
        multiprocessing_context=get_context('loky')
    )

    val_loader = DataLoader(
        RotAugCategoryGrasp(evalsets),
        batch_size=config.batch_size,
        num_workers=8,
        multiprocessing_context=get_context('loky')

    )

    test_loader = DataLoader(
        RotAugCategoryGrasp(testsets),
        batch_size=config.batch_size,
        num_workers=8,
        shuffle=False,
        multiprocessing_context=get_context('loky')
    )

    return train_loader, val_loader, test_loader


def get_logger_and_callbacks(config, object_names:List):
    seed = config.seed
    config = config.algo

    run_name = "+".join(object_names)

    wandb_logger = WandbLogger(
        # project="GraspSamplerToy",
        project="GraspSamplerTest",
        name=run_name + config.method,
    )
    dirpath = None
    model_ckpt = None
    if config.method == "VAE":
        dirpath = str(
            ROOT
            / f"checkpoints/{run_name}/{config.method}_z{config.latent_dim}_kl{config.beta}/{seed}"
        )
        model_ckpt = ModelCheckpoint(
            dirpath=dirpath,
            monitor="val/loss",
            save_top_k=1,
        )
    elif config.method == "FLOW":
        dirpath = str(
            ROOT
            / f"checkpoints/{run_name}/{config.method}_scale{config.scale_fn_name}_N{config.num_flow}/{seed}"
        )
        model_ckpt = ModelCheckpoint(
            dirpath=dirpath,
            monitor="val/loss",
            save_top_k=1,
        )
    elif config.method == "GAN":
        dirpath = str(
            ROOT
            / f"checkpoints/{run_name}/{config.method}_z{config.latent_dim}/{seed}"
        )
        # save lateset only
        model_ckpt = ModelCheckpoint(
            dirpath=dirpath,
        )

    callbacks = [
        model_ckpt,
        LearningRateMonitor(),
    ]

    return [wandb_logger], callbacks

def train_all(config, object_names):
    seeding(config.seed)

    train_loader, val_loader, test_loader = get_dataloader(config, object_names)

    config.algo.condition_dim = len(object_names)

    model = MODEL_ZOO[config.algo.method](config)

    loggers, callbacks = get_logger_and_callbacks(config, object_names)

    trainer = pl.Trainer(
        logger=loggers,
        log_every_n_steps=1,
        gpus=config.gpu,
        max_epochs=config.max_epoch,
        callbacks=callbacks,
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader, ckpt_path="best")
    wandb.finish() # for multiple run


@hydra.main(config_path="../../config/", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    train_all(config, config.object.object_names)

if __name__ == "__main__":
    main()
