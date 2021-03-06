import os

os.environ["WANDB_API_KEY"] = "6325b41c1a09c4bc612dd4c50f5e3791dbb9eabe"
os.environ[
    "WANDB_START_METHOD"
] = "fork"  # this is important to compatible with joblib loky backend

from pathlib import Path

import hydra
import pytorch_lightning as pl
import wandb
from joblib.externals.loky.backend.context import get_context
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from gsl import ROOT
from gsl.dataset import GraspOnly, read_grasps
from gsl.models import MODEL_ZOO


def seeding(seed):
    pl.seed_everything(seed)


def get_dataloader(config, object_name):
    grasp_path = ROOT / f"data/grasps/{object_name}/grasps.h5"
    trainset, evalset, testset = read_grasps(
        grasp_path, config.num_train, config.num_test
    )

    train_loader = DataLoader(
        GraspOnly(trainset),
        batch_size=config.batch_size,
        num_workers=8,
        shuffle=True,
        multiprocessing_context=get_context("loky"),
    )

    val_loader = DataLoader(
        GraspOnly(evalset),
        batch_size=config.batch_size,
        num_workers=8,
        multiprocessing_context=get_context("loky"),
    )

    test_loader = DataLoader(
        GraspOnly(testset),
        batch_size=config.batch_size,
        num_workers=8,
        shuffle=False,
        multiprocessing_context=get_context("loky"),
    )

    return train_loader, val_loader, test_loader


def get_logger_and_callbacks(config, object_name):
    seed = config.seed
    config = config.algo

    wandb_logger = WandbLogger(
        project="GraspSamplerToy",
        name=object_name + config.method,
    )
    dirpath = None
    model_ckpt = None
    if config.method == "VAE":
        dirpath = str(
            ROOT
            / f"checkpoints/{object_name}/{config.method}_z{config.latent_dim}_kl{config.beta}/{seed}"
        )
        model_ckpt = ModelCheckpoint(
            dirpath=dirpath,
            monitor="val/loss",
            save_top_k=1,
        )
    elif config.method == "FLOW":
        dirpath = str(
            ROOT
            / f"checkpoints/{object_name}/{config.method}_scale{config.scale_fn_name}_N{config.num_flow}/{seed}"
        )
        model_ckpt = ModelCheckpoint(
            dirpath=dirpath,
            monitor="val/loss",
            save_top_k=1,
        )
    elif config.method == "GAN":
        dirpath = str(
            ROOT
            / f"checkpoints/{object_name}/{config.method}_z{config.latent_dim}/{seed}"
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


def train_each(config, object_name):
    seeding(config.seed)

    train_loader, val_loader, test_loader = get_dataloader(config, object_name)

    model = MODEL_ZOO[config.algo.method](config)

    loggers, callbacks = get_logger_and_callbacks(config, object_name)

    trainer = pl.Trainer(
        logger=loggers,
        log_every_n_steps=1,
        gpus=config.gpu,
        max_epochs=config.max_epoch,
        callbacks=callbacks,
    )

    trainer.fit(model, train_loader, val_loader)

    trainer.test(model, test_loader, ckpt_path="best")
    wandb.finish()  # for multiple run


@hydra.main(config_path="../../config/", config_name="config")
def main(config):
    print(OmegaConf.to_yaml(config))
    for object_name in config.object.object_names:
        print(object_name)
        # continue
        train_each(config, object_name)


if __name__ == "__main__":
    main()
