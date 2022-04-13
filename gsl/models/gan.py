import math
from typing import Union

import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from .base import PLWrapper
from .flow import StandardNormal


# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/lightning_examples/basic-gan.html
class GANGraspNet(PLWrapper):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        algo = config.algo
        # networks
        self.generator = Generator(algo.data_dim, algo.condition_dim, algo.latent_dim)
        self.discriminator = Discriminator(algo.data_dim, algo.condition_dim)
        self.base_dist = StandardNormal((algo.latent_dim,))
        self.save_hyperparameters()

    def forward(self, z, context):
        x_hat = self.generator(z, context)
        t_hat = x_hat[..., :3]
        R_hat = x_hat[..., 3:]
        R_hat = roma.special_procrustes(R_hat.reshape(-1, 3, 3))
        x_hat = torch.cat([t_hat, R_hat.reshape(-1, 9)], -1)
        return x_hat

    def step_loss(self, batch_data, optimizer_idx=1):
        x = batch_data["x"]  # (B, 7) transquat
        if "c" in batch_data.keys():
            c = batch_data["c"]  # (B, condition_dim) or None
        else:
            c = None

        # similar arch as 6dof graspnet
        R = roma.unitquat_to_rotmat(x[..., 3:])
        t = x[..., :3]

        x = torch.cat([t, R.reshape(-1, 9)], -1)  # (B, 12)
        # noise sample
        z = self.base_dist.sample(x.shape[0])

        # generator train
        if optimizer_idx == 0:
            x_hat = self(z, c)
            d_out = self.discriminator(x_hat, c).squeeze(1)
            loss = nn.BCELoss()(d_out, torch.ones(x.shape[0], device=d_out.device))
            log_dict = {"gen_loss": loss}
        # discriminator train
        if optimizer_idx == 1:
            d_out = self.discriminator(x, c).squeeze(1)
            real_loss = nn.BCELoss()(d_out, torch.ones(x.shape[0], device=d_out.device))

            x_hat = self(z, c)
            d_out = self.discriminator(x_hat.detach(), c).squeeze(1)
            fake_loss = nn.BCELoss()(
                d_out, torch.zeros(x.shape[0], device=d_out.device)
            )

            loss = (real_loss + fake_loss) / 2.0
            log_dict = {"loss": loss}

        return loss, log_dict

    # overwrite
    def training_step(self, batch_data, batch_idx, optimizer_idx):
        # print(batch_idx)
        total_loss, log_dict = self.step_loss(batch_data, optimizer_idx)
        self.logging_metric("train", log_dict)
        return total_loss

    def configure_optimizers(self):
        lr = self.config.lr

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        return [opt_g, opt_d], []

    def grasp_latent(self, x, c):
        raise ValueError("GAN has no reverse Mode")

    def sample_grasp(self, num_samples, c=None):
        z = self.base_dist.sample(num_samples)
        if c is None:
            x_hat = self(z, c)
        else:
            assert num_samples == c.shape[0]
            x_hat = self(torch.cat([z, c], -1))

        return x_hat


class Generator(nn.Module):
    def __init__(self, data_dim, condition_dim, latent_dim):
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + condition_dim, 256, normalize=False),
            *block(256, 256, normalize=False),
            nn.Linear(256, data_dim),
        )

    def forward(self, z, context):
        if context is None:
            x_hat = self.model(z)
        else:
            x_hat = self.model(torch.cat([z, context], -1))
        return x_hat


class Discriminator(nn.Module):
    def __init__(self, data_dim, condition_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(data_dim + condition_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, context):
        if context is None:
            validity = self.model(x)
        else:
            validity = self.model(torch.cat([x, context], -1))
        return validity
