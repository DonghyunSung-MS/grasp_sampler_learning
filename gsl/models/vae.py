import math
from typing import Union

import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from .base import PLWrapper
from .flow import StandardNormal


def MLP(layers, act=nn.ReLU(inplace=True), bn=True):
    tmp = []
    for i in range(len(layers) - 2):
        tmp.append(nn.Linear(layers[i], layers[i + 1]))
        tmp.append(nn.BatchNorm1d(layers[i + 1]) if bn else nn.Identity())
        tmp.append(act)
    tmp.append(nn.Linear(layers[-2], layers[-1]))
    return nn.Sequential(*tmp)


class VAEGraspNet(PLWrapper):
    def __init__(self, config: OmegaConf) -> None:
        super().__init__(config)
        config = config.algo
        self.net = VAE(config.data_dim, config.condition_dim, config.latent_dim)
        self.beta = config.beta
        self.save_hyperparameters()

    def step_loss(self, batch_data):
        x = batch_data["x"]  # (B, 7) transquat
        if "c" in batch_data.keys():
            c = batch_data["c"]  # (B, condition_dim) or None
        else:
            c = None

        # similar arch as 6dof graspnet
        R = roma.unitquat_to_rotmat(x[..., 3:])
        t = x[..., :3]

        x = torch.cat([t, R.reshape(-1, 9)], -1)  # (B, 12)

        x_hat, kld = self.net(x, c)

        t_hat = x_hat[..., :3]
        R_hat = x_hat[..., 3:]
        R_hat = roma.special_procrustes(R_hat.reshape(-1, 3, 3))

        trans_recon = (torch.norm(t - t_hat, 2, -1) ** 2).mean()
        rot_recon = (torch.norm(R - R_hat, dim=[1, 2]) ** 2).mean()

        recon_loss = trans_recon + rot_recon
        loss = recon_loss + self.beta * kld

        return loss, {
            "loss": loss,
            "KLD": kld,
            "recon_trans": trans_recon,
            "recon_rot": rot_recon,
            "recon": recon_loss,
        }


class VAE(nn.Module):
    def __init__(self, data_dim, condition_dim, latent_dim) -> None:
        super().__init__()

        self.encoder = MLP([data_dim + condition_dim, 256, 256, latent_dim * 2])
        self.decoder = MLP([latent_dim + condition_dim, 256, 256, data_dim])
        self.base_dist = StandardNormal((latent_dim,))

    def forward(self, x, context):
        if context is None:
            mu_logvar = self.encoder(x)
        else:
            mu_logvar = self.encoder(torch.cat([x, context], -1))

        mu, logvar = torch.chunk(mu_logvar, 2, -1)

        # reparam MCMC
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=mu.device)
        z = mu + eps * std

        kld = -0.5 * torch.sum(1.0 + 2.0 * std.log() - mu ** 2 - std ** 2, -1)
        kld = kld.mean()

        if context is None:
            x_hat = self.decoder(z)
        else:
            x_hat = self.decoder(torch.cat([z, context], -1))
        return x_hat, kld

    def sample(self, num_samples, context):
        z = self.base_dist.sample(num_samples)
        if context is None:
            x_hat = self.decoder(z)
        else:
            assert num_samples.shape[0] == context.shape[0]
            x_hat = self.decoder(torch.cat([z, context], -1))
        return x_hat, z
