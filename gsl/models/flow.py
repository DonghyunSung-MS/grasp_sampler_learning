import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from gsl.utils.geometry import torch_transquat2transrotvec

from .base import PLWrapper

SCALE_FN = {
    "sigmoid": lambda x: torch.sigmoid(x),
    "sigmoidsoftplus": lambda x: torch.sigmoid(torch.log1p(torch.exp(x))),
    "exp": lambda x: torch.exp(x),
    "softplus": lambda x: torch.log1p(torch.exp(x)),
    "tanh_exp": lambda s: torch.exp(2.0 * torch.tanh(s / 2.0)),
}


class FlowGraspNet(PLWrapper):
    def __init__(self, config: OmegaConf):
        super().__init__(config)
        algo = config.algo

        data_dim = algo.data_dim
        condition_dim = algo.condition_dim
        num_flow = algo.num_flow

        context_net = nn.Identity()
        scale_fn_name = algo.scale_fn_name

        transforms = []

        def cnet():
            return nn.Sequential(
                nn.Linear(data_dim // 2 + condition_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, data_dim // 2 * 2),
            )

        if condition_dim > 0:
            for i in range(num_flow):
                transforms.append(
                    ConditionalAffineCoupling(cnet(), scale_fn_name, context_net)
                )
                transforms.append(ActNorm(data_dim))
                if i != num_flow - 1:
                    transforms.append(Reverse(data_dim))

            self.net = ConditionalFlow(StandardNormal((data_dim,)), transforms)
        elif condition_dim == 0:
            for i in range(num_flow):
                transforms.append(AffineCoupling(cnet(), scale_fn_name))
                transforms.append(ActNorm(data_dim))
                if i != num_flow - 1:
                    transforms.append(Reverse(data_dim))

            self.net = ConditionalFlow(StandardNormal((data_dim,)), transforms)
        else:
            raise ValueError(f"{condition_dim} error !")

        self.save_hyperparameters()  # save for loading

    def step_loss(self, batch_data):
        x = batch_data["x"]  # (B, 7) transquat

        if "c" in batch_data.keys():
            c = batch_data["c"]
        else:
            c = None

        x = torch_transquat2transrotvec(x)  # (B, 6)

        z, logprob = self.net(x, c)
        loss = -logprob.mean()
        return loss, {"loss": loss}

    def sample_grasp(self, num_samples, c=None):
        return self.net.sample(num_samples, c)

    def grasp_latent(self, x, c):
        x = torch_transquat2transrotvec(x)  # (B, 6)
        return self.net(x, c)[0]


"""
Code is Modified from SurVAE public code
"""
# TODO lower triangluar weight norm exp(alpha) * w / |w|
# TODO learned prior
# TODO residual flow(free form) too hard to understand jacobian

SCALE_FN = {
    "sigmoid": lambda x: torch.sigmoid(x),
    "sigmoidsoftplus": lambda x: torch.sigmoid(torch.log1p(torch.exp(x))),
    "exp": lambda x: torch.exp(x),
    "softplus": lambda x: torch.log1p(torch.exp(x)),
    "tanh_exp": lambda s: torch.exp(2.0 * torch.tanh(s / 2.0)),
}


"""
Conditional Affine couping

x1, x2 = chuck(x)
mu, logs = NN(x1, C)
z1 = x1
z2 = mu + x2 * exp(log_s)

log_det_jac = sum(log_s)

x1 = z1
x2 = (z2 - mu) / exp(log_s)

"""


class ConditionalAffineCoupling(nn.Module):
    def __init__(self, coupling_net, scale_fn_name, context_net) -> None:
        super().__init__()
        self.need_context = True
        self.coupling_net = coupling_net
        self.context_net = context_net
        self.scale_fn = SCALE_FN[scale_fn_name]

    def forward(self, x, c):
        # data to noise
        x1, x2 = torch.chunk(x, 2, -1)
        mu_logs = self.coupling_net(torch.cat([x1, self.context_net(c)], -1))
        mu, logs = torch.chunk(mu_logs, 2, -1)
        z1 = x1  # for readability
        z2 = mu + x2 * self.scale_fn(logs)

        z = torch.cat([z1, z2], -1)
        ldj = (torch.log(self.scale_fn(logs))).sum(-1)

        # print("Affine", ldj.mean().item())
        # print(z.min().item(), z.max().item())
        # print("==========")

        return z, ldj

    def inverse(self, z, c):
        # noise to data
        with torch.no_grad():
            z1, z2 = torch.chunk(z, 2, -1)
            mu_logs = self.coupling_net(torch.cat([z1, self.context_net(c)], -1))
            mu, logs = torch.chunk(mu_logs, 2, -1)
            x1 = z1
            x2 = (z2 - mu) / self.scale_fn(logs)

            x = torch.cat([x1, x2], -1)
        return x


class AffineCoupling(nn.Module):
    def __init__(self, coupling_net, scale_fn_name) -> None:
        super().__init__()
        self.need_context = False
        self.coupling_net = coupling_net
        self.scale_fn = SCALE_FN[scale_fn_name]

    def forward(self, x):
        # data to noise
        x1, x2 = torch.chunk(x, 2, -1)
        mu_logs = self.coupling_net(x1)
        mu, logs = torch.chunk(mu_logs, 2, -1)
        z1 = x1  # for readability
        z2 = mu + x2 * self.scale_fn(logs)

        z = torch.cat([z1, z2], -1)
        ldj = (torch.log(self.scale_fn(logs))).sum(-1)

        # print("Affine", ldj.mean().item())
        # print(z.min().item(), z.max().item())
        # print("==========")

        return z, ldj

    @torch.no_grad()
    def inverse(self, z):
        # noise to data
        z1, z2 = torch.chunk(z, 2, -1)
        mu_logs = self.coupling_net(z1)
        mu, logs = torch.chunk(mu_logs, 2, -1)
        x1 = z1
        x2 = (z2 - mu) / self.scale_fn(logs)

        x = torch.cat([x1, x2], -1)
        return x


"""     
Conditional ActNorm

log_s, b = NN(C) s, b in R^d

z = exp(log_s) * x + b

log_det_jac = sum(log_s)

x = (z - b) / exp(log_s)
"""

# AKA affine injector in SuperResolution Flow
class ConditionalActNorm(nn.Module):
    def __init__(self, context_net) -> None:
        super().__init__()
        self.need_context = True
        self.context_net = context_net

    def forward(self, x, c):
        mu_logs = self.context_net(c)
        mu, logs = torch.chunk(mu_logs, 2, -1)
        z = (x - mu) * torch.exp(-logs)

        ldj = -logs.sum(-1)

        # print("actnorm", ldj.mean().item())
        # print(z.min().item(), z.max().item())
        # print("==========")
        return z, ldj

    def inverse(self, z, c):
        with torch.no_grad():
            mu_logs = self.context_net(c)
            mu, logs = torch.chunk(mu_logs, 2, -1)

            x = z * torch.exp(logs) + mu

        return x


"""
reverse
does not affect ldj due to zero return
"""


class Reverse(nn.Module):
    def __init__(self, data_dim, dim=-1):
        super().__init__()
        self.need_context = False
        self.dim = dim
        permutation = torch.arange(data_dim - 1, -1, -1)
        self.register_buffer("permutation", permutation)

    @property
    def inverse_permutation(self):
        return torch.argsort(self.permutation)

    def forward(self, x):
        return torch.index_select(x, self.dim, self.permutation), torch.zeros(
            x.shape[0], device=x.device, dtype=x.dtype
        )

    def inverse(self, z):
        return torch.index_select(z, self.dim, self.inverse_permutation)


class ActNorm(nn.Module):
    def __init__(self, data_dim, data_dep_init=True, eps=1e-6):
        super().__init__()
        self.need_context = False
        self.num_features = data_dim
        self.eps = eps

        self.register_buffer(
            "initialized", torch.zeros(1) if data_dep_init else torch.ones(1)
        )
        self.register_params()

    def data_init(self, x):
        self.initialized += 1.0
        with torch.no_grad():
            x_mean, x_std = self.compute_stats(x)
            # print(x_mean, x_std)
            self.shift.data = x_mean
            self.log_scale.data = torch.log(x_std + self.eps)

    def forward(self, x):
        if self.training and not self.initialized:
            self.data_init(x)
        z = (x - self.shift) * torch.exp(-self.log_scale)
        ldj = torch.sum(-self.log_scale).expand([x.shape[0]]) * self.ldj_multiplier(x)
        return z, ldj

    def inverse(self, z):
        return self.shift + z * torch.exp(self.log_scale)

    def register_params(self):
        self.need_context = False
        """Register parameters shift and log_scale"""
        self.register_parameter(
            "shift", nn.Parameter(torch.zeros(1, self.num_features))
        )
        self.register_parameter(
            "log_scale", nn.Parameter(torch.zeros(1, self.num_features))
        )

    def compute_stats(self, x):
        """Compute x_mean and x_std"""
        x_mean = torch.mean(x, dim=0, keepdim=True)
        x_std = torch.std(x, dim=0, keepdim=True)
        return x_mean, x_std

    def ldj_multiplier(self, x):
        """Multiplier for ldj"""
        return 1


class ConditionalWeight(nn.Module):
    def __init__(self, data_dim, scale_fn_name, context_net):
        super().__init__()
        self.data_dim = data_dim
        self.context_net = context_net
        self.need_context = True

        tril_indices = torch.tril_indices(row=data_dim, col=data_dim, offset=-1)
        self.register_buffer("tril_indices", tril_indices)

        self.num_tril = int(self.data_dim * (self.data_dim - 1) / 2)
        self.scale_fn = SCALE_FN[scale_fn_name]

    def get_weight(self, c: torch.Tensor, forward=True):
        weights_vec = self.context_net(c)

        L_vec = weights_vec[:, : self.num_tril]
        U_vec = weights_vec[:, self.num_tril : self.num_tril * 2]
        D_vec = weights_vec[:, self.num_tril * 2 :]

        L = (
            torch.eye(self.data_dim, device=c.device)
            .unsqueeze(0)
            .repeat(c.shape[0], 1, 1)
        )
        L[:, self.tril_indices[0], self.tril_indices[1]] = L_vec

        U = (
            torch.eye(self.data_dim, device=c.device)
            .unsqueeze(0)
            .repeat(c.shape[0], 1, 1)
        )
        U[:, self.tril_indices[0], self.tril_indices[1]] = U_vec
        U = U.transpose(1, -1)

        D = torch.diag_embed(self.scale_fn(D_vec))

        if forward:
            return L @ D @ U, (torch.log(self.scale_fn(D_vec))).sum(-1)
        else:
            batch_eye = (
                torch.eye(self.data_dim, device=c.device)
                .unsqueeze(0)
                .repeat(c.shape[0], 1, 1)
            )
            L_inv = torch.triangular_solve(batch_eye, L, False, False, True)[0]

            U_inv = torch.triangular_solve(batch_eye, U, True, False, True)[0]

            return U_inv @ torch.diag_embed(1.0 / self.scale_fn(D_vec)) @ L_inv

    def forward(self, x, c):
        w, ldj = self.get_weight(c)

        z = torch.bmm(w, x.unsqueeze(2)).squeeze(2)

        return z, ldj

    def inverse(self, z, c):
        inv_w = self.get_weight(c, False)
        x = torch.bmm(inv_w, z.unsqueeze(2)).squeeze(2)
        return x


##################### DISTRIBUTION ################################


class StandardNormal(nn.Module):
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super(StandardNormal, self).__init__()
        self.need_context = False
        self.shape = torch.Size(shape)
        self.register_buffer("buffer", torch.zeros(1))

    def log_prob(self, x):
        log_base = -0.5 * math.log(2 * math.pi)
        log_inner = -0.5 * x ** 2
        return (log_base + log_inner).reshape(x.shape[0], -1).sum(-1)

    def sample(self, num_samples):
        return torch.randn(
            num_samples, *self.shape, device=self.buffer.device, dtype=self.buffer.dtype
        )


##################### Discrete Conditional Flow #####################


class ConditionalFlow(nn.Module):
    def __init__(self, base_dist, transforms) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        self.base_dist = base_dist

    def forward(self, x, c):
        log_prob = torch.zeros(x.shape[0], device=x.device)

        for transform in self.transforms:
            ldj = None
            if transform.need_context:
                x, ldj = transform(x, c)
            else:
                x, ldj = transform(x)

            log_prob += ldj

        if self.base_dist.need_context:
            log_prob += self.base_dist.log_prob(x, c)
        else:
            log_prob += self.base_dist.log_prob(x)
        z = x  # for readability
        return z, log_prob

    def sample(self, num_samples, c):
        if self.base_dist.need_context:
            z = self.base_dist.sample(c)
        else:
            z = self.base_dist.sample(num_samples)

        for transform in reversed(self.transforms):
            if transform.need_context:
                z = transform.inverse(z, c)
            else:
                z = transform.inverse(z)

        x = z  # for readability
        return x

    def inverse(self, z, c):
        for transform in reversed(self.transforms):
            if transform.need_context:
                z = transform.inverse(z, c)
            else:
                z = transform.inverse(z)

        x = z  # for readability
        return x
