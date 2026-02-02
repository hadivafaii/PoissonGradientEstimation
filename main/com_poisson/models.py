from itertools import permutations

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


def poisson_log_prob(x, rate, eps=1e-8):
    return -rate + x * torch.log(torch.clamp(rate, min=eps)) - torch.lgamma(x + 1)


def com_poisson_log_prob(x, rate, nu, max_iter=100, eps=1e-8):
    """
    x: Tensor of shape (...,)
    rate: Tensor of shape (...,)
    nu: Tensor of shape (...,)
    """
    log_lam = nu * torch.log(torch.clamp(rate + (nu - 1) / (2 * nu), min=eps))
    k_list = torch.arange(0, max_iter, device=x.device).float()  # (max_iter,)
    log_z = torch.logsumexp(
        log_lam.unsqueeze(-1) @ k_list.unsqueeze(0)
        - nu.unsqueeze(-1) @ torch.lgamma(k_list + 1).unsqueeze(0),  # (..., max_iter)
        dim=-1,
    )  # (...,)
    return x * log_lam - torch.lgamma(x + 1) * nu - log_z


@torch.inference_mode()
def compute_com_poisson_upperbound(rate):
    """
    rate: Tensor of shape (...,)
    """
    max_rate = rate.max().item()
    upperbound = int(max_rate + 10 * np.sqrt(max_rate))
    return upperbound


def com_poisson_logit(rate, nu, upperbound=100, eps=1e-8):
    """
    rate: Tensor of shape (...,)
    nu: Tensor of shape (...,)
    """
    log_lam = nu * torch.log(torch.clamp(rate + (nu - 1) / (2 * nu), min=eps))
    k_list = torch.arange(0, upperbound, device=rate.device).float()  # (upperbound,)
    logit = log_lam.unsqueeze(-1) @ k_list.unsqueeze(0) - nu.unsqueeze(
        -1
    ) @ torch.lgamma(k_list + 1).unsqueeze(
        0
    )  # (..., upperbound)
    return logit


def cubic_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Cubic Hermite interpolation (Smoothstep).
    Maps [-1, 1] to [0, 1] with C1 smoothness (zero derivative at boundaries).
    f(u) = 3u^2 - 2u^3, where u = (x+1)/2
    """
    # 1. Normalize x from [-1, 1] to u in [0, 1]
    u = torch.clamp(0.5 * x + 0.5, min=0.0, max=1.0)
    # 2. Cubic polynomial
    return 3 * u.pow(2) - 2 * u.pow(3)


class LinearNonlinearBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        return x


class LogSigmoid(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(torch.clamp(10 * torch.sigmoid(x), min=1e-8))


class TwoLayerNetwork(nn.Module):
    def __init__(
        self,
        dims: tuple[int, ...],
    ):
        super().__init__()
        self.dims = dims
        self.prior_log_rate = nn.Parameter(torch.zeros(dims[0]))
        self.prior_log_nu = nn.Parameter(torch.zeros(dims[0]))
        self.decoder_log_nu = nn.Parameter(torch.zeros(dims[-1]))
        self.encoder_log_nu = nn.Parameter(torch.zeros(dims[0]))
        self.decoder = nn.Sequential(
            *[
                (
                    LinearNonlinearBlock(dims[i], dims[i + 1])
                    if i < len(dims) - 2
                    else nn.Linear(dims[i], dims[i + 1])
                )
                for i in range(len(dims) - 1)
            ]
            + [LogSigmoid()]
        )  # output log_rate
        reversed_dims = dims[::-1]
        self.encoder = nn.Sequential(
            *[
                (
                    LinearNonlinearBlock(reversed_dims[i], reversed_dims[i + 1])
                    if i < len(reversed_dims) - 2
                    else nn.Linear(reversed_dims[i], reversed_dims[i + 1])
                )
                for i in range(len(reversed_dims) - 1)
            ]
            + [LogSigmoid()]
        )  # output log_rate

    @torch.inference_mode()
    def sample(self, n_samples: int = 1):
        prior_log_rate = self.prior_log_rate.unsqueeze(0).expand(n_samples, -1)
        prior_rate = prior_log_rate.exp()
        prior_log_nu = self.prior_log_nu.unsqueeze(0).expand(n_samples, -1)
        prior_upperbound = compute_com_poisson_upperbound(prior_rate)
        prior_logit = com_poisson_logit(
            rate=prior_rate,
            nu=prior_log_nu.exp(),
            upperbound=prior_upperbound,
        )  # (n_samples, dims[0], prior_upperbound)

        cat = torch.distributions.Categorical(logits=prior_logit, validate_args=False)
        z_samples = cat.sample().float()  # (n_samples, dims[0])

        decoder_log_rate = self.decoder(z_samples)  # (n_samples, dims[-1])
        decoder_rate = decoder_log_rate.exp()
        decoder_log_nu = self.decoder_log_nu.unsqueeze(0).expand(n_samples, -1)
        decoder_upperbound = compute_com_poisson_upperbound(decoder_rate)
        decoder_logit = com_poisson_logit(
            rate=decoder_rate, nu=decoder_log_nu.exp(), upperbound=decoder_upperbound
        )  # (n_samples, dims[-1], decoder_upperbound)

        cat = torch.distributions.Categorical(logits=decoder_logit, validate_args=False)
        x_samples = cat.sample().float()  # (n_samples, dims[-1])

        return z_samples, x_samples


class LitScore(L.LightningModule):
    def __init__(
        self,
        model: TwoLayerNetwork,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

    def training_step(self, batch, batch_idx):
        x = batch[0]  # (batch, dims[-1])
        batch_size = x.shape[0]

        # Encoder rates
        encoder_log_rate = self.model.encoder(x)  # (batch, dims[0])
        encoder_rate = encoder_log_rate.exp()
        encoder_log_nu = self.model.encoder_log_nu.unsqueeze(0).expand(batch_size, -1)
        encoder_upperbound = compute_com_poisson_upperbound(encoder_rate)
        encoder_logit = com_poisson_logit(
            rate=encoder_rate, nu=encoder_log_nu.exp(), upperbound=encoder_upperbound
        )  # (batch, dims[0], encoder_upperbound)
        encoder_cat = torch.distributions.Categorical(
            logits=encoder_logit, validate_args=False
        )
        z_samples = encoder_cat.sample().float()  # (batch, dims[0])
        ln_qzgx = encoder_cat.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Prior rates
        prior_log_rate = self.model.prior_log_rate.unsqueeze(0).expand(batch_size, -1)
        prior_log_nu = self.model.prior_log_nu.unsqueeze(0).expand(batch_size, -1)
        prior_logit = com_poisson_logit(
            rate=prior_log_rate.exp(),
            nu=prior_log_nu.exp(),
            upperbound=encoder_upperbound,
        )  # (batch, dims[0], encoder_upperbound)
        prior_cat = torch.distributions.Categorical(
            logits=prior_logit, validate_args=False
        )
        ln_pz = prior_cat.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Decoder rates
        decoder_log_rate = self.model.decoder(z_samples)  # (batch, dims[-1])
        decoder_rate = decoder_log_rate.exp()
        decoder_log_nu = self.model.decoder_log_nu.unsqueeze(0).expand(batch_size, -1)
        decoder_upperbound = max(
            compute_com_poisson_upperbound(decoder_rate),
            x.max().int().item() + 1,
        )
        decoder_logit = com_poisson_logit(
            rate=decoder_rate, nu=decoder_log_nu.exp(), upperbound=decoder_upperbound
        )  # (batch, dims[-1], 40)
        decoder_cat = torch.distributions.Categorical(
            logits=decoder_logit, validate_args=False
        )
        ln_pxgz = decoder_cat.log_prob(x).sum(dim=-1)  # (batch,)

        ln_p = ln_pz + ln_pxgz  # (batch,)
        ln_q = ln_qzgx  # (batch,)

        ln_p_values = ln_p.detach()
        ln_q_values = ln_q.detach()
        elbo_values = ln_p_values - ln_q_values
        elbo = (
            ln_p - ln_p_values + elbo_values * (ln_q - ln_q_values) + elbo_values
        )  # (batch,)
        loss = -elbo.mean()

        self.log_dict({"train/elbo": -loss.item()})
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]  # (batch, dims[-1])
        z = batch[1]  # (batch, dims[0])
        batch_size = x.shape[0]

        # Encoder rates
        encoder_log_rate = self.model.encoder(x)  # (batch, dims[0])
        encoder_log_nu = self.model.encoder_log_nu.unsqueeze(0).expand(batch_size, -1)
        encoder_rate = encoder_log_rate.exp()
        encoder_upperbound = max(
            compute_com_poisson_upperbound(encoder_rate),
            z.max().int().item() + 1,
        )
        encoder_logit = com_poisson_logit(
            rate=encoder_rate, nu=encoder_log_nu.exp(), upperbound=encoder_upperbound
        )  # (batch, dims[0], encoder_upperbound)
        encoder_cat = torch.distributions.Categorical(
            logits=encoder_logit,
            validate_args=False,
        )
        z_samples = encoder_cat.sample().float()  # (batch, dims[0])
        ln_qzgx = encoder_cat.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Prior rates
        prior_log_rate = self.model.prior_log_rate.unsqueeze(0).expand(batch_size, -1)
        prior_log_nu = self.model.prior_log_nu.unsqueeze(0).expand(batch_size, -1)
        prior_logit = com_poisson_logit(
            rate=prior_log_rate.exp(),
            nu=prior_log_nu.exp(),
            upperbound=encoder_upperbound,
        )  # (batch, dims[0], encoder_upperbound)
        prior_cat = torch.distributions.Categorical(
            logits=prior_logit, validate_args=False
        )
        ln_pz = prior_cat.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Decoder rates
        decoder_log_rate = self.model.decoder(z_samples)  # (batch, dims[-1])
        decoder_rate = decoder_log_rate.exp()
        decoder_log_nu = self.model.decoder_log_nu.unsqueeze(0).expand(batch_size, -1)
        decoder_upperbound = max(
            compute_com_poisson_upperbound(decoder_rate),
            x.max().int().item() + 1,
        )
        decoder_logit = com_poisson_logit(
            rate=decoder_rate, nu=decoder_log_nu.exp(), upperbound=decoder_upperbound
        )  # (batch, dims[-1], decoder_upperbound)
        decoder_cat = torch.distributions.Categorical(
            logits=decoder_logit,
            validate_args=False,
        )
        ln_pxgz = decoder_cat.log_prob(x).sum(dim=-1)  # (batch,)

        ln_p = ln_pz + ln_pxgz  # (batch,)
        ln_q = ln_qzgx  # (batch,)

        ln_p_values = ln_p.detach()
        ln_q_values = ln_q.detach()
        elbo_values = ln_p_values - ln_q_values
        elbo = (
            ln_p - ln_p_values + elbo_values * (ln_q - ln_q_values) + elbo_values
        )  # (batch,)

        hidden_log_likelihood = encoder_cat.log_prob(z).sum(dim=-1)  # (batch,)

        decoder_log_rate = self.model.decoder(z)  # (batch, dims[-1])
        decoder_log_nu = self.model.decoder_log_nu.unsqueeze(0).expand(batch_size, -1)
        decoder_upperbound = max(
            compute_com_poisson_upperbound(decoder_log_rate.exp()),
            x.max().int().item() + 1,
        )
        decoder_logit = com_poisson_logit(
            rate=decoder_log_rate.exp(),
            nu=decoder_log_nu.exp(),
            upperbound=decoder_upperbound,
        )  # (batch, dims[-1], decoder_upperbound)
        decoder_cat = torch.distributions.Categorical(
            logits=decoder_logit,
            validate_args=False,
        )
        conditional_log_likelihood = decoder_cat.log_prob(x).sum(dim=-1)  # (batch,)

        self.log_dict(
            {
                "val/elbo": elbo.mean().item(),
                "val/cll": conditional_log_likelihood.mean().item(),
                "val/hll": hidden_log_likelihood.mean().item(),
            }
        )

    def on_test_epoch_start(self):
        self.df_metrics = pd.DataFrame(columns=["elbo", "cll", "hll"])
        self.z_samples_list = []

    def test_step(self, batch, batch_idx):
        x = batch[0]  # (batch, dims[-1])
        z = batch[1]  # (batch, dims[0])
        batch_size = x.shape[0]

        # Encoder rates
        encoder_log_rate = self.model.encoder(x)  # (batch, dims[0])
        encoder_rate = encoder_log_rate.exp()
        encoder_log_nu = self.model.encoder_log_nu.unsqueeze(0).expand(batch_size, -1)
        encoder_upperbound = max(
            compute_com_poisson_upperbound(encoder_rate),
            z.max().int().item() + 1,
        )
        encoder_logit = com_poisson_logit(
            rate=encoder_rate, nu=encoder_log_nu.exp(), upperbound=encoder_upperbound
        )  # (batch, dims[0], encoder_upperbound)
        encoder_cat = torch.distributions.Categorical(
            logits=encoder_logit,
            validate_args=False,
        )
        z_samples = encoder_cat.sample().float()  # (batch, dims[0])
        ln_qzgx = encoder_cat.log_prob(z_samples).sum(dim=-1)  # (batch,)
        self.z_samples_list.append(z_samples.cpu())

        # Prior rates
        prior_log_rate = self.model.prior_log_rate.unsqueeze(0).expand(batch_size, -1)
        proir_rate = prior_log_rate.exp()
        prior_log_nu = self.model.prior_log_nu.unsqueeze(0).expand(batch_size, -1)
        prior_logit = com_poisson_logit(
            rate=proir_rate,
            nu=prior_log_nu.exp(),
            upperbound=encoder_upperbound,
        )  # (batch, dims[0], encoder_upperbound)
        prior_cat = torch.distributions.Categorical(
            logits=prior_logit, validate_args=False
        )
        ln_pz = prior_cat.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Decoder rates
        decoder_log_rate = self.model.decoder(z_samples)  # (batch, dims[-1])
        decoder_rate = decoder_log_rate.exp()
        decoder_log_nu = self.model.decoder_log_nu.unsqueeze(0).expand(batch_size, -1)
        decoder_upperbound = max(
            compute_com_poisson_upperbound(decoder_rate),
            x.max().int().item() + 1,
        )
        decoder_logit = com_poisson_logit(
            rate=decoder_rate, nu=decoder_log_nu.exp(), upperbound=decoder_upperbound
        )  # (batch, dims[-1], decoder_upperbound)
        decoder_cat = torch.distributions.Categorical(
            logits=decoder_logit,
            validate_args=False,
        )
        ln_pxgz = decoder_cat.log_prob(x).sum(dim=-1)  # (batch,)

        ln_p = ln_pz + ln_pxgz  # (batch,)
        ln_q = ln_qzgx  # (batch,)

        ln_p_values = ln_p.detach()
        ln_q_values = ln_q.detach()
        elbo_values = ln_p_values - ln_q_values
        elbo = (
            ln_p - ln_p_values + elbo_values * (ln_q - ln_q_values) + elbo_values
        )  # (batch,)

        hidden_log_likelihood = encoder_cat.log_prob(z).sum(dim=-1)  # (batch,)

        decoder_log_rate = self.model.decoder(z)  # (batch, dims[-1])
        decoder_rate = decoder_log_rate.exp()
        decoder_log_nu = self.model.decoder_log_nu.unsqueeze(0).expand(batch_size, -1)
        decoder_upperbound = max(
            compute_com_poisson_upperbound(decoder_rate),
            x.max().int().item() + 1,
        )
        decoder_logit = com_poisson_logit(
            rate=decoder_rate,
            nu=decoder_log_nu.exp(),
            upperbound=decoder_upperbound,
        )  # (batch, dims[-1], decoder_upperbound)
        decoder_cat = torch.distributions.Categorical(
            logits=decoder_logit,
            validate_args=False,
        )
        conditional_log_likelihood = decoder_cat.log_prob(x).sum(dim=-1)  # (batch,)

        self.df_metrics.at[0, "elbo"] = elbo.mean().item()
        self.df_metrics.at[0, "cll"] = conditional_log_likelihood.mean().item()
        self.df_metrics.at[0, "hll"] = hidden_log_likelihood.mean().item()

    def on_test_epoch_end(self):
        self.z_samples = torch.cat(self.z_samples_list, dim=0)  # (n_samples, dims[0])
        dispersion = self.z_samples.var(dim=0) / self.z_samples.mean(dim=0)
        self.df_metrics.at[0, "min_dispersion"] = dispersion.min().item()
        self.df_metrics.at[0, "max_dispersion"] = dispersion.max().item()
        self.df_metrics.at[0, "dispersion_range"] = (
            dispersion.max().item() - dispersion.min().item()
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        return optimizer


class LitGS(LitScore):
    def __init__(
        self,
        model: TwoLayerNetwork,
        tau: float = 0.2,
    ):
        super().__init__(
            model=model,
        )
        self.save_hyperparameters(ignore=["model"])

    def aggregate_samples(self, gumbel_samples: torch.Tensor):
        upperbound = gumbel_samples.shape[-1]
        k = torch.arange(upperbound, device=gumbel_samples.device).float()
        return gumbel_samples @ k

    def training_step(self, batch, batch_idx):
        x = batch[0]  # (batch, dims[-1])
        batch_size = x.shape[0]

        # Encoder rates
        encoder_log_rate = self.model.encoder(x)  # (batch, dims[0])
        encoder_rate = encoder_log_rate.exp()
        encoder_log_nu = self.model.encoder_log_nu.unsqueeze(0).expand(batch_size, -1)
        encoder_upperbound = compute_com_poisson_upperbound(encoder_rate)
        encoder_logit = com_poisson_logit(
            rate=encoder_rate, nu=encoder_log_nu.exp(), upperbound=encoder_upperbound
        )  # (batch, dims[0], upperbound)
        z_gs_samples = F.gumbel_softmax(
            encoder_logit,
            tau=self.hparams.tau,
            hard=False,
            dim=-1,
        )  # (batch, dims[0], upperbound)
        z_samples = self.aggregate_samples(z_gs_samples)  # (batch, dims[0])
        ln_qzgx = -F.cross_entropy(
            encoder_logit.permute(0, 2, 1),  # (batch, upperbound, dims[0])
            z_gs_samples.permute(0, 2, 1),  # (batch, upperbound, dims[0])
            reduction="none",
        ).sum(
            dim=-1
        )  # (batch,)

        # Prior rates
        prior_log_rate = self.model.prior_log_rate.unsqueeze(0).expand(batch_size, -1)
        prior_log_nu = self.model.prior_log_nu.unsqueeze(0).expand(batch_size, -1)
        prior_logit = com_poisson_logit(
            rate=prior_log_rate.exp(),
            nu=prior_log_nu.exp(),
            upperbound=encoder_upperbound,
        )  # (batch, dims[0], upperbound)
        ln_pz = -F.cross_entropy(
            prior_logit.permute(0, 2, 1),  # (batch, upperbound, dims[0])
            z_gs_samples.permute(0, 2, 1),  # (batch, upperbound, dims[0])
            reduction="none",
        ).sum(
            dim=-1
        )  # (batch,)

        # Decoder rates
        decoder_log_rate = self.model.decoder(z_samples)  # (batch, dims[-1])
        decoder_log_nu = self.model.decoder_log_nu.unsqueeze(0).expand(batch_size, -1)
        decoder_upperbound = max(
            compute_com_poisson_upperbound(decoder_log_rate.exp()),
            x.max().int().item() + 1,
        )
        decoder_logit = com_poisson_logit(
            rate=decoder_log_rate.exp(),
            nu=decoder_log_nu.exp(),
            upperbound=decoder_upperbound,
        )  # (batch, dims[-1], upperbound)
        decoder_cat = torch.distributions.Categorical(
            logits=decoder_logit, validate_args=False
        )
        ln_pxgz = decoder_cat.log_prob(x).sum(dim=-1)  # (batch,)
        elbo = ln_pz + ln_pxgz - ln_qzgx  # (batch,)
        loss = -elbo.mean()

        self.log_dict({"train/elbo": -loss.item()})
        return loss


class LitExpSigmoid(LitScore):
    def __init__(
        self,
        model: TwoLayerNetwork,
        tau: float = 0.2,
    ):
        super().__init__(
            model=model,
        )
        self.save_hyperparameters(ignore=["model"])

    def aggregate_samples(self, gumbel_samples: torch.Tensor):
        upperbound = gumbel_samples.shape[-1]
        k = torch.arange(upperbound, device=gumbel_samples.device).float()
        return gumbel_samples @ k

    def training_step(self, batch, batch_idx):
        x = batch[0]  # (batch, dims[-1])
        batch_size = x.shape[0]

        # Encoder rates
        encoder_log_rate = self.model.encoder(x)  # (batch, dims[0])
        encoder_rate = encoder_log_rate.exp()
        encoder_upperbound = compute_com_poisson_upperbound(encoder_rate)
        z_exp_samples = -(
            1
            - torch.rand(
                (
                    batch_size,
                    self.model.dims[0],
                    encoder_upperbound,
                ),
                device=x.device,
            )
        ).log() / encoder_rate.unsqueeze(
            -1
        )  # (batch, dims[0], upperbound)
        z_samples = (
            torch.sigmoid((1 - torch.cumsum(z_exp_samples, dim=-1)) / self.hparams.tau)
        ).sum(
            dim=-1
        )  # (batch, dims[0])
        ln_qzgx = poisson_log_prob(z_samples, encoder_rate).sum(dim=-1)  # (batch,)

        # Prior rates
        prior_log_rate = self.model.prior_log_rate.unsqueeze(0).expand(batch_size, -1)
        ln_pz = poisson_log_prob(z_samples, prior_log_rate.exp()).sum(
            dim=-1
        )  # (batch,)

        # Decoder rates
        decoder_log_rate = self.model.decoder(z_samples)  # (batch, dims[-1])
        decoder_log_nu = self.model.decoder_log_nu.unsqueeze(0).expand(batch_size, -1)
        decoder_upperbound = max(
            compute_com_poisson_upperbound(decoder_log_rate.exp()),
            x.max().int().item() + 1,
        )
        decoder_logit = com_poisson_logit(
            rate=decoder_log_rate.exp(),
            nu=decoder_log_nu.exp(),
            upperbound=decoder_upperbound,
        )  # (batch, dims[-1], upperbound)
        decoder_cat = torch.distributions.Categorical(
            logits=decoder_logit, validate_args=False
        )
        ln_pxgz = decoder_cat.log_prob(x).sum(dim=-1)  # (batch,)

        elbo = ln_pz + ln_pxgz - ln_qzgx  # (batch,)
        loss = -elbo.mean()

        self.log_dict({"train/elbo": -loss.item()})
        return loss


class LitExpCubic(LitScore):
    def __init__(
        self,
        model: TwoLayerNetwork,
        tau: float = 0.2,
    ):
        super().__init__(
            model=model,
        )
        self.save_hyperparameters(ignore=["model"])

    def aggregate_samples(self, gumbel_samples: torch.Tensor):
        upperbound = gumbel_samples.shape[-1]
        k = torch.arange(upperbound, device=gumbel_samples.device).float()
        return gumbel_samples @ k

    def training_step(self, batch, batch_idx):
        x = batch[0]  # (batch, dims[-1])
        batch_size = x.shape[0]

        # Encoder rates
        encoder_log_rate = self.model.encoder(x)  # (batch, dims[0])
        encoder_rate = encoder_log_rate.exp()
        encoder_upperbound = compute_com_poisson_upperbound(encoder_rate)
        z_exp_samples = -(
            1
            - torch.rand(
                (
                    batch_size,
                    self.model.dims[0],
                    encoder_upperbound,
                ),
                device=x.device,
            )
        ).log() / encoder_rate.unsqueeze(
            -1
        )  # (batch, dims[0], upperbound)
        z_samples = (
            cubic_sigmoid((1 - torch.cumsum(z_exp_samples, dim=-1)) / self.hparams.tau)
        ).sum(
            dim=-1
        )  # (batch, dims[0])
        ln_qzgx = poisson_log_prob(z_samples, encoder_rate).sum(dim=-1)  # (batch,)

        # Prior rates
        prior_log_rate = self.model.prior_log_rate.unsqueeze(0).expand(batch_size, -1)
        ln_pz = poisson_log_prob(z_samples, prior_log_rate.exp()).sum(
            dim=-1
        )  # (batch,)

        # Decoder rates
        decoder_log_rate = self.model.decoder(z_samples)  # (batch, dims[-1])
        decoder_log_nu = self.model.decoder_log_nu.unsqueeze(0).expand(batch_size, -1)
        decoder_upperbound = max(
            compute_com_poisson_upperbound(decoder_log_rate.exp()),
            x.max().int().item() + 1,
        )
        decoder_logit = com_poisson_logit(
            rate=decoder_log_rate.exp(),
            nu=decoder_log_nu.exp(),
            upperbound=decoder_upperbound,
        )  # (batch, dims[-1], upperbound)
        decoder_cat = torch.distributions.Categorical(
            logits=decoder_logit, validate_args=False
        )
        ln_pxgz = decoder_cat.log_prob(x).sum(dim=-1)  # (batch,)

        elbo = ln_pz + ln_pxgz - ln_qzgx  # (batch,)
        loss = -elbo.mean()

        self.log_dict({"train/elbo": -loss.item()})
        return loss
