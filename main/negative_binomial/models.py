from itertools import permutations

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


def negative_binomial_log_prob(x, rate, r, eps=1e-8):
    p = r / (r + rate)
    return (
        torch.lgamma(x + r)
        - torch.lgamma(r)
        - torch.lgamma(x + 1)
        + r * torch.log(torch.clamp(p, min=eps))
        + x * torch.log(torch.clamp(1 - p, min=eps))
    )


def compute_negative_binomial_truncation(
    rate: float, r: float, p: float = 1e-6, min_val: int = 3
):
    assert rate > 0.0, f"must be positive, got: {rate}"
    assert r > 0.0, f"must be positive, got: {r}"
    p_nb = r / (r + rate)
    neg_binom = stats.nbinom(r, p_nb)
    truncation = neg_binom.ppf(1.0 - p)
    return max(int(truncation), min_val)


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
        self.prior_log_r = nn.Parameter(torch.zeros(dims[0]))
        self.decoder_log_r = nn.Parameter(torch.zeros(dims[-1]))
        self.encoder_log_r = nn.Parameter(torch.zeros(dims[0]))
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
        prior_log_r = self.prior_log_r.unsqueeze(0).expand(n_samples, -1)
        prior_logits = prior_log_rate - prior_log_r

        nb = torch.distributions.NegativeBinomial(
            logits=prior_logits, total_count=torch.exp(prior_log_r), validate_args=False
        )
        z_samples = nb.sample()  # (n_samples, dims[0])

        decoder_log_rate = self.decoder(z_samples)  # (n_samples, dims[-1])
        decoder_log_r = self.decoder_log_r.unsqueeze(0).expand(n_samples, -1)
        decoder_logits = decoder_log_rate - decoder_log_r
        nb = torch.distributions.NegativeBinomial(
            logits=decoder_logits,
            total_count=torch.exp(decoder_log_r),
            validate_args=False,
        )
        x_samples = nb.sample()  # (n_samples, dims[-1])

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
        encoder_log_r = self.model.encoder_log_r.unsqueeze(0).expand(batch_size, -1)
        encoder_logits = encoder_log_rate - encoder_log_r
        encoder_nb = torch.distributions.NegativeBinomial(
            logits=encoder_logits,
            total_count=torch.exp(encoder_log_r),
            validate_args=False,
        )
        z_samples = encoder_nb.sample()  # (batch, dims[0])
        ln_qzgx = encoder_nb.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Prior rates
        prior_log_rate = self.model.prior_log_rate.unsqueeze(0).expand(batch_size, -1)
        prior_log_r = self.model.prior_log_r.unsqueeze(0).expand(batch_size, -1)
        prior_logits = prior_log_rate - prior_log_r
        prior_nb = torch.distributions.NegativeBinomial(
            logits=prior_logits, total_count=torch.exp(prior_log_r), validate_args=False
        )
        ln_pz = prior_nb.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Decoder rates
        decoder_log_rate = self.model.decoder(z_samples)  # (batch, dims[-1])
        decoder_log_r = self.model.decoder_log_r.unsqueeze(0).expand(batch_size, -1)
        decoder_logits = decoder_log_rate - decoder_log_r
        decoder_nb = torch.distributions.NegativeBinomial(
            logits=decoder_logits,
            total_count=torch.exp(decoder_log_r),
            validate_args=False,
        )
        ln_pxgz = decoder_nb.log_prob(x).sum(dim=-1)  # (batch,)

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
        encoder_log_r = self.model.encoder_log_r.unsqueeze(0).expand(batch_size, -1)
        encoder_logits = encoder_log_rate - encoder_log_r
        encoder_nb = torch.distributions.NegativeBinomial(
            logits=encoder_logits,
            total_count=torch.exp(encoder_log_r),
            validate_args=False,
        )
        z_samples = encoder_nb.sample()  # (batch, dims[0])
        ln_qzgx = encoder_nb.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Prior rates
        prior_log_rate = self.model.prior_log_rate.unsqueeze(0).expand(batch_size, -1)
        prior_log_r = self.model.prior_log_r.unsqueeze(0).expand(batch_size, -1)
        prior_logits = prior_log_rate - prior_log_r
        prior_nb = torch.distributions.NegativeBinomial(
            logits=prior_logits, total_count=torch.exp(prior_log_r), validate_args=False
        )
        ln_pz = prior_nb.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Decoder rates
        decoder_log_rate = self.model.decoder(z_samples)  # (batch, dims[-1])
        decoder_log_r = self.model.decoder_log_r.unsqueeze(0).expand(batch_size, -1)
        decoder_logits = decoder_log_rate - decoder_log_r
        decoder_nb = torch.distributions.NegativeBinomial(
            logits=decoder_logits,
            total_count=torch.exp(decoder_log_r),
            validate_args=False,
        )
        ln_pxgz = decoder_nb.log_prob(x).sum(dim=-1)  # (batch,)

        ln_p = ln_pz + ln_pxgz  # (batch,)
        ln_q = ln_qzgx  # (batch,)

        ln_p_values = ln_p.detach()
        ln_q_values = ln_q.detach()
        elbo_values = ln_p_values - ln_q_values
        elbo = (
            ln_p - ln_p_values + elbo_values * (ln_q - ln_q_values) + elbo_values
        )  # (batch,)

        conditional_log_likelihood = encoder_nb.log_prob(z).sum(dim=-1)  # (batch,)

        decoder_log_rate = self.model.decoder(z)  # (batch, dims[-1])
        decoder_log_r = self.model.decoder_log_r.unsqueeze(0).expand(batch_size, -1)
        decoder_logits = decoder_log_rate - decoder_log_r
        decoder_nb = torch.distributions.NegativeBinomial(
            logits=decoder_logits,
            total_count=torch.exp(decoder_log_r),
            validate_args=False,
        )
        hidden_log_likelihood = decoder_nb.log_prob(x).sum(dim=-1)  # (batch,)

        self.log_dict(
            {
                "val/elbo": elbo.mean().item(),
                "val/cll": conditional_log_likelihood.mean().item(),
                "val/hll": hidden_log_likelihood.mean().item(),
            }
        )

    def on_test_epoch_start(self):
        self.df_metrics = pd.DataFrame(columns=["elbo", "cll", "hll"])

    def test_step(self, batch, batch_idx):
        x = batch[0]  # (batch, dims[-1])
        z = batch[1]  # (batch, dims[0])
        batch_size = x.shape[0]

        # Encoder rates
        encoder_log_rate = self.model.encoder(x)  # (batch, dims[0])
        encoder_log_r = self.model.encoder_log_r.unsqueeze(0).expand(batch_size, -1)
        encoder_logits = encoder_log_rate - encoder_log_r
        encoder_nb = torch.distributions.NegativeBinomial(
            logits=encoder_logits,
            total_count=torch.exp(encoder_log_r),
            validate_args=False,
        )
        z_samples = encoder_nb.sample()  # (batch, dims[0])
        ln_qzgx = encoder_nb.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Prior rates
        prior_log_rate = self.model.prior_log_rate.unsqueeze(0).expand(batch_size, -1)
        prior_log_r = self.model.prior_log_r.unsqueeze(0).expand(batch_size, -1)
        prior_logits = prior_log_rate - prior_log_r
        prior_nb = torch.distributions.NegativeBinomial(
            logits=prior_logits, total_count=torch.exp(prior_log_r), validate_args=False
        )
        ln_pz = prior_nb.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Decoder rates
        decoder_log_rate = self.model.decoder(z_samples)  # (batch, dims[-1])
        decoder_log_r = self.model.decoder_log_r.unsqueeze(0).expand(batch_size, -1)
        decoder_logits = decoder_log_rate - decoder_log_r
        decoder_nb = torch.distributions.NegativeBinomial(
            logits=decoder_logits,
            total_count=torch.exp(decoder_log_r),
            validate_args=False,
        )
        ln_pxgz = decoder_nb.log_prob(x).sum(dim=-1)  # (batch,)

        ln_p = ln_pz + ln_pxgz  # (batch,)
        ln_q = ln_qzgx  # (batch,)

        ln_p_values = ln_p.detach()
        ln_q_values = ln_q.detach()
        elbo_values = ln_p_values - ln_q_values
        elbo = (
            ln_p - ln_p_values + elbo_values * (ln_q - ln_q_values) + elbo_values
        )  # (batch,)

        conditional_log_likelihood = encoder_nb.log_prob(z).sum(dim=-1)  # (batch,)

        decoder_log_rate = self.model.decoder(z)  # (batch, dims[-1])
        decoder_log_r = self.model.decoder_log_r.unsqueeze(0).expand(batch_size, -1)
        decoder_logits = decoder_log_rate - decoder_log_r
        decoder_nb = torch.distributions.NegativeBinomial(
            logits=decoder_logits,
            total_count=torch.exp(decoder_log_r),
            validate_args=False,
        )
        hidden_log_likelihood = decoder_nb.log_prob(x).sum(dim=-1)  # (batch,)

        self.df_metrics.at[0, "elbo"] = elbo.mean().item()
        self.df_metrics.at[0, "cll"] = conditional_log_likelihood.mean().item()
        self.df_metrics.at[0, "hll"] = hidden_log_likelihood.mean().item()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        return optimizer


class LitGS2(LitScore):
    def __init__(
        self,
        model: TwoLayerNetwork,
        tau: float = 0.2,
        upperbound_method: str = "fixed",
        upperbound_param: int = 8,
    ):
        super().__init__(
            model=model,
        )
        self.upperbound_method = upperbound_method
        self.upperbound_param = upperbound_param
        if self.upperbound_method == "fixed":
            self.upperbound = int(self.upperbound_param)
        # elif self.upperbound_method == "quantile":
        #     self.upperbound_percentile = self.upperbound_param
        else:
            raise ValueError(f"unknown upperbound_method: {self.upperbound_method}")
        self.save_hyperparameters(ignore=["model"])

    def logit_pi(self, rate):
        k = torch.arange(self.upperbound, device=rate.device).float()
        return k * rate.log().unsqueeze(-1) - torch.lgamma(k + 1)

    def aggregate_samples(self, gumbel_samples: torch.Tensor):
        k = torch.arange(self.upperbound, device=gumbel_samples.device).float()
        return gumbel_samples @ k

    def training_step(self, batch, batch_idx):
        x = batch[0]  # (batch, dims[-1])
        batch_size = x.shape[0]

        # Encoder rates
        encoder_log_rate = self.model.encoder(x)  # (batch, dims[0])
        encoder_log_r = self.model.encoder_log_r.unsqueeze(0).expand(batch_size, -1)
        encoder_logits = encoder_log_rate - encoder_log_r
        encoder_rate = encoder_log_rate.exp()
        encoder_r = encoder_log_r.exp()
        encoder_gamma = torch.distributions.Gamma(encoder_r, encoder_r / encoder_rate)
        lam_samples = encoder_gamma.rsample()  # (batch, dims[0])
        z_gs_samples = F.gumbel_softmax(
            self.logit_pi(lam_samples),
            tau=self.hparams.tau,
            hard=False,
            dim=-1,
        )  # (batch, dims[0], upperbound)
        z_samples = self.aggregate_samples(z_gs_samples)  # (batch, dims[0])
        ln_qzgx = (
            torch.distributions.NegativeBinomial(
                logits=encoder_logits, total_count=encoder_r, validate_args=False
            )
            .log_prob(z_samples)
            .sum(dim=-1)
        )  # (batch,)

        # Prior rates
        prior_log_rate = self.model.prior_log_rate.unsqueeze(0).expand(batch_size, -1)
        prior_log_r = self.model.prior_log_r.unsqueeze(0).expand(batch_size, -1)
        prior_logits = prior_log_rate - prior_log_r
        prior_nb = torch.distributions.NegativeBinomial(
            logits=prior_logits, total_count=prior_log_r.exp(), validate_args=False
        )
        ln_pz = prior_nb.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Decoder rates
        decoder_log_rate = self.model.decoder(z_samples)  # (batch, dims[-1])
        decoder_log_r = self.model.decoder_log_r.unsqueeze(0).expand(batch_size, -1)
        decoder_logits = decoder_log_rate - decoder_log_r
        decoder_nb = torch.distributions.NegativeBinomial(
            logits=decoder_logits, total_count=decoder_log_r.exp(), validate_args=False
        )
        ln_pxgz = decoder_nb.log_prob(x).sum(dim=-1)  # (batch,)

        elbo = ln_pz + ln_pxgz - ln_qzgx  # (batch,)
        loss = -elbo.mean()

        self.log_dict({"train/elbo": -loss.item()})
        return loss


class LitGS(LitScore):
    def __init__(
        self,
        model: TwoLayerNetwork,
        tau: float = 0.2,
        upperbound_method: str = "fixed",
        upperbound_param: int = 8,
    ):
        super().__init__(
            model=model,
        )
        self.upperbound_method = upperbound_method
        self.upperbound_param = upperbound_param
        if self.upperbound_method == "fixed":
            self.upperbound = int(self.upperbound_param)
        # elif self.upperbound_method == "quantile":
        #     self.upperbound_percentile = self.upperbound_param
        else:
            raise ValueError(f"unknown upperbound_method: {self.upperbound_method}")
        self.save_hyperparameters(ignore=["model"])

    def aggregate_samples(self, gumbel_samples: torch.Tensor):
        k = torch.arange(self.upperbound, device=gumbel_samples.device).float()
        return gumbel_samples @ k

    def training_step(self, batch, batch_idx):
        x = batch[0]  # (batch, dims[-1])
        batch_size = x.shape[0]

        # Encoder rates
        encoder_log_rate = self.model.encoder(x)  # (batch, dims[0])
        encoder_log_r = self.model.encoder_log_r.unsqueeze(0).expand(batch_size, -1)
        encoder_logits = encoder_log_rate - encoder_log_r
        encoder_nb = torch.distributions.NegativeBinomial(
            logits=encoder_logits, total_count=encoder_log_r.exp(), validate_args=False
        )
        encoder_logit_pi = encoder_nb.log_prob(
            torch.arange(self.upperbound, device=x.device).float()[:, None, None],
        ).permute(
            1, 2, 0
        )  # (batch, dims[0], upperbound)
        z_gs_samples = F.gumbel_softmax(
            encoder_logit_pi,
            tau=self.hparams.tau,
            hard=False,
            dim=-1,
        )  # (batch, dims[0], upperbound)
        z_samples = self.aggregate_samples(z_gs_samples)  # (batch, dims[0])
        ln_qzgx = encoder_nb.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Prior rates
        prior_log_rate = self.model.prior_log_rate.unsqueeze(0).expand(batch_size, -1)
        prior_log_r = self.model.prior_log_r.unsqueeze(0).expand(batch_size, -1)
        prior_logits = prior_log_rate - prior_log_r
        prior_nb = torch.distributions.NegativeBinomial(
            logits=prior_logits, total_count=prior_log_r.exp(), validate_args=False
        )
        ln_pz = prior_nb.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Decoder rates
        decoder_log_rate = self.model.decoder(z_samples)  # (batch, dims[-1])
        decoder_log_r = self.model.decoder_log_r.unsqueeze(0).expand(batch_size, -1)
        decoder_logits = decoder_log_rate - decoder_log_r
        decoder_nb = torch.distributions.NegativeBinomial(
            logits=decoder_logits, total_count=decoder_log_r.exp(), validate_args=False
        )
        ln_pxgz = decoder_nb.log_prob(x).sum(dim=-1)  # (batch,)

        elbo = ln_pz + ln_pxgz - ln_qzgx  # (batch,)
        loss = -elbo.mean()

        self.log_dict({"train/elbo": -loss.item()})
        return loss


class LitExpSigmoid(LitScore):
    def __init__(
        self,
        model: TwoLayerNetwork,
        tau: float = 0.2,
        upperbound_method: str = "fixed",
        upperbound_param: int = 8,
    ):
        super().__init__(
            model=model,
        )
        self.upperbound_method = upperbound_method
        self.upperbound_param = upperbound_param
        if self.upperbound_method == "fixed":
            self.upperbound = int(self.upperbound_param)
        # elif self.upperbound_method == "quantile":
        #     self.upperbound_percentile = self.upperbound_param
        else:
            raise ValueError(f"unknown upperbound_method: {self.upperbound_method}")
        self.save_hyperparameters(ignore=["model"])

    def logit_pi(self, rate):
        k = torch.arange(self.upperbound, device=rate.device).float()
        return k * rate.log().unsqueeze(-1) - torch.lgamma(k + 1)

    def aggregate_samples(self, gumbel_samples: torch.Tensor):
        k = torch.arange(self.upperbound, device=gumbel_samples.device).float()
        return gumbel_samples @ k

    def training_step(self, batch, batch_idx):
        x = batch[0]  # (batch, dims[-1])
        batch_size = x.shape[0]

        # Encoder rates
        encoder_log_rate = self.model.encoder(x)  # (batch, dims[0])
        encoder_log_r = self.model.encoder_log_r.unsqueeze(0).expand(batch_size, -1)
        encoder_logits = encoder_log_rate - encoder_log_r
        encoder_rate = encoder_log_rate.exp()
        encoder_r = encoder_log_r.exp()
        encoder_gamma = torch.distributions.Gamma(encoder_r, encoder_r / encoder_rate)
        lam_samples = encoder_gamma.rsample()  # (batch, dims[0])
        z_exp_samples = -(
            1
            - torch.rand(
                (
                    batch_size,
                    self.model.dims[0],
                    self.upperbound,
                ),
                device=x.device,
            )
        ).log() / lam_samples.unsqueeze(
            -1
        )  # (batch, dims[0], upperbound)
        z_samples = (
            torch.sigmoid((1 - torch.cumsum(z_exp_samples, dim=-1)) / self.hparams.tau)
        ).sum(
            dim=-1
        )  # (batch, dims[0])
        ln_qzgx = (
            torch.distributions.NegativeBinomial(
                logits=encoder_logits, total_count=encoder_r, validate_args=False
            )
            .log_prob(z_samples)
            .sum(dim=-1)
        )  # (batch,)

        # Prior rates
        prior_log_rate = self.model.prior_log_rate.unsqueeze(0).expand(batch_size, -1)
        prior_log_r = self.model.prior_log_r.unsqueeze(0).expand(batch_size, -1)
        prior_logits = prior_log_rate - prior_log_r
        prior_nb = torch.distributions.NegativeBinomial(
            logits=prior_logits, total_count=prior_log_r.exp(), validate_args=False
        )
        ln_pz = prior_nb.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Decoder rates
        decoder_log_rate = self.model.decoder(z_samples)  # (batch, dims[-1])
        decoder_log_r = self.model.decoder_log_r.unsqueeze(0).expand(batch_size, -1)
        decoder_logits = decoder_log_rate - decoder_log_r
        decoder_nb = torch.distributions.NegativeBinomial(
            logits=decoder_logits, total_count=decoder_log_r.exp(), validate_args=False
        )
        ln_pxgz = decoder_nb.log_prob(x).sum(dim=-1)  # (batch,)

        elbo = ln_pz + ln_pxgz - ln_qzgx  # (batch,)
        loss = -elbo.mean()

        self.log_dict({"train/elbo": -loss.item()})
        return loss


class LitExpCubic(LitScore):
    def __init__(
        self,
        model: TwoLayerNetwork,
        tau: float = 0.2,
        upperbound_method: str = "fixed",
        upperbound_param: int = 8,
    ):
        super().__init__(
            model=model,
        )
        self.upperbound_method = upperbound_method
        self.upperbound_param = upperbound_param
        if self.upperbound_method == "fixed":
            self.upperbound = int(self.upperbound_param)
        # elif self.upperbound_method == "quantile":
        #     self.upperbound_percentile = self.upperbound_param
        else:
            raise ValueError(f"unknown upperbound_method: {self.upperbound_method}")
        self.save_hyperparameters(ignore=["model"])

    def logit_pi(self, rate):
        k = torch.arange(self.upperbound, device=rate.device).float()
        return k * rate.log().unsqueeze(-1) - torch.lgamma(k + 1)

    def aggregate_samples(self, gumbel_samples: torch.Tensor):
        k = torch.arange(self.upperbound, device=gumbel_samples.device).float()
        return gumbel_samples @ k

    def training_step(self, batch, batch_idx):
        x = batch[0]  # (batch, dims[-1])
        batch_size = x.shape[0]

        # Encoder rates
        encoder_log_rate = self.model.encoder(x)  # (batch, dims[0])
        encoder_log_r = self.model.encoder_log_r.unsqueeze(0).expand(batch_size, -1)
        encoder_logits = encoder_log_rate - encoder_log_r
        encoder_rate = encoder_log_rate.exp()
        encoder_r = encoder_log_r.exp()
        encoder_gamma = torch.distributions.Gamma(encoder_r, encoder_r / encoder_rate)
        lam_samples = encoder_gamma.rsample()  # (batch, dims[0])
        z_exp_samples = -(
            1
            - torch.rand(
                (
                    batch_size,
                    self.model.dims[0],
                    self.upperbound,
                ),
                device=x.device,
            )
        ).log() / lam_samples.unsqueeze(
            -1
        )  # (batch, dims[0], upperbound)
        z_samples = (
            cubic_sigmoid((1 - torch.cumsum(z_exp_samples, dim=-1)) / self.hparams.tau)
        ).sum(
            dim=-1
        )  # (batch, dims[0])
        ln_qzgx = (
            torch.distributions.NegativeBinomial(
                logits=encoder_logits, total_count=encoder_r, validate_args=False
            )
            .log_prob(z_samples)
            .sum(dim=-1)
        )  # (batch,)

        # Prior rates
        prior_log_rate = self.model.prior_log_rate.unsqueeze(0).expand(batch_size, -1)
        prior_log_r = self.model.prior_log_r.unsqueeze(0).expand(batch_size, -1)
        prior_logits = prior_log_rate - prior_log_r
        prior_nb = torch.distributions.NegativeBinomial(
            logits=prior_logits, total_count=prior_log_r.exp(), validate_args=False
        )
        ln_pz = prior_nb.log_prob(z_samples).sum(dim=-1)  # (batch,)

        # Decoder rates
        decoder_log_rate = self.model.decoder(z_samples)  # (batch, dims[-1])
        decoder_log_r = self.model.decoder_log_r.unsqueeze(0).expand(batch_size, -1)
        decoder_logits = decoder_log_rate - decoder_log_r
        decoder_nb = torch.distributions.NegativeBinomial(
            logits=decoder_logits, total_count=decoder_log_r.exp(), validate_args=False
        )
        ln_pxgz = decoder_nb.log_prob(x).sum(dim=-1)  # (batch,)

        elbo = ln_pz + ln_pxgz - ln_qzgx  # (batch,)
        loss = -elbo.mean()

        self.log_dict({"train/elbo": -loss.item()})
        return loss
