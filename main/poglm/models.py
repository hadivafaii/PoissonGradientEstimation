from itertools import permutations

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


def poisson_log_prob(x, rate, eps=1e-8):
    return -rate + x * torch.log(torch.clamp(rate, min=eps)) - torch.lgamma(x + 1)


def gumbel_softmax_log_prob(samples, logit_pi, temp, eps=1e-8):
    ln_x = torch.clamp(samples, min=eps).log()
    upperbound = logit_pi.shape[-1]
    return (
        torch.lgamma(torch.tensor(upperbound))
        + (upperbound - 1) * torch.tensor(temp).log()
        - upperbound * torch.logsumexp(logit_pi - temp * ln_x, dim=-1)
        + (logit_pi - (temp + 1) * ln_x).sum(dim=-1)
    )


def compute_n_exp(rate: float, p: float = 1e-6):
    assert rate > 0.0, f"must be positive, got: {rate}"
    pois = stats.poisson(rate)
    n_exp = pois.ppf(1.0 - p)
    return int(n_exp)


@torch.inference_mode()
def match_hidden_neurons_according_conv_weights(
    true_weights: torch.Tensor,
    learned_weights: torch.Tensor,
    n_vis_neurons: int,
):
    n_neurons = true_weights.shape[0]
    n_hid_neurons = n_neurons - n_vis_neurons

    all_possible_permutations = torch.tensor(list(permutations(range(n_hid_neurons))))
    n_possible_permutations = all_possible_permutations.shape[0]
    all_possible_permutations = torch.cat(
        (
            torch.arange(n_vis_neurons)
            .unsqueeze(0)
            .expand(n_possible_permutations, -1),
            all_possible_permutations + n_vis_neurons,
        ),
        dim=1,
    )

    error_list = torch.zeros(n_possible_permutations)
    for permutation in range(n_possible_permutations):
        permuted_learned_weights = learned_weights[
            all_possible_permutations[permutation], :, :
        ][:, all_possible_permutations[permutation], :]
        error_list[permutation] = (true_weights - permuted_learned_weights).abs().sum()
    true_to_learned = all_possible_permutations[error_list.argmin()]
    return true_to_learned


class POGLM(nn.Module):
    def __init__(
        self,
        n_vis_neurons: int,
        n_hid_neurons: int,
        kernel_size: int,
        max_rate: float = 10,
    ):
        super().__init__()
        self.n_vis_neurons = n_vis_neurons
        self.n_hid_neurons = n_hid_neurons
        self.n_neurons = n_vis_neurons + n_hid_neurons
        self.kernel_size = kernel_size
        self.max_rate = max_rate

        self.conv_generative = nn.Conv1d(
            in_channels=self.n_neurons,
            out_channels=self.n_neurons,
            kernel_size=kernel_size,
            bias=True,
        )

        self.conv_variational = nn.Conv1d(
            in_channels=self.n_vis_neurons,
            out_channels=self.n_hid_neurons,
            kernel_size=2 * kernel_size + 1,
            padding=kernel_size,
            bias=True,
        )

    @torch.inference_mode()
    def permute_hidden_neurons(self, true_to_learned):
        self.conv_generative.weight.data[:, :, :] = self.conv_generative.weight.data[
            true_to_learned, :, :
        ][:, true_to_learned, :]
        self.conv_generative.bias.data[:] = self.conv_generative.bias.data[
            true_to_learned
        ]
        true_to_learned = true_to_learned[self.n_vis_neurons :] - self.n_vis_neurons
        self.conv_variational.weight.data[:, :, :] = self.conv_variational.weight.data[
            true_to_learned, :, :
        ]
        self.conv_variational.bias.data[:] = self.conv_variational.bias.data[
            true_to_learned
        ]

    def generative(self, y):
        #  y: (batch, time, n_neurons)
        y = y.permute(0, 2, 1)  # (batch, n_neurons, time)
        conv_out = self.conv_generative(y)
        rate = F.softplus(conv_out)
        rate = torch.clamp(rate, max=self.max_rate)
        rate = rate.permute(0, 2, 1)  # (batch, time, n_neurons)
        return rate

    def variational(self, x):
        #  x: (batch, time, n_vis_neurons)
        x = x.permute(0, 2, 1)  # (batch, n_vis_neurons, time)
        conv_out = self.conv_variational(x)
        rate = F.softplus(conv_out)
        rate = torch.clamp(rate, max=self.max_rate)
        rate = rate.permute(0, 2, 1)  # (batch, time, n_hid_neurons)
        return rate

    @torch.inference_mode()
    def sample(self, n_time_bins: int, n_samples: int = 1):
        rate = torch.zeros(n_samples, n_time_bins, self.n_neurons)
        y = torch.zeros(n_samples, n_time_bins + self.kernel_size, self.n_neurons)
        for t in range(n_time_bins):
            rate[:, t, :] = self.generative(y[:, t : t + self.kernel_size, :])[:, 0, :]
            y[:, t + self.kernel_size, :] = torch.poisson(rate[:, t, :])
        return rate, y[:, self.kernel_size :, :]


class LitPOGLM(L.LightningModule):
    def __init__(
        self,
        poglm: POGLM,
        n_monte_carlo: int = 1,
        true_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.poglm = poglm
        self.save_hyperparameters(ignore=["poglm"])

    def training_step(self, batch, batch_idx):
        y = batch[0]  # (batch, n_time_bins, n_vis_neurons)
        x = y[:, :, : self.poglm.n_vis_neurons]  # (batch, n_time_bins, n_vis_neurons)
        batch_size, n_time_bins, _ = y.shape

        # Variational rates
        q_rate = self.poglm.variational(x)  # (batch, n_time_bins, n_hid_neurons)

        # Monte Carlo samples from variational distribution
        z_samples = torch.poisson(
            q_rate.unsqueeze(1).expand(
                batch_size, self.hparams.n_monte_carlo, n_time_bins, -1
            )
        )  # (batch, n_mc, n_time_bins, n_hid_neurons)

        # Combine visible and hidden neurons
        y_samples = torch.cat(
            [
                x.unsqueeze(1).expand(-1, self.hparams.n_monte_carlo, -1, -1),
                z_samples,
            ],
            dim=-1,
        )  # (batch, n_mc, n_time_bins, n_neurons)

        # Generative rates
        p_rate = self.poglm.generative(
            F.pad(
                y_samples.reshape(-1, n_time_bins, self.poglm.n_neurons),
                (0, 0, self.poglm.kernel_size, -1),
            )
        ).reshape(
            batch_size, self.hparams.n_monte_carlo, n_time_bins, self.poglm.n_neurons
        )  # (batch, n_mc, n_time_bins, n_neurons)

        # Compute log likelihood
        ln_p = poisson_log_prob(y_samples, p_rate).mean(dim=[-2, -1])  # (batch, n_mc)
        ln_q = poisson_log_prob(
            z_samples,
            q_rate.unsqueeze(1).expand(
                batch_size, self.hparams.n_monte_carlo, n_time_bins, -1
            ),
        ).mean(
            dim=[-2, -1]
        )  # (batch, n_mc)

        ln_p_values = ln_p.detach()
        ln_q_values = ln_q.detach()
        elbo_values = ln_p_values - ln_q_values
        elbo = (
            ln_p - ln_p_values + elbo_values * (ln_q - ln_q_values) + elbo_values
        ).mean(
            dim=-1
        )  # (batch,)
        loss = -elbo.mean()

        self.log_dict({"train/elbo": -loss.item()})
        return loss

    def on_train_epoch_end(self):
        if self.hparams.true_weights is not None:
            true_to_learned = match_hidden_neurons_according_conv_weights(
                true_weights=self.hparams.true_weights,
                learned_weights=self.poglm.conv_generative.weight.data,
                n_vis_neurons=self.poglm.n_vis_neurons,
            )
            self.poglm.permute_hidden_neurons(true_to_learned=true_to_learned)
            self.log_dict(
                {
                    "train/weights_error": (
                        self.hparams.true_weights
                        - self.poglm.conv_generative.weight.data
                    )
                    .abs()
                    .mean()
                    .item()
                }
            )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.poglm.parameters(), lr=1e-3)
        return optimizer


class LitGSPOGLM(LitPOGLM):
    def __init__(
        self,
        poglm: POGLM,
        n_monte_carlo: int = 1,
        true_weights: torch.Tensor | None = None,
        temp: float = 0.2,
        upperbound_method: str = "fixed",
        upperbound_param: int = 8,
    ):
        super().__init__(
            poglm=poglm,
            n_monte_carlo=n_monte_carlo,
            true_weights=true_weights,
        )
        self.temp = temp
        self.upperbound_method = upperbound_method
        self.upperbound_param = upperbound_param
        if self.upperbound_method == "fixed":
            self.upperbound = int(self.upperbound_param)
        elif self.upperbound_method == "std_ratio":
            self.upperbound = int(
                self.rate.detach().sqrt().cpu().numpy().max() * self.upperbound_param
            )
        elif self.upperbound_method == "quantile":
            self.upperbound = compute_n_exp(
                rate=self.rate.detach().cpu().numpy().max(),
                p=self.upperbound_param,
            )
        else:
            raise ValueError(f"unknown upperbound_method: {self.upperbound_method}")
        self.save_hyperparameters(ignore=["poglm"])

    def logit_pi(self, rate):
        k = torch.arange(self.upperbound, device=rate.device)
        return k * rate.log().unsqueeze(-1) - torch.lgamma(k + 1)

    def aggregate_samples(self, gumbel_samples: torch.Tensor):
        k = torch.arange(self.upperbound, device=gumbel_samples.device).float()
        return gumbel_samples @ k

    # def training_step(self, batch, batch_idx):
    #     y = batch[0]  # (batch, n_time_bins, n_vis_neurons)
    #     x = y[:, :, : self.poglm.n_vis_neurons]  # (batch, n_time_bins, n_vis_neurons)
    #     batch_size, n_time_bins, _ = y.shape

    #     # Variational rates
    #     q_rate = self.poglm.variational(x)  # (batch, n_time_bins, n_hid_neurons)

    #     # Monte Carlo samples from variational distribution
    #     q_logit_pi = self.logit_pi(
    #         q_rate
    #     )  # (batch, n_time_bins, n_hid_neurons, upperbound)
    #     z_gs_samples = F.gumbel_softmax(
    #         q_logit_pi.unsqueeze(1).expand(
    #             batch_size,
    #             self.hparams.n_monte_carlo,
    #             n_time_bins,
    #             self.poglm.n_hid_neurons,
    #             self.upperbound,
    #         ),
    #         tau=self.temp,
    #         hard=False,
    #     )  # (batch, n_mc, n_time_bins, n_hid_neurons, upperbound)
    #     z_samples = self.aggregate_samples(
    #         z_gs_samples
    #     )  # (batch, n_mc, n_time_bins, n_hid_neurons

    #     # Combine visible and hidden neurons
    #     y_samples = torch.cat(
    #         [
    #             x.unsqueeze(1).expand(-1, self.hparams.n_monte_carlo, -1, -1),
    #             z_samples,
    #         ],
    #         dim=-1,
    #     )  # (batch, n_mc, n_time_bins, n_neurons)

    #     # Generative rates
    #     p_rate = self.poglm.generative(
    #         F.pad(
    #             y_samples.reshape(-1, n_time_bins, self.poglm.n_neurons),
    #             (0, 0, self.poglm.kernel_size, -1),
    #         )
    #     ).reshape(
    #         batch_size, self.hparams.n_monte_carlo, n_time_bins, self.poglm.n_neurons
    #     )  # (batch, n_mc, n_time_bins, n_neurons)
    #     p_logit_pi = self.logit_pi(
    #         p_rate[:, :, :, self.poglm.n_vis_neurons :]
    #     )  # (batch, n_mc, n_time_bins, n_hid_neurons, upperbound)

    #     # Compute log likelihood
    #     ln_p = poisson_log_prob(
    #         y_samples[:, :, :, : self.poglm.n_vis_neurons],
    #         p_rate[:, :, :, : self.poglm.n_vis_neurons],
    #     ).mean(dim=[-2, -1]) + gumbel_softmax_log_prob(
    #         z_gs_samples,
    #         p_logit_pi,
    #         temp=self.temp,
    #     ).mean(
    #         dim=[-2, -1]
    #     )  # (batch, n_mc)
    #     ln_q = gumbel_softmax_log_prob(
    #         z_gs_samples,
    #         q_logit_pi.unsqueeze(1).expand(
    #             batch_size,
    #             self.hparams.n_monte_carlo,
    #             n_time_bins,
    #             self.poglm.n_hid_neurons,
    #             self.upperbound,
    #         ),
    #         temp=self.temp,
    #     ).mean(
    #         dim=[-2, -1]
    #     )  # (batch, n_mc)

    #     elbo = (ln_p - ln_q).mean(dim=-1)  # (batch,)
    #     loss = -elbo.mean()

    #     self.log_dict({"train/elbo": -loss.item()})
    #     return loss

    def training_step(self, batch, batch_idx):
        y = batch[0]  # (batch, n_time_bins, n_vis_neurons)
        x = y[:, :, : self.poglm.n_vis_neurons]  # (batch, n_time_bins, n_vis_neurons)
        batch_size, n_time_bins, _ = y.shape

        # Variational rates
        q_rate = self.poglm.variational(x)  # (batch, n_time_bins, n_hid_neurons)

        # Monte Carlo samples from variational distribution
        q_logit_pi = self.logit_pi(
            q_rate
        )  # (batch, n_time_bins, n_hid_neurons, upperbound)
        z_gs_samples = F.gumbel_softmax(
            q_logit_pi.unsqueeze(1).expand(
                batch_size,
                self.hparams.n_monte_carlo,
                n_time_bins,
                self.poglm.n_hid_neurons,
                self.upperbound,
            ),
            tau=self.temp,
            hard=True,
        )  # (batch, n_mc, n_time_bins, n_hid_neurons, upperbound)
        z_samples = self.aggregate_samples(
            z_gs_samples
        )  # (batch, n_mc, n_time_bins, n_hid_neurons

        # Combine visible and hidden neurons
        y_samples = torch.cat(
            [
                x.unsqueeze(1).expand(-1, self.hparams.n_monte_carlo, -1, -1),
                z_samples,
            ],
            dim=-1,
        )  # (batch, n_mc, n_time_bins, n_neurons)

        # Generative rates
        p_rate = self.poglm.generative(
            F.pad(
                y_samples.reshape(-1, n_time_bins, self.poglm.n_neurons),
                (0, 0, self.poglm.kernel_size, -1),
            )
        ).reshape(
            batch_size, self.hparams.n_monte_carlo, n_time_bins, self.poglm.n_neurons
        )  # (batch, n_mc, n_time_bins, n_neurons)

        # Compute log likelihood
        ln_p = poisson_log_prob(
            y_samples,
            p_rate,
        ).mean(
            dim=[-2, -1]
        )  # (batch, n_mc)
        ln_q = poisson_log_prob(
            z_samples,
            q_rate.unsqueeze(1).expand(
                batch_size,
                self.hparams.n_monte_carlo,
                n_time_bins,
                self.poglm.n_hid_neurons,
            ),
        ).mean(
            dim=[-2, -1]
        )  # (batch, n_mc)

        ln_p_values = ln_p.detach()
        ln_q_values = ln_q.detach()
        elbo_values = ln_p_values - ln_q_values
        elbo = (
            ln_p - ln_p_values + elbo_values * (ln_q - ln_q_values) + elbo_values
        ).mean(
            dim=-1
        )  # (batch,)
        loss = -elbo.mean()
        self.log_dict({"train/elbo": -loss.item()})
        return loss
