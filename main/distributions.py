from base.utils_model import *

dists.Distribution.set_default_validate_args(False)


class Poisson:
    def __init__(
        self,
        log_rate: torch.Tensor,
        temp: float = 0.0,
        clamp: float | None = None,
        n_exp: int | str = "infer",
        n_exp_p: float = 1e-3,
    ):
        assert temp >= 0.0, f"must be non-neg: {temp}"
        self.temp = temp
        self.clamp = clamp
        # setup rate & exp dist
        if clamp is not None:
            log_rate = softclamp_upper(log_rate, clamp)
        eps = torch.finfo(torch.float32).eps
        self.rate = torch.exp(log_rate) + eps
        self._exp = dists.Exponential(self.rate)
        # compute n_exp
        if n_exp == "infer":
            max_rate = self.rate.max().item()
            n_exp = compute_n_exp(max_rate, n_exp_p)
        self.n_exp = int(n_exp)

    def __repr__(self):
        parts = [
            f"rate: {self.rate.shape}",
            f"temp: {self.temp}",
            f"n_exp: {self.n_exp}",
        ]
        return f"Poisson({', '.join(parts)})"

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate

    # noinspection PyTypeChecker
    def rsample(self):
        if self.temp == 0.0:
            return self.sample()
        x = self._exp.rsample((self.n_exp,))  # inter-event times
        times = torch.cumsum(x, dim=0)  # arrival t of events
        indicator = torch.sigmoid((1 - times) / self.temp)  # events within [0, 1]
        z = indicator.sum(0).float()  # event counts
        return z

    @torch.no_grad()
    def sample(self):
        return torch.poisson(self.rate).float()

    def log_prob(self, samples: torch.Tensor):
        return -self.rate - torch.lgamma(samples + 1) + samples * torch.log(self.rate)


class GumbelSoftmaxPoisson:
    def __init__(
        self,
        log_rate: torch.Tensor,
        temp: float = 1.0,
        upperbound_method: str = "fixed",
        upperbound_param: int | float = 5,
    ):
        assert temp >= 0.0, f"must be non-neg: {temp}"
        self.temp = temp
        self.log_rate = log_rate
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

    def __repr__(self):
        parts = [
            f"rate: {self.rate.shape}",
            f"temp: {self.temp}",
            f"upperbound: {self.upperbound}",
        ]
        return f"Poisson({', '.join(parts)})"

    @property
    def rate(self):
        return torch.exp(self.log_rate)

    @property
    def mean(self):
        return self.rate

    @property
    def variance(self):
        return self.rate

    @property
    def logit_pi(self):
        k = torch.arange(self.upperbound, device=self.rate.device)
        return k * self.log_rate.unsqueeze(-1) - torch.lgamma(k + 1)

    @torch.no_grad()
    def sample(self, n_samples: int | None = None):
        if n_samples is not None:
            rate = self.rate.unsqueeze(0).expand(n_samples, -1)
            return torch.poisson(rate).float()
        return torch.poisson(self.rate).float()

    def rsample(self, n_samples: int | None = None):
        if self.temp == 0.0:
            return self.sample(n_samples=n_samples)
        logit_pi = self.logit_pi
        if n_samples is not None:
            logit_pi = logit_pi.unsqueeze(0).expand(n_samples, -1, -1)
        z = F.gumbel_softmax(
            logits=logit_pi,
            tau=self.temp,
            hard=False,
        )  # (..., upperbound)
        return z

    def aggregate_samples(self, gumbel_samples: torch.Tensor):
        k = torch.arange(self.upperbound, device=self.rate.device).float()
        return gumbel_samples @ k

    def log_prob(self, samples: torch.Tensor, eps: float = 1e-8):
        ln_x = torch.clamp(samples, min=eps).log()
        logit_pi = self.logit_pi
        return (
            torch.lgamma(torch.tensor(self.upperbound))
            + (self.upperbound - 1) * torch.tensor(self.temp).log()
            - self.upperbound * torch.logsumexp(logit_pi - self.temp * ln_x, dim=-1)
            + (logit_pi - (self.temp + 1) * ln_x).sum(dim=-1)
        )


# Used to construct decoder
# noinspection PyAbstractClass
class Normal(dists.Normal):
    def __init__(
        self,
        loc: torch.Tensor,
        log_scale: torch.Tensor,
        temp: float = 1.0,
        clamp: float | None = None,
        **kwargs,
    ):
        if clamp is not None:
            log_scale = softclamp_sym(x=log_scale, c=clamp)
        scale = torch.exp(log_scale)

        if temp != 1.0:
            scale = scale * temp

        super().__init__(loc=loc, scale=scale, **kwargs)

        self.temp = temp
        self.clamp = clamp

    def kl(self, p: dists.Normal = None):
        if p is None:
            term1 = self.mean
            term2 = self.scale
        else:
            term1 = (self.mean - p.mean) / p.scale
            term2 = self.scale / p.scale
        kl = 0.5 * (term1.pow(2) + term2.pow(2) + torch.log(term2).mul(-2) - 1)
        return kl

    @torch.no_grad()
    def retrieve_noise(self, samples: torch.Tensor):
        return (samples - self.loc).div(self.scale)


def compute_n_exp(rate: float, p: float = 1e-6):
    assert rate > 0.0, f"must be positive, got: {rate}"
    pois = sp_stats.poisson(rate)
    n_exp = pois.ppf(1.0 - p)
    return int(n_exp)


def softclamp_upper(x: torch.Tensor, c: float):
    return c - F.softplus(c - x)


def softclamp_sym(x: torch.Tensor, c: float):
    return x.div(c).tanh_().mul(c)
