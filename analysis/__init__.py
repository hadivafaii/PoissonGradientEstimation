import gc
from utils.generic import *
from main.distributions import Poisson, GumbelSoftmaxPoisson


EPS = float(np.finfo(np.float32).eps)
DEFAULT_TEMPERATURES = [
	0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
]
DEFAULT_RATES = [
	0.1, 0.5, 1, 2, 5, 10, 20, 40, 70, 100,
]


def clear_gpu_memory():
    """Aggressively free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return
