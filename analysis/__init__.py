import gc
from utils.generic import *
from main.distributions import Poisson, GumbelSoftmaxPoisson


DEFAULT_TEMPERATURES = [
	0.02, 0.05, 0.1, 0.15, 0.2,
	0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
]
DEFAULT_RATES = [
	0.1, 0.5, 1, 2, 5, 8, 10,
	15, 20, 30, 40, 70, 100,
]
COLORMAPS = {
	'EAT_sigmoid': 'C0',
	'EAT_cubic': 'C3',
	'EAT_quintic': 'C4',
	'Gumbel-Softmax': 'C2',
	'GS': 'C2',
	'Exact': 'k',
	'Score': 'C7',
	'OVIS': 'C4',
}
EPS = float(np.finfo(np.float32).eps)
N_EXP = 'infer'
N_EXP_P = 1e-3

def clear_gpu_memory():
    """Aggressively free GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return
