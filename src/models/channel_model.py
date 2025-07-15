import numpy as np
import torch
import gpytorch
from src.models.dp import StickBreakingDP
from src.models.gp import create_gp_model

class Path:
    def __init__(self, delay, gain_fn):
        self.delay = delay
        self.gain_fn = gain_fn

    def evaluate_gain(self, time_points):
        return self.gain_fn(time_points)


class DelayDopplerChannel:
    def __init__(self, alpha, base_delay_sampler, kernel_type='RBF'):
        self.alpha = alpha
        self.base_delay_sampler = base_delay_sampler
        self.kernel_type = kernel_type
        self.paths = []

    def sample_paths(self, time_points):
        self.paths = []
        fixed_delay = 2
        gain_fn = lambda t: np.ones_like(t)
        self.paths.append(Path(fixed_delay, gain_fn))

    def _build_kernel(self, time_points):
        if self.kernel_type == 'RBF':
            class RBFKernel(gpytorch.kernels.Kernel):
                def forward(self, x1, x2, diag=False, **params):
                    dist = torch.cdist(x1, x2)**2
                    return torch.exp(-0.5 * dist)
            return RBFKernel()
        raise NotImplementedError("Only RBF kernel supported currently.")

    def _sample_gp_function(self, time_points, kernel):
        x = torch.tensor(time_points).float().unsqueeze(1)
        mean = torch.zeros(len(time_points))
        cov = kernel(x, x).evaluate()
        L = torch.linalg.cholesky(cov + 1e-5 * torch.eye(len(time_points)))
        sample = mean + L @ torch.randn_like(mean)
        sample_np = sample.numpy()
        return lambda t: np.interp(t, time_points, sample_np)

    def evaluate(self, time_index):
        taps = []
        for path in self.paths:
            tap_gain = path.evaluate_gain(time_index)
            taps.append((path.delay, tap_gain))
        return taps


def sum_paths_at_time(channel_paths, t):
    return sum(p.evaluate_gain(t) for p in channel_paths)
