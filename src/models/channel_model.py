import numpy as np
import torch
import gpytorch
import copy
from src.models.dp import StickBreakingDP
from src.models.gp import create_gp_model

class Path:
    """Represents a single path with a delay and a time-varying gain function."""
    def __init__(self, delay, gain_fn):
        self.delay = delay
        self.gain_fn = gain_fn

    def evaluate_gain(self, time_points):
        return self.gain_fn(time_points)

class DelayDopplerChannel:
    """
    The full Bayesian Nonparametric channel model using DP for path delays
    and GPs for time-varying gains.
    """
    def __init__(self, alpha: float = 1.0, base_delay_sampler=None, kernel_type: str = 'RBF'):
        self.alpha = alpha
        self.base_delay_sampler = base_delay_sampler
        self.kernel_type = kernel_type
        self.dp = StickBreakingDP(alpha=alpha, base_measure=self.base_delay_sampler)
        self.paths = []

    def sample_paths(self, time_points: np.ndarray):
        """
        Samples a realistic multi-path channel with COMPLEX gains.
        """
        _, delays = self.dp.sample()
        self.paths = []
        train_x = torch.tensor(time_points, dtype=torch.float32).view(-1, 1)

        for delay in delays:
            model_real, likelihood_real = create_gp_model(train_x, torch.zeros_like(train_x.squeeze()), kernel_type=self.kernel_type)
            model_imag, likelihood_imag = create_gp_model(train_x, torch.zeros_like(train_x.squeeze()), kernel_type=self.kernel_type)
            model_real.eval()
            likelihood_real.eval()
            model_imag.eval()
            likelihood_imag.eval()

            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                prior_real = model_real(train_x)
                gains_real = prior_real.sample().squeeze().numpy()
                prior_imag = model_imag(train_x)
                gains_imag = prior_imag.sample().squeeze().numpy()

            complex_gains = gains_real + 1j * gains_imag

            def gain_fn(t, _gains=complex_gains, _t=time_points):
                interp_real = np.interp(t, _t, _gains.real, left=_gains.real[0], right=_gains.real[-1])
                interp_imag = np.interp(t, _t, _gains.imag, left=_gains.imag[0], right=_gains.imag[-1])
                return interp_real + 1j * interp_imag


            self.paths.append(Path(delay, gain_fn))

    def evaluate(self, time_points: np.ndarray) -> list:
        """Evaluates the gains of all paths at given time points."""
        evaluated_paths = []
        for path in self.paths:
            gains = path.evaluate_gain(time_points)
            evaluated_paths.append((path.delay, gains))
        return evaluated_paths

    def __deepcopy__(self, memo):
        new_obj = DelayDopplerChannel(self.alpha, self.base_delay_sampler, self.kernel_type)
        memo[id(self)] = new_obj
        new_obj.dp = copy.deepcopy(self.dp, memo)
        new_obj.paths = copy.deepcopy(self.paths, memo)
        return new_obj

def sum_paths_at_time(channel_paths, t):
    total_gain = 0.0
    for path in channel_paths:
        gain_at_t = path.evaluate_gain(np.array([t]))
        total_gain += gain_at_t[0]
    return total_gain
