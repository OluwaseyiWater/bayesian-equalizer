"""
Full Bayesian nonparametric channel model combining DP-distributed path parameters
(e.g., delays) and GP-modeled time-varying amplitudes. Used for both simulation and inference.
"""

import numpy as np
import torch
from src.models.dp import StickBreakingDP
from src.models.gp import create_gp_model


class DelayDopplerPath:
    def __init__(self, delay: float, gain_fn):
        """Represents a single path with fixed delay and time-varying gain modeled by a GP."""
        self.delay = delay
        self.gain_fn = gain_fn  # Callable: t -> a(t)

    def evaluate_gain(self, time_points: np.ndarray) -> np.ndarray:
        return self.gain_fn(time_points)


class DelayDopplerChannel:
    def __init__(self, alpha: float, base_delay_sampler, kernel_type: str = 'RBF', max_paths: int = 20):
        """
        Delay-Doppler channel with DP prior over delays and GP prior over gains.

        Args:
            alpha: DP concentration parameter
            base_delay_sampler: function to draw from base distribution G0 over delays
            kernel_type: GP kernel type ('RBF' or 'Matern')
            max_paths: maximum number of paths (truncation for stick-breaking)
        """
        self.dp = StickBreakingDP(alpha, base_delay_sampler, max_atoms=max_paths)
        self.kernel_type = kernel_type
        self.paths = []

    def sample_paths(self, train_times: np.ndarray):
        """
        Sample paths from the DP prior and fit GPs to initial random gains.
        """
        weights, delays = self.dp.sample()
        self.paths = []

        for delay in delays:
            # Simulate training data: random GP samples for gains at train_times
            train_x = torch.tensor(train_times, dtype=torch.float32)
            train_y = torch.randn_like(train_x) * 0.5  # Placeholder for gain samples

            model, likelihood = create_gp_model(train_x, train_y, kernel_type=self.kernel_type)
            model.eval()  # In inference, model is pre-trained or pre-fit

            def gain_fn(t):
                t_tensor = torch.tensor(t, dtype=torch.float32)
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    pred = model(t_tensor)
                return pred.mean.numpy()

            self.paths.append(DelayDopplerPath(delay, gain_fn))

    def evaluate(self, time_idx: np.ndarray) -> np.ndarray:
        """
        Evaluate the full channel impulse response at given time steps.
        Returns a list of tuples (delay, a(t)) at each time.
        """
        impulse_response = []
        for path in self.paths:
            gain_t = path.evaluate_gain(time_idx)
            impulse_response.append((path.delay, gain_t))
        return impulse_response
