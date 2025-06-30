"""
Sequential Monte Carlo (SMC) inference engine for tracking Bayesian nonparametric
channel models with time-varying delay-Doppler taps.
Each particle represents a hypothesis of active paths and their amplitudes.
"""

import numpy as np
from typing import List, Callable, Tuple
from src.channel_model import DelayDopplerChannel
from src.inference.resampling import get_resampler


class Particle:
    def __init__(self, channel: DelayDopplerChannel):
        """
        Represents a single particle (hypothesis about channel).
        """
        self.channel = channel
        self.weight = 1.0

    def predict(self, time_n: float):
        """
        Propagate particle state forward in time (e.g., via GP prior dynamics).
        """
        # GP predictions are embedded in the channel.evaluate call.
        return self.channel.evaluate(np.array([time_n]))

    def compute_likelihood(self, y_n: complex, x_hist: np.ndarray, noise_var: float, time_n: float) -> float:
        """
        Compute p(y_n | x_hist, particle state).
        """
        # Evaluate impulse response at time_n
        taps = self.channel.evaluate(np.array([time_n]))
        y_hat = 0.0
        for delay, gains in taps:
            tap_idx = int(round(delay))
            if tap_idx < len(x_hist):
                y_hat += gains[0] * x_hist[-(tap_idx + 1)]  # Reverse index: x[n - delay]

        # Complex Gaussian likelihood
        error = y_n - y_hat
        likelihood = (1 / (np.pi * noise_var)) * np.exp(-np.abs(error) ** 2 / noise_var)
        return likelihood


class SMCFilter:
    def __init__(self, num_particles: int, channel_factory: Callable[[], DelayDopplerChannel], noise_var: float, resample_method: str = "systematic"):
        self.num_particles = num_particles
        self.particles = [Particle(channel_factory()) for _ in range(num_particles)]
        self.noise_var = noise_var
        self.resampler = get_resampler(resample_method)

    def initialize(self, train_times: np.ndarray):
        for p in self.particles:
            p.channel.sample_paths(train_times)

    def step(self, y_n: complex, x_hist: np.ndarray, time_n: float):
        weights = []
        for p in self.particles:
            likelihood = p.compute_likelihood(y_n, x_hist, self.noise_var, time_n)
            p.weight *= likelihood
            weights.append(p.weight)

        # Normalize weights
        weights = np.array(weights)
        weights /= np.sum(weights)
        for i, p in enumerate(self.particles):
            p.weight = weights[i]

        # Resample
        self.resample(weights)

    def resample(self, weights):
        indices = self.resampler(weights)
        new_particles = [self.copy_particle(self.particles[i]) for i in indices]
        self.particles = new_particles
        for p in self.particles:
            p.weight = 1.0 / self.num_particles

    def copy_particle(self, particle: Particle) -> Particle:
        # Deep copy with new channel instance
        new_channel = particle.channel  # Placeholder: implement deepcopy if mutation occurs
        return Particle(new_channel)

    def estimate_symbol(self, x_hist: np.ndarray, time_n: float) -> Tuple[complex, float]:
        """
        Estimate the received symbol (posterior mean) and uncertainty.
        """
        y_preds = []
        weights = []
        for p in self.particles:
            taps = p.channel.evaluate(np.array([time_n]))
            y_hat = 0.0
            for delay, gains in taps:
                tap_idx = int(round(delay))
                if tap_idx < len(x_hist):
                    y_hat += gains[0] * x_hist[-(tap_idx + 1)]
            y_preds.append(y_hat)
            weights.append(p.weight)

        weights = np.array(weights)
        weights /= np.sum(weights)
        y_mean = np.sum(np.array(y_preds) * weights)
        y_var = np.sum(np.abs(np.array(y_preds) - y_mean) ** 2 * weights)
        return y_mean, y_var
