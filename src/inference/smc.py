import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import torch
import copy
import gpytorch
from typing import List, Callable, Tuple
from src.models.channel_model import DelayDopplerChannel
from src.inference.resampling import get_resampler


class Particle:
    def __init__(self, channel: DelayDopplerChannel):
        """
        Represents a single particle (hypothesis about channel).
        """
        self.channel = channel
        self.weight = 1.0

    def predict(self, time_n: float):
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
        self.channel_factory = channel_factory
        self.noise_var = noise_var
        self.resampler = get_resampler(resample_method)
        self.particles = []  # time-indexed list of particle lists

    def initialize(self, train_times: np.ndarray):
        initial_particles = []
        for _ in range(self.num_particles):
            channel = self.channel_factory()
            channel.sample_paths(train_times)
            initial_particles.append(Particle(channel))
        self.particles = [initial_particles]  

    def step(self, y_n: complex, x_hist: np.ndarray, time_n: float):
        prev_particles = self.particles[-1]
        weights = []

        for p in prev_particles:
            likelihood = p.compute_likelihood(y_n, x_hist, self.noise_var, time_n)
            p.weight *= likelihood
            weights.append(p.weight)

        weights = np.array(weights)
        total = np.sum(weights)
        if total == 0 or not np.isfinite(total):
            weights[:] = 1.0 / len(weights)
        else:
            weights /= total

        for i, p in enumerate(prev_particles):
            p.weight = weights[i]

        indices = self.resampler(weights)
        new_particles = [self.copy_particle(prev_particles[i]) for i in indices]

        self.particles.append(new_particles)

    def copy_particle(self, particle: Particle) -> Particle:
        new_channel = copy.deepcopy(particle.channel) 
        return Particle(new_channel)

    def get_mean_channel_estimate(self, time_n, max_delay_taps):
        """
        Calculates the mean channel impulse response estimate from all particles.

        Args:
            time_n (float): The current time at which to evaluate the gains.
            max_delay_taps (int): The length of the impulse response vector to create.

        Returns:
            np.ndarray: The estimated channel impulse response vector, h_hat.
        """
        all_h_hats = []
        current_particles = self.particles[-1]

        for p in current_particles:
            h_particle = np.zeros(max_delay_taps, dtype=np.complex64)
            for path in p.channel.paths:
                delay_tap = int(round(path.delay))
                if 0 <= delay_tap < max_delay_taps:
                    gain = path.gain_fn(np.array([time_n]))
                    h_particle[delay_tap] += gain[0]
            all_h_hats.append(h_particle)
        h_hat_mean = np.mean(all_h_hats, axis=0)
        return h_hat_mean

    def estimate_symbol(self, x_hist: np.ndarray, time_n: float) -> Tuple[complex, float]:
        y_preds = []
        weights = []
        for p in self.particles[-1]:  
            taps = p.channel.evaluate(np.array([time_n]))
            y_hat = 0.0
            for delay, gains in taps:
                tap_idx = int(round(delay))
                if tap_idx < len(x_hist):
                    y_hat += gains[0] * x_hist[-(tap_idx + 1)]
            y_preds.append(y_hat)
            weights.append(p.weight)

        weights = np.array(weights)
        total = np.sum(weights)
        if total == 0 or not np.isfinite(total):
            weights[:] = 1.0 / len(weights)
        else:
            weights /= total

        y_preds = np.array(y_preds)
        y_mean = np.sum(y_preds * weights)
        print(f"Step {time_n}, y_hat = {y_mean}")
        y_var = np.sum(np.abs(y_preds - y_mean) ** 2 * weights)
        return y_mean, y_var
