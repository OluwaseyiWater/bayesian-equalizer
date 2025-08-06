import numpy as np
from typing import List
from src.models.channel_model import sum_paths_at_time

def compute_ber(true_symbols: np.ndarray, est_symbols: np.ndarray, ignore_indices=None):
    true_symbols = np.array(true_symbols)
    est_symbols = np.array(est_symbols)

    if ignore_indices is not None and len(ignore_indices) > 0:
        data_indices = np.ones(len(true_symbols), dtype=bool)
        data_indices[ignore_indices] = False
        true_symbols = true_symbols[data_indices]
        est_symbols = est_symbols[data_indices]

    if len(true_symbols) == 0:
        return 0.0 

    errors = np.sum(true_symbols != est_symbols)
    return errors / len(true_symbols)

def compute_channel_mse(true_gains: list, particles: list, time_points: np.ndarray):
    """
    Compute the Mean Squared Error between the true channel and the particle filter's estimate.
    """
    total_squared_error = 0.0
    num_time_steps = len(time_points)

    for n in range(num_time_steps):
        true_total_gain = sum(gain[0] for _, gain in true_gains[n])
        particle_estimates = [sum_paths_at_time(p.channel.paths, time_points[n]) for p in particles[n]]
        mean_estimated_gain = np.mean(particle_estimates)
        total_squared_error += np.abs(true_total_gain - mean_estimated_gain)**2

    return total_squared_error / num_time_steps
