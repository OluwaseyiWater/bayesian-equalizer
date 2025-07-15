import numpy as np
from typing import List
from src.models.channel_model import sum_paths_at_time

def compute_ber(true_symbols, est_symbols, ignore_indices=None):
    true_symbols = np.array(true_symbols)
    est_symbols = np.array(est_symbols)

    if ignore_indices is not None:
        mask = np.ones_like(true_symbols, dtype=bool)
        mask[ignore_indices] = False
        true_symbols = true_symbols[mask]
        est_symbols = est_symbols[mask]

    errors = true_symbols != est_symbols
    return np.mean(errors)


def compute_channel_mse(true_gains, particles, time_points):
    mse = 0.0
    for n, t in enumerate(time_points):
        true_sum = sum(g for _, g in true_gains[n])  #sum true tap gains
        particle_gains = np.array([sum_paths_at_time(p.channel.paths, t) for p in particles[n]])
        mean_est = np.mean(particle_gains)
        mse += np.abs(true_sum - mean_est) ** 2
    return mse / len(time_points)

