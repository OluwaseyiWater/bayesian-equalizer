"""
Evaluation metrics for equalizer performance.
Includes Bit Error Rate (BER) and channel Mean Squared Error (MSE).
"""

import numpy as np
from typing import List


def compute_ber(true_symbols: np.ndarray, est_symbols: np.ndarray) -> float:
    """
    Compute bit error rate (BER) between transmitted and detected symbols.
    Assumes BPSK mapping: symbol errors map directly to bit errors.

    Args:
        true_symbols: transmitted BPSK symbols
        est_symbols: estimated/detected BPSK symbols

    Returns:
        BER (0.0 to 1.0)
    """
    errors = np.sum(true_symbols != est_symbols)
    return errors / len(true_symbols)


def compute_channel_mse(true_gains: List[List[tuple]], particles: List, time_points: np.ndarray) -> float:
    """
    Compute average channel MSE over time between true and estimated tap amplitudes.
    Only compares amplitudes at same delays.

    Args:
        true_gains: list of (delay, gain) tuples per time step from ground truth channel
        particles: list of particles from SMC filter
        time_points: time indices over which to evaluate

    Returns:
        MSE over all time steps (float)
    """
    mse_total = 0.0
    for t_idx, t in enumerate(time_points):
        # Average over particles
        particle_gains = []
        for p in particles:
            taps = p.channel.evaluate(np.array([t]))
            particle_gains.append({round(d): g[0] for d, g in taps})

        # Average estimated amplitude per delay
        delays = set(k for g in particle_gains for k in g.keys())
        est_avg = {k: np.mean([g.get(k, 0.0) for g in particle_gains]) for k in delays}

        # Compare with true taps
        true_taps = {round(d): g[0] for d, g in true_gains[t_idx]}
        mse = sum((np.abs(est_avg.get(k, 0.0) - true_taps.get(k, 0.0)) ** 2) for k in delays)
        mse_total += mse / max(len(delays), 1)

    return mse_total / len(time_points)
