from src.utils.metrics import compute_ber, compute_channel_mse
import numpy as np
import matplotlib.pyplot as plt
from src.utils.plotting import plot_delay_doppler_surface


def evaluate_run(cfg, tx_symbols, est_symbols, true_gains, smc_particles, time_points, silent=False, ignore_indices=None):
    ber = compute_ber(tx_symbols, est_symbols, ignore_indices=ignore_indices)
    mse = compute_channel_mse(true_gains, smc_particles, time_points)

    if not silent:
        print(f"Evaluation Results:\n  BER: {ber:.4f}\n  Channel MSE: {mse:.4f}")

    return ber, mse
