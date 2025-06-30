"""
Evaluate a saved SMC model or run-time inference result using stored predictions.
Computes BER, channel MSE, and optionally visualizes delay–Doppler structure.
"""

import numpy as np
import os
from omegaconf import OmegaConf
from src.utils.metrics import compute_ber, compute_channel_mse
from src.utils.plotting import plot_delay_doppler_surface
from src.utils.logger import save_csv


def evaluate_run(cfg, tx_symbols, est_symbols, true_gains, smc_particles, time_points):
    ber = compute_ber(tx_symbols, est_symbols)
    mse = compute_channel_mse(true_gains, smc_particles, time_points)

    print(f"Evaluation Results:")
    print(f"  BER: {ber:.4f}")
    print(f"  Channel MSE: {mse:.4f}")

    if cfg.logging.use_csv:
        os.makedirs(cfg.logging.save_dir, exist_ok=True)
        save_csv(
            [[len(tx_symbols), ber, mse]],
            os.path.join(cfg.logging.save_dir, f"{cfg.logging.run_name}_eval.csv"),
            header=["time", "BER", "MSE"]
        )

    if cfg.get("plotting"):
        plot_delay_doppler_surface(
            taps_over_time=true_gains,
            time_points=time_points,
            delay_bins=cfg.plotting.delay_bins,
            max_delay=cfg.plotting.max_delay,
            title="True Delay-Doppler Channel"
        )


if __name__ == '__main__':
    print("This script is a utility. Import `evaluate_run(...)` and call from a sweep or main script.")
