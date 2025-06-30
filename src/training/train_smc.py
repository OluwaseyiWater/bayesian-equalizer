"""
Main training/evaluation loop for SMC-based Bayesian nonparametric channel equalizer.
Generates synthetic data, runs the particle filter, and tracks performance metrics.
"""

import numpy as np
import os
from omegaconf import OmegaConf
from src.inference.smc import SMCFilter
from src.channel_model import DelayDopplerChannel
from src.utils.simulators import generate_symbols, generate_channel_outputs
from src.utils.logger import init_wandb, log_wandb, save_csv
from src.training.evaluate import evaluate_run


def base_delay_sampler(min_delay, max_delay):
    return lambda: np.random.uniform(min_delay, max_delay)


def channel_factory(cfg):
    return DelayDopplerChannel(
        alpha=cfg.dp_alpha,
        base_delay_sampler=base_delay_sampler(*cfg.delay_range),
        kernel_type=cfg.kernel_type
    )


def main():
    cfg = OmegaConf.load("configs/default.yaml")
    np.random.seed(42)

    num_symbols = cfg.num_symbols
    snr_db = cfg.snr_db
    noise_var = 10 ** (-snr_db / 10) if cfg.noise_var is None else cfg.noise_var
    train_times = np.linspace(0, num_symbols - 1, num_symbols)

    tx_symbols = generate_symbols(num_symbols, modulation=cfg.modulation)
    true_channel = channel_factory(cfg)
    true_channel.sample_paths(train_times)
    rx_signal, true_gains = generate_channel_outputs(true_channel, tx_symbols, noise_var, train_times)

    smc = SMCFilter(
        num_particles=cfg.num_particles,
        channel_factory=lambda: channel_factory(cfg),
        noise_var=noise_var,
        resample_method=cfg.resample_method
    )
    smc.initialize(train_times)

    est_symbols = []
    for n in range(num_symbols):
        x_hist = tx_symbols[max(0, n - 10):n + 1]
        smc.step(rx_signal[n], x_hist, time_n=n)
        y_est, _ = smc.estimate_symbol(x_hist, time_n=n)
        est_symbols.append(np.sign(np.real(y_est)))

    # Evaluate
    evaluate_run(
        cfg=cfg,
        tx_symbols=tx_symbols,
        est_symbols=est_symbols,
        true_gains=true_gains,
        smc_particles=smc.particles,
        time_points=train_times
    )


if __name__ == '__main__':
    main()
