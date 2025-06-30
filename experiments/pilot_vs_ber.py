"""
Sweep over pilot symbol fraction to evaluate BER and determine minimum pilot overhead
required for reliable channel tracking.
"""

import numpy as np
import os
from omegaconf import OmegaConf
from src.inference.smc import SMCFilter
from src.channel_model import DelayDopplerChannel
from src.utils.simulators import generate_symbols, generate_channel_outputs
from src.utils.metrics import compute_ber
from src.utils.plotting import plot_metric_vs_parameter
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


def insert_pilots(data, pilot_symbol=1+0j, pilot_frac=0.1):
    N = len(data)
    num_pilots = int(N * pilot_frac)
    indices = np.linspace(0, N - 1, num=num_pilots, dtype=int)
    tx = data.copy()
    tx[indices] = pilot_symbol
    return tx, indices


def run_with_pilots(cfg, tx_with_pilots, pilot_indices, rx_signal, train_times, true_data, true_gains, noise_var):
    smc = SMCFilter(
        num_particles=cfg.num_particles,
        channel_factory=lambda: channel_factory(cfg),
        noise_var=noise_var,
        resample_method=cfg.resample_method
    )
    smc.initialize(train_times)

    est_symbols = []
    for n in range(len(tx_with_pilots)):
        x_hist = tx_with_pilots[max(0, n - 10):n + 1]
        smc.step(rx_signal[n], x_hist, time_n=n)
        y_est, _ = smc.estimate_symbol(x_hist, time_n=n)
        est = tx_with_pilots[n] if n in pilot_indices else np.sign(np.real(y_est))
        est_symbols.append(est)

    ber, _ = evaluate_run(
        cfg=cfg,
        tx_symbols=true_data,
        est_symbols=est_symbols,
        true_gains=true_gains,
        smc_particles=smc.particles,
        time_points=train_times,
        silent=True
    )
    return ber


if __name__ == '__main__':
    cfg = OmegaConf.load("configs/default.yaml")
    np.random.seed(456)

    num_symbols = cfg.num_symbols
    snr_db = cfg.snr_db
    noise_var = 10 ** (-snr_db / 10) if cfg.noise_var is None else cfg.noise_var
    train_times = np.linspace(0, num_symbols - 1, num_symbols)

    true_data = generate_symbols(num_symbols, modulation=cfg.modulation)
    true_channel = channel_factory(cfg)
    true_channel.sample_paths(train_times)
    rx_signal, true_gains = generate_channel_outputs(true_channel, true_data, noise_var, train_times)

    ber_list = []

    if cfg.logging.use_wandb:
        init_wandb(
            project=cfg.logging.project_name,
            run_name="pilot-vs-ber",
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    csv_rows = []
    csv_header = ["pilot_frac", "BER"]

    for pfrac in cfg.pilot_fracs:
        tx_with_pilots, pilot_idx = insert_pilots(true_data, pilot_frac=pfrac)
        ber = run_with_pilots(cfg, tx_with_pilots, pilot_idx, rx_signal, train_times, true_data, true_gains, noise_var)
        print(f"Pilot frac = {pfrac:.2f}, BER = {ber:.4f}")
        ber_list.append(ber)

        if cfg.logging.use_wandb:
            log_wandb({"BER": ber, "pilot_frac": pfrac})
        if cfg.logging.use_csv:
            csv_rows.append([pfrac, ber])

    if cfg.logging.use_csv:
        os.makedirs(cfg.logging.save_dir, exist_ok=True)
        log_path = os.path.join(cfg.logging.save_dir, "pilot-vs-ber.csv")
        save_csv(csv_rows, log_path, header=csv_header)

    plot_metric_vs_parameter(
        x_vals=np.array(cfg.pilot_fracs),
        y_vals=np.array(ber_list),
        metric_name="BER",
        param_name="Pilot Fraction",
        log_x=False
    )
