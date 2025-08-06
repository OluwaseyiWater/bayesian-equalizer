import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from omegaconf import OmegaConf
from src.inference.smc import SMCFilter
from src.models.channel_model import DelayDopplerChannel
from src.utils.simulators import generate_symbols, generate_channel_outputs
from src.training.evaluate import evaluate_run
from src.utils.plotting import plot_metric_vs_parameter

def channel_factory(cfg):
    """Factory to create a random channel based on the config."""
    return DelayDopplerChannel(
        alpha=cfg.dp_alpha,
        base_delay_sampler=lambda: np.random.uniform(*cfg.delay_range),
        kernel_type=cfg.kernel_type
    )

def insert_pilots(data, pilot_symbol=1+0j, pilot_frac=0.1):
    N = len(data)
    if pilot_frac <= 0:
        return data.copy(), []
    num_pilots = int(N * pilot_frac)
    indices = np.linspace(0, N - 1, num=num_pilots, dtype=int)
    tx = data.copy()
    tx[indices] = pilot_symbol
    return tx, list(indices)

def run_with_pilots(cfg, tx_with_pilots, pilot_indices, rx_signal, train_times, true_data, true_gains, noise_var):
    """Runs the SMC filter and returns the calculated BER and MSE."""
    smc = SMCFilter(
        num_particles=cfg.smc.num_particles,
        channel_factory=lambda: channel_factory(cfg),
        noise_var=noise_var,
        resample_method=cfg.smc.resample_method
    )
    smc.initialize(train_times)

    est_symbols = []
    x_hist_len = 15
    max_delay = cfg.delay_range[1]

    for n in range(len(tx_with_pilots)):
        x_hist = tx_with_pilots[max(0, n - x_hist_len + 1):n + 1]
        smc.step(rx_signal[n], x_hist, time_n=n)
        y_est, _ = smc.estimate_symbol(x_hist, time_n=n)
        decoded_symbol = 1 if np.real(y_est) >= 0 else -1
        est_symbols.append(decoded_symbol)

    # Align arrays before evaluation
    aligned_true = true_data[:len(true_data)-max_delay]
    aligned_est = est_symbols[max_delay:]
    aligned_pilots = [idx - max_delay for idx in pilot_indices if idx >= max_delay and idx < len(true_data)]


    ber, mse = evaluate_run(
        cfg=cfg,
        tx_symbols=aligned_true,
        est_symbols=aligned_est,
        true_gains=true_gains,
        smc_particles=smc.particles,
        time_points=train_times,
        ignore_indices=aligned_pilots,
        silent=True
    )
    return ber, mse

if __name__ == '__main__':
    cfg = OmegaConf.load("configs/default.yaml")
    np.random.seed(42)

    num_symbols = cfg.num_symbols
    snr_db = cfg.snr_db
    noise_var = 10**(-snr_db / 10)
    train_times = np.arange(num_symbols)

    ber_results = []
    print("--- Running Pilot Fraction Sweep with Realistic Fading Channel ---")

    for pfrac in cfg.pilot_fracs:
        true_channel = channel_factory(cfg)
        true_channel.sample_paths(train_times)
        true_data = generate_symbols(num_symbols, modulation=cfg.modulation)
        tx_with_pilots, pilot_indices = insert_pilots(true_data, pilot_frac=pfrac)
        rx_signal, true_gains = generate_channel_outputs(true_channel, tx_with_pilots, noise_var, train_times)

        ber, mse = run_with_pilots(cfg, tx_with_pilots, pilot_indices, rx_signal, train_times, true_data, true_gains, noise_var)
        
        print(f"Pilot Fraction: {pfrac*100: >4.1f}% -> BER: {ber:.4f}, MSE: {mse:.4f}")
        ber_results.append(ber)

    plot_metric_vs_parameter(
        x_vals=np.array(cfg.pilot_fracs),
        y_vals=np.array(ber_results),
        metric_name="Bit Error Rate (BER)",
        param_name="Pilot Fraction",
    )
