import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from src.models.channel_model import DelayDopplerChannel
from src.inference.smc import SMCFilter
from src.inference.classical_dfe import calculate_dfe_taps, run_dfe_equalizer
from src.utils.simulators import generate_symbols, generate_channel_outputs
from src.utils.metrics import compute_ber

def channel_factory(cfg):
    """Factory to create a random channel based on the config."""
    return DelayDopplerChannel(
        alpha=cfg.dp_alpha,
        base_delay_sampler=lambda: np.random.uniform(*cfg.delay_range),
        kernel_type=cfg.kernel_type
    )

def extract_true_channel_taps(channel, time_points, max_delay_taps):
    """
    Converts the true channel object into a discrete-time FIR tap vector.
    This is necessary for the "oracle" DFE to know the channel.
    """
    h_true_time_varying = []
    for t in time_points:
        h_t = np.zeros(max_delay_taps, dtype=np.complex64)
        for path in channel.paths:
            delay_tap = int(round(path.delay))
            if 0 <= delay_tap < max_delay_taps:
                gain = path.gain_fn(np.array([t]))
                h_t[delay_tap] += gain[0]
        h_true_time_varying.append(h_t)
    return h_true_time_varying[len(time_points) // 2]


def main():
    cfg = OmegaConf.load("configs/default.yaml")
    np.random.seed(42)

    num_symbols = cfg.num_symbols
    snr_db = cfg.snr_db
    noise_var = 10**(-snr_db / 10.0)
    train_times = np.arange(num_symbols)
    ff_taps = 12
    fb_taps = 6
    max_delay_taps = int(cfg.delay_range[1]) + 5 

    print("--- Starting DFE Comparison Experiment ---")
    print(f"SNR: {snr_db} dB, Particles: {cfg.smc.num_particles}, Symbols: {num_symbols}\n")

    true_channel = channel_factory(cfg)
    true_channel.sample_paths(train_times)
    h_true = extract_true_channel_taps(true_channel, train_times, max_delay_taps)

    tx_symbols = generate_symbols(num_symbols, modulation=cfg.modulation)
    rx_signal, _ = generate_channel_outputs(true_channel, tx_symbols, noise_var, train_times)

    print("Running Benchmark DFE (Oracle)...")
    w_oracle, b_oracle = calculate_dfe_taps(h_true, snr_db, ff_taps, fb_taps)
    est_symbols_oracle = run_dfe_equalizer(rx_signal, w_oracle, b_oracle)
    ber_oracle = compute_ber(tx_symbols, est_symbols_oracle)
    print(f"-> Benchmark DFE BER (Perfect Knowledge): {ber_oracle:.4f}\n")

    print("Running SMC Filter to estimate the channel...")
    smc = SMCFilter(
        num_particles=cfg.smc.num_particles,
        channel_factory=lambda: channel_factory(cfg),
        noise_var=noise_var,
        resample_method=cfg.smc.resample_method
    )
    smc.initialize(train_times)

    for n in tqdm(range(num_symbols), desc="SMC Progress"):
        x_hist = tx_symbols[max(0, n - 15 + 1):n + 1]
        smc.step(rx_signal[n], x_hist, n)
    
    h_hat = smc.get_mean_channel_estimate(num_symbols - 1, max_delay_taps)
    
    print("\nRunning DFE with SMC-Estimated Channel...")
    w_learning, b_learning = calculate_dfe_taps(h_hat, snr_db, ff_taps, fb_taps)
    est_symbols_learning = run_dfe_equalizer(rx_signal, w_learning, b_learning)
    ber_learning = compute_ber(tx_symbols, est_symbols_learning)
    print(f"-> SMC-DFE BER (Learned Channel): {ber_learning:.4f}\n")
    
    print("--- Experiment Complete ---")
    print(f"Performance Gap: The learning-based DFE was {abs(ber_learning - ber_oracle):.4f} away in BER from the oracle.")


if __name__ == '__main__':
    main()
