import numpy as np
from src.models.channel_model import DelayDopplerChannel
from src.inference.smc import SMCFilter
from src.inference.classical_dfe import calculate_dfe_taps, run_dfe_equalizer
from src.utils.simulators import generate_symbols, generate_channel_outputs
from src.utils.metrics import compute_ber

def main():
    true_channel = channel_factory(cfg)
    true_channel.sample_paths(train_times)
    h_true = true_channel
    
    tx_symbols = generate_symbols(...)
    rx_signal, _ = generate_channel_outputs(true_channel, tx_symbols, noise_var, train_times)

    w_oracle, b_oracle = calculate_dfe_taps(h_true, snr_db, ff_taps=10, fb_taps=5)
    est_symbols_oracle = run_dfe_equalizer(rx_signal, w_oracle, b_oracle)
    ber_oracle = compute_ber(tx_symbols, est_symbols_oracle)
    print(f"Benchmark DFE BER (Perfect Knowledge): {ber_oracle:.4f}")

    smc = SMCFilter(...)
    smc.initialize(train_times)
    
    est_symbols_learning = np.zeros_like(tx_symbols)
    update_period = 10 
    w_learning, b_learning = None, None

    for n in range(num_symbols):
        if n % update_period == 0:
            h_hat = smc.get_mean_channel_estimate() 
            w_learning, b_learning = calculate_dfe_taps(h_hat, snr_db, ff_taps=10, fb_taps=5)

        x_hist = tx_symbols[max(0, n-10):n+1] 
        smc.step(rx_signal[n], x_hist, n)
        
        ff_out = np.dot(w_learning, rx_signal[max(0, n-10):n+1][::-1])
        history = est_symbols_learning[max(0, n - 5):n]
        fb_in = np.dot(b_learning[:len(history)], history[::-1])
        z_prime = ff_out - fb_in
        est_symbols_learning[n] = 1 if np.real(z_prime) >= 0 else -1

    ber_learning = compute_ber(tx_symbols, est_symbols_learning)
    print(f"SMC-DFE BER (Learned Channel): {ber_learning:.4f}")
