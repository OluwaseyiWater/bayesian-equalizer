import numpy as np
from scipy.linalg import toeplitz, inv

def calculate_dfe_taps(channel_h, snr_db, ff_taps, fb_taps):
    """
    Calculates the optimal MMSE-DFE taps assuming perfect channel knowledge.
    Based on the principles in Chapter 3.

    Args:
        channel_h (np.ndarray): The true channel impulse response.
        snr_db (float): The Signal-to-Noise Ratio in dB.
        ff_taps (int): The number of feedforward taps (Nf).
        fb_taps (int): The number of feedback taps (Nb).

    Returns:
        tuple: (w, b) feedforward and feedback taps.
    """
    channel_h = np.asarray(channel_h).flatten()
    
    # SNR = Ex / sigma^2. Assume Ex = 1 for BPSK.
    snr_linear = 10**(snr_db / 10.0)
    noise_var = 1.0 / snr_linear

    channel_len = len(channel_h)
    h_padded = np.concatenate([channel_h, np.zeros(ff_taps - 1)])
    H = toeplitz(h_padded[:ff_taps], np.zeros(channel_len + ff_taps - 1))
    R_yy = H @ H.conj().T + noise_var * np.identity(ff_taps)
    delta = int(np.floor(channel_len / 2)) + int(np.floor(ff_taps / 2))
    r_xy = np.zeros(ff_taps, dtype=np.complex64)
    if delta < channel_len:
        r_xy[:channel_len-delta] = channel_h[delta:]
    w = inv(R_yy) @ r_xy
    b = channel_h[1:fb_taps+1]

    return w, b

def run_dfe_equalizer(y_signal, w_ff, b_fb):
    """
    Runs the DFE equalization on a received signal.

    Args:
        y_signal (np.ndarray): The received signal.
        w_ff (np.ndarray): The feedforward taps.
        b_fb (np.ndarray): The feedback taps.

    Returns:
        np.ndarray: The estimated symbols.
    """
    ff_out = np.convolve(y_signal, w_ff, mode='full')
    
    est_symbols = np.zeros_like(y_signal, dtype=int)
    fb_len = len(b_fb)
    
    for i in range(len(y_signal)):
        feedback = 0.0
        if i > 0:
            history = est_symbols[max(0, i - fb_len):i]
            feedback = np.dot(b_fb[:len(history)], history[::-1])
        
        z_prime = ff_out[i] - feedback
        est_symbols[i] = 1 if np.real(z_prime) >= 0 else -1
        
    return est_symbols
