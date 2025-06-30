"""
Utilities for generating synthetic symbol sequences and simulating received signals
through a delay-Doppler fading channel with additive Gaussian noise.
"""

import numpy as np
from typing import Tuple


def generate_symbols(num_symbols: int, modulation: str = 'BPSK') -> np.ndarray:
    """
    Generate a sequence of modulated transmit symbols.

    Args:
        num_symbols: number of symbols to generate
        modulation: 'BPSK' or 'QPSK' (extendable)

    Returns:
        Array of complex symbols
    """
    if modulation == 'BPSK':
        bits = np.random.randint(0, 2, size=num_symbols)
        symbols = 2 * bits - 1  # Map 0 -> -1, 1 -> +1
        return symbols.astype(np.complex64)
    elif modulation == 'QPSK':
        bits = np.random.randint(0, 4, size=num_symbols)
        symbols = np.exp(1j * (np.pi/4 + np.pi/2 * bits))
        return symbols.astype(np.complex64)
    else:
        raise ValueError("Unsupported modulation scheme")


def generate_channel_outputs(channel, tx_symbols: np.ndarray, noise_var: float, time_points: np.ndarray) -> Tuple[np.ndarray, list]:
    """
    Simulate the received signal y[n] from transmitted x[n] and a fading channel.

    Args:
        channel: DelayDopplerChannel instance
        tx_symbols: transmitted symbol sequence
        noise_var: variance of AWGN
        time_points: time index array for evaluation

    Returns:
        Tuple of received signal array and list of true tap gains
    """
    rx_signal = []
    true_gains = []
    for n, t in enumerate(time_points):
        taps = channel.evaluate(np.array([t]))
        true_gains.append(taps)

        y_n = 0.0
        for delay, gain in taps:
            tap_idx = int(round(delay))
            if tap_idx < len(tx_symbols) and n - tap_idx >= 0:
                y_n += gain[0] * tx_symbols[n - tap_idx]

        noise = np.sqrt(noise_var / 2) * (np.random.randn() + 1j * np.random.randn())
        rx_signal.append(y_n + noise)

    return np.array(rx_signal), true_gains
