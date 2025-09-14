
import numpy as np
def ber(bits_true, bits_hat):
    bits_true = np.asarray(bits_true).astype(int)
    bits_hat = np.asarray(bits_hat).astype(int)
    N = min(len(bits_true), len(bits_hat))
    if N == 0: return float('nan')
    return np.mean(bits_true[:N] != bits_hat[:N])
