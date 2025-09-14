
import numpy as np
from numpy.typing import ArrayLike

def pr1d():
    h = np.array([1.0, 1.0], dtype=np.complex128)
    h = h / np.linalg.norm(h)
    return h

def epr4():
    h = np.array([1, 1, -1, -1], dtype=np.complex128)
    h = h / np.linalg.norm(h)
    return h

def random_fir(L:int, seed:int=0):
    rng = np.random.default_rng(seed)
    h = (rng.standard_normal(L) + 1j*rng.standard_normal(L))/np.sqrt(2)
    h = h / np.linalg.norm(h)
    return h

def apply_fir_causal(x, h):
    y_full = np.convolve(np.asarray(x), np.asarray(h), mode='full')
    return y_full[:len(x)]

def awgn(y, snr_db: float, Es: float = 1.0, h_energy: float = 1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    snr_lin = 10**(snr_db/10.0)
    sig_pow = Es * h_energy
    N0 = sig_pow / snr_lin
    n = (rng.standard_normal(len(y)) + 1j*rng.standard_normal(len(y))) * (N0/2.0)**0.5
    return y + n, N0

def make_channel(h):
    h = np.asarray(h)
    h_energy = float(np.vdot(h, h).real)
    def channel(x, snr_db, Es=1.0, rng=None):
        y = apply_fir_causal(x, h)  
        y_noisy, noise_var = awgn(y, snr_db, Es=Es, h_energy=h_energy, rng=rng)
        return y_noisy, noise_var, h
    return channel