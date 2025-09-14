import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 
import numpy as np

def _channel_autocorr(h, maxlag):
    h = np.asarray(h, dtype=np.complex128)
    r_full = np.convolve(h, np.conj(h[::-1]))                # lags = -(Lh-1) .. +(Lh-1)
    Lh = len(h); center = Lh - 1
    out = []
    for lag in range(-maxlag, maxlag+1):
        idx = lag + center
        out.append(r_full[idx] if 0 <= idx < len(r_full) else 0.0+0.0j)
    return np.array(out, dtype=np.complex128)

def mmse_le_design(h, noise_var, Lw=7, delay=None):
    """
    Design an FIR MMSE linear equalizer for known channel h and noise variance.
    Observation y_vec[n] = [y[n+delay], y[n+delay-1], ..., y[n+delay-Lw+1]].
    Solve (Ryy) w = r_y,s  where Ryy[i,j] = r_y[i-j] + N0*δ_ij and r_y,s[m] = h[delay-m].
    """
    h = np.asarray(h, dtype=np.complex128)
    Lh = len(h)
    if delay is None:
        delay = min(Lw-1, Lh-1)  # put main tap inside the window
    maxlag = Lw - 1
    rvec = _channel_autocorr(h, maxlag)                      # length 2*Lw-1
    # Ryy Toeplitz
    R = np.empty((Lw, Lw), dtype=np.complex128)
    for i in range(Lw):
        for j in range(Lw):
            R[i, j] = rvec[i - j + maxlag]
            if i == j:
                R[i, j] += noise_var
    # Cross-corr vector p[m] = h[delay - m] (0 if out of range)
    p = np.zeros(Lw, dtype=np.complex128)
    for m in range(Lw):
        k = delay - m
        if 0 <= k < Lh:
            p[m] = h[k]
    w = np.linalg.solve(R, p)
    return w, delay

def mmse_le_apply(y, w, delay):
    """
    Apply LE: ŝ[n] = w^H [ y[n+delay], y[n+delay-1], ..., y[n+delay-Lw+1] ].
    Returns 'pre' of same length as y (aligned to original symbol times).
    """
    y = np.asarray(y, dtype=np.complex128)
    Lw = len(w)
    pad_left  = Lw - 1 - delay
    pad_right = delay
    ypad = np.pad(y, (pad_left, pad_right), mode='constant')
    N = len(y)
    pre = np.empty(N, dtype=np.complex128)
    for n in range(N):
        idx = n + delay + pad_left
        vec = ypad[idx - np.arange(Lw)]   # [y[idx], y[idx-1], ..., y[idx-Lw+1]]
        pre[n] = np.vdot(w, vec)          # w^H * vec
    return pre

def mmse_le(y, h, noise_var, Lw=7, delay=None):
    w, d = mmse_le_design(h, noise_var, Lw=Lw, delay=delay)
    pre = mmse_le_apply(y, w, d)
    return pre, w, d
