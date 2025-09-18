import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 
import numpy as np

def _convmtx(h, Lw):
    """Build convolution matrix C so (C @ w) = g = h * w, length Lg=Lh+Lw-1."""
    h = np.asarray(h, dtype=np.complex128)
    Lh = len(h); Lg = Lh + Lw - 1
    C = np.zeros((Lg, Lw), dtype=np.complex128)
    # g[n] = sum_{m=0}^{Lw-1} w[m] * h[n-m]
    for n in range(Lg):
        for m in range(Lw):
            k = n - m
            if 0 <= k < Lh:
                C[n, m] = h[k]
    return C

def _apply_ff(y, w, delay):
    """pre[n] = w^H [ y[n+delay], ..., y[n+delay-Lw+1] ] (same as LE apply)."""
    y = np.asarray(y, dtype=np.complex128)
    Lw = len(w)
    pad_left  = Lw - 1 - delay
    pad_right = delay
    ypad = np.pad(y, (pad_left, pad_right), mode='constant')
    N = len(y)
    pre = np.empty(N, dtype=np.complex128)
    for n in range(N):
        idx = n + delay + pad_left
        vec = ypad[idx - np.arange(Lw)]
        pre[n] = np.vdot(w, vec)
    return pre

def zf_dfe_design(h, Lw=7, delay=None, reg=0.0):
    """Zero-forcing DFE design by LS fit to a delta target at 'delay'."""
    h = np.asarray(h, dtype=np.complex128)
    Lh = len(h)
    if delay is None:
        delay = min(Lw-1, Lh-1)  
    C = _convmtx(h, Lw)                      
    Lg = C.shape[0]
    t = np.zeros(Lg, dtype=np.complex128)
    t[delay] = 1.0                            
    # Regularized LS for numerical stability (reg=0 => plain LS)
    if reg > 0:
        w = np.linalg.solve(C.conj().T @ C + reg*np.eye(Lw), C.conj().T @ t)
    else:
        w, *_ = np.linalg.lstsq(C, t, rcond=None)
    g = C @ w                                  # combined impulse response
    # Feedback taps are negative of postcursors after 'delay'
    post = g[delay+1:]
    b = -post.copy()                            # length Nb = Lg-(delay+1)
    return w, b, delay, g

def zf_dfe_detect(y, w, b, delay, slicer):
    pre = _apply_ff(y, w, delay)
    N = len(pre); Nb = len(b)
    s_hat = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        fb = 0.0 + 0.0j
        kmax = min(Nb, n)          
        if kmax > 0:
            past = s_hat[n-kmax:n][::-1]   
            fb = np.dot(b[:kmax], past)
        s_hat[n] = slicer(pre[n] + fb)
    return pre, s_hat
