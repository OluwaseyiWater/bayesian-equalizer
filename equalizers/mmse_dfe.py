import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 
import numpy as np

def _channel_autocorr(h, maxlag):
    h = np.asarray(h, dtype=np.complex128)
    r_full = np.convolve(h, np.conj(h[::-1]))
    Lh = len(h); center = Lh - 1
    out = []
    for lag in range(-maxlag, maxlag+1):
        idx = lag + center
        out.append(r_full[idx] if 0 <= idx < len(r_full) else 0.0+0.0j)
    return np.array(out, dtype=np.complex128)

def _convmtx(h, Lw):
    h = np.asarray(h, dtype=np.complex128)
    Lh = len(h); Lg = Lh + Lw - 1
    C = np.zeros((Lg, Lw), dtype=np.complex128)
    for n in range(Lg):
        for m in range(Lw):
            k = n - m
            if 0 <= k < Lh:
                C[n, m] = h[k]
    return C  # g = C @ w

def mmse_dfe_design(h, noise_var, Lw=9, Nb=None, delay=None, post_thresh=1e-3):
    """Wiener feed-forward + feedback from combined response; adaptive Nb if None."""
    h = np.asarray(h, dtype=np.complex128)
    Lh = len(h)
    if delay is None:
        delay = min(Lw-1, Lh-1)

    # Wiener FF
    maxlag = Lw - 1
    rvec = _channel_autocorr(h, maxlag)
    R = np.empty((Lw, Lw), dtype=np.complex128)
    for i in range(Lw):
        for j in range(Lw):
            R[i, j] = rvec[i - j + maxlag]
            if i == j:
                R[i, j] += noise_var
    p = np.zeros(Lw, dtype=np.complex128)
    for m in range(Lw):
        k = delay - m
        if 0 <= k < Lh:
            p[m] = h[k]
    w = np.linalg.solve(R, p)

    # Combined impulse and FB taps
    C = _convmtx(h, Lw)
    g = C @ w  # len Lh+Lw-1

    # Choose Nb adaptively if not provided (cancel significant postcursors)
    post_full = g[delay+1:]
    if Nb is None:
        sig = np.where(np.abs(post_full) > post_thresh)[0]
        Nb = int(min(len(post_full), max(1, (sig[-1]+1 if len(sig)>0 else 1))))
    post = post_full[:Nb]
    b = -post.copy()

    # Normalize so main tap is 1
    main = g[delay] if g[delay] != 0 else 1.0
    w /= main; b /= main; g /= main
    return w, b, delay, g


def _apply_ff(y, w, delay):
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

def mmse_dfe_detect(y, w, b, delay, slicer, s_true=None, warmup=0, gate_tau=None, soft_seq=None):
    """
    s_true: pilots for first 'warmup' symbols (oracle FB only during warmup)
    soft_seq: array of soft symbols for *all* times; when provided (and n>=warmup),
              use soft_seq for feedback instead of hard decisions.
    gate_tau: reliability gate on previous slicer input (for DD only).
    Returns: pre (FF), s_hat (decisions), pre_fb (FF+FB slicer input)
    """
    pre = _apply_ff(y, w, delay)
    N = len(pre); Nb = len(b)
    s_hat = np.zeros(N, dtype=np.complex128)
    pre_fb = np.empty(N, dtype=np.complex128)

    for n in range(N):
        # choose FB source
        use_true = (s_true is not None) and (n < warmup)
        kmax = min(Nb, n)

        if kmax > 0:
            if use_true:
                past = s_true[n-kmax:n][::-1]
            elif (soft_seq is not None) and (n >= warmup):
                past = soft_seq[n-kmax:n][::-1]
            else:
                past = s_hat[n-kmax:n][::-1]
            # optional reliability gate only for DD
            if (not use_true) and (soft_seq is None) and (gate_tau is not None) and (n > 0):
                kprev = min(Nb, n-1)
                if kprev > 0:
                    past_prev = s_hat[(n-1)-kprev:(n-1)][::-1]
                    z_prev = pre[n-1] + np.dot(b[:kprev], past_prev)
                else:
                    z_prev = pre[n-1]
                s_prev = slicer(z_prev)
                if np.abs(z_prev - s_prev) > gate_tau:
                    if past.size > 0:
                        past = past[1:]
                        kmax = past.size
        else:
            past = np.zeros(0, dtype=np.complex128)

        fb = np.dot(b[:kmax], past) if kmax > 0 else 0.0+0.0j
        z = pre[n] + fb
        pre_fb[n] = z
        s_hat[n] = slicer(z)

    return pre, s_hat, pre_fb


def mmse_dfe_design_train(y, s_true, Lw=11, Nb=1, delay=1, lam=0.0):
    """
    Joint LS/MMSE design of FF (w) and FB (b) using a known pilot (y, s_true).
    Solves min_{w,b} || Phi [conj(w); b] - s_true ||^2 + lam ||[w;b]||^2,
    where Phi rows are [y_vec, s_past], y_vec = [y[n+d],...,y[n+d-Lw+1]],
    s_past = [s[n-1],...,s[n-Nb]].
    """
    y = np.asarray(y, dtype=np.complex128)
    s_true = np.asarray(s_true, dtype=np.complex128)
    N = len(y)
    pad_left  = Lw - 1 - delay
    pad_right = delay
    ypad = np.pad(y, (pad_left, pad_right), mode='constant')

    rows = []
    targets = []
    start = max(delay, 1)  # need at least one past symbol if Nb>0
    for n in range(start, N):
        idx = n + delay + pad_left
        y_vec = ypad[idx - np.arange(Lw)]
        kmax = min(Nb, n)
        s_past = (s_true[n-kmax:n][::-1] if kmax > 0 else np.zeros(0, dtype=np.complex128))
        phi = np.concatenate([y_vec, s_past])
        rows.append(phi)
        targets.append(s_true[n])

    if len(rows) == 0:
        raise ValueError("Pilot too short for the chosen Lw/Nb/delay.")

    Phi = np.vstack(rows)                 # T x (Lw+Nb)
    t   = np.asarray(targets)             # T
    A = Phi.conj().T @ Phi + lam * np.eye(Lw+Nb, dtype=np.complex128)
    b_vec = Phi.conj().T @ t
    theta = np.linalg.solve(A, b_vec)

    c = theta[:Lw]        # c = conj(w)
    b = theta[Lw:]        # feedback taps
    w = np.conj(c)
    return w, b, delay