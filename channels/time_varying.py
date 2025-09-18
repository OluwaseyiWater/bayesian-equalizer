import numpy as np

def tv_ar1_fir(L=2, rho=0.9995, q_var=1e-6, seed=0):
    """
    Build a time-varying FIR generator with AR(1) tap dynamics:
      h_{n+1,k} = rho * h_{n,k} + sqrt(q_var) * w_{n,k},  w ~ CN(0,1)
    Returns a function channel(x, snr_db, Es, rng) -> (y, noise_var, h0)
    where h0 is the *initial* tap vector (for reference).
    """
    rng0 = np.random.default_rng(seed)
    # initialize taps at n=0 (small random)
    h0 = (rng0.standard_normal(L) + 1j*rng0.standard_normal(L)) / np.sqrt(2*L)

    def channel(x, snr_db, Es=1.0, rng=None):
        rng = np.random.default_rng(seed) if rng is None else rng
        N = len(x)
        # time-varying taps matrix H[n, k], k=0..L-1
        H = np.zeros((N, L), dtype=np.complex128)
        H[0, :] = h0
        alpha = np.sqrt(q_var)
        for n in range(1, N):
            w = (rng.standard_normal(L) + 1j*rng.standard_normal(L)) / np.sqrt(2.0)
            H[n, :] = rho * H[n-1, :] + alpha * w

        # streaming time-varying convolution: y[n] = sum_k H[n,k] x[n-k]
        xpad = np.pad(x, (L-1, 0), constant_values=0.0+0.0j)
        y = np.zeros(N, dtype=np.complex128)
        for n in range(N):
            acc = 0.0 + 0.0j
            for k in range(L):
                idx = n + (L-1) - k
                acc += H[n, k] * xpad[idx]
            y[n] = acc

        snr_lin = 10.0**(snr_db/10.0)
        # average per-symbol signal power at channel output (time-varying)
        sig_pow = np.mean(np.abs(y)**2)  # includes channel power drift
        noise_var = sig_pow / snr_lin
        v = (rng.standard_normal(N) + 1j*rng.standard_normal(N)) * np.sqrt(noise_var/2.0)
        y_noisy = y + v

        return y_noisy, noise_var, H[0, :].copy()  

    return channel
