
import numpy as np
def snr_receiver(x, xhat):
    x = np.asarray(x); xhat = np.asarray(xhat)
    N = min(len(x), len(xhat))
    if N == 0: return float('nan')
    e = x[:N] - xhat[:N]
    es = np.mean(np.abs(x[:N])**2)
    mse = np.mean(np.abs(e)**2) + 1e-15
    return es / mse

def matched_filter_bound(h, noise_var, Es=1.0):
    h = np.asarray(h)
    h_energy = float(np.vdot(h, h).real)
    return (Es * h_energy) / (noise_var + 1e-15)

def gap_to_mfb_db(snr_r, snr_mfb):
    if snr_r <= 0 or snr_mfb <= 0: return float('nan')
    import numpy as np
    return 10*np.log10(snr_mfb) - 10*np.log10(snr_r)
