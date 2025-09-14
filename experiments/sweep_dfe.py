import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np, yaml
from utils.modem import qpsk_mod
from utils.detection import hard_slicer
from channels.linear_isi import pr1d, make_channel
from metrics.ber_fer import ber
from equalizers.mmse_dfe import mmse_dfe_design_train, mmse_dfe_detect

def run_once(seed, n_bits, snr_db, Lw, delay, Nb, warmup, gate_tau, lam):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_bits, dtype=int)
    x = qpsk_mod(bits)
    h = pr1d()
    ch = make_channel(h)
    y, noise_var, _ = ch(x, snr_db, Es=1.0, rng=rng)

    # train on pilot (known) then decision-directed (no oracle in data)
    y_train, s_train = y[:warmup], x[:warmup]
    w, b, dly = mmse_dfe_design_train(y_train, s_train, Lw=Lw, Nb=Nb, delay=delay, lam=lam)

    def slicer_one(z):
        s, _ = hard_slicer(np.array([z]), 'qpsk'); return s[0]

    pre, s_hat, pre_fb = mmse_dfe_detect(y, w, b, dly, slicer_one,
                                         s_true=x, warmup=warmup, gate_tau=gate_tau)
    _, bits_hat = hard_slicer(s_hat, 'qpsk')
    return ber(bits, bits_hat)

if __name__ == "__main__":
    grid = {
        "Lw":      [9, 11, 13],
        "delay":   [1],
        "Nb":      [1],
        "warmup":  [200, 300, 500],
        "gate":    [None, 0.3, 0.4, 0.5],
        "lam":     [0.0, 1e-4, 1e-3],
    }
    snr_db, n_bits, seed = 8.0, 20000, 0
    best = (1.0, None)
    for Lw in grid["Lw"]:
        for delay in grid["delay"]:
            for Nb in grid["Nb"]:
                for warm in grid["warmup"]:
                    for gate in grid["gate"]:
                        for lam in grid["lam"]:
                            b = run_once(seed, n_bits, snr_db, Lw, delay, Nb, warm, gate, lam)
                            cfg = (Lw, delay, Nb, warm, gate, lam)
                            print(f"BER={b:.4f} | Lw={Lw} delay={delay} Nb={Nb} warm={warm} gate={gate} lam={lam}")
                            if b < best[0]:
                                best = (b, cfg)
    print("\nBEST:", best)
