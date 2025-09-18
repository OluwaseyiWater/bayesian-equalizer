import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  

import math, numpy as np, csv, matplotlib.pyplot as plt

from utils.modem import qpsk_mod
from collections import defaultdict
from utils.detection import hard_slicer
from channels.linear_isi import pr1d, epr4, random_fir, make_channel
from metrics.ber_fer import ber
from metrics.snrr_mfb import matched_filter_bound, snr_receiver, gap_to_mfb_db
from equalizers.mmse_le import mmse_le
from equalizers.mmse_dfe import mmse_dfe_detect, mmse_dfe_design_train
from baselines.mlsd_viterbi.viterbi import viterbi_mlsd
from utils.llr import qpsk_llrs, soft_symbol_from_llrs
from utils.interleave import make_block_interleaver, interleave, deinterleave
from channel_codes.ldpc_jax import LDPCJAX as LDPCCode
from channel_codes.ldpc_jax import LLR_MAX

# -------------------
# globals / knobs
# -------------------
SNRs   = list(range(4, 13))  # 4..12 dB Es/N0
SEEDS  = [0, 1, 2]
N_BITS = 20000               # uncoded bits for uncoded baselines
CHANNEL = "PR1D"             # PR1D | EPR4 | RANDOM

# DFE hyperparams (same across runs for consistency)
DFE_LW    = 13
DFE_DELAY = 1
DFE_NB    = 1
DFE_WARM  = 500
DFE_LAM   = 0.0

# LDPC + turbo-eq knobs
LDPC_K      = 8000   # message bits (n = 2k coded)
LDPC_DV     = 3      # column weight for A in H=[A|I]
LDPC_ALPHA  = 0.9    # min-sum scaling
LDPC_ITERS  = 50     # BP iterations inside LDPC
TURBO_ITERS = 2      # equalizer <-> LDPC iterations

# -------------------
# utils
# -------------------
def ensure_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

def gen_data_qpsk(snr_db, n_bits, seed, channel="PR1D"):
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=n_bits, dtype=int)
    x = qpsk_mod(bits)  # unit Es
    if channel.upper() == "PR1D":
        h = pr1d()
    elif channel.upper() == "EPR4":
        h = epr4()
    elif channel.upper() == "RANDOM":
        h = random_fir(5, seed=seed)
    else:
        raise ValueError("Unknown channel")
    ch = make_channel(h)
    y, noise_var, _h = ch(x, snr_db, Es=1.0, rng=rng)
    return bits, x, y, noise_var, _h

def mean_std(rows, key_field, val_field):
    d = defaultdict(list)
    for r in rows:
        v = r.get(val_field)
        if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
            continue
        d[r[key_field]].append(v)
    out = {}
    for k, arr in d.items():
        a = np.array(arr, dtype=float)
        out[k] = (float(np.mean(a)), float(np.std(a)) if len(a) > 1 else 0.0)
    return out

def errplot(xs, ys, es, **kw):
    xs, ys, es = np.array(xs), np.array(ys), np.array(es)
    return plt.errorbar(xs, ys, yerr=es, capsize=3, **kw)

def ebn0_offset_db(mod_bits=2, rate=0.5):
    # Eb/N0 = Es/N0 - 10*log10(mod_bits * rate)
    k = mod_bits * rate
    return -10.0 * np.log10(k)

def annotate_ebn0(ax, offset_db, which='bottom'):
    if abs(offset_db) < 1e-6:
        ax.set_xlabel("SNR (Es/N0) [dB]  =  Eb/N0 [dB] (QPSK, R=1/2)")
    else:
        ax.set_xlabel(f"SNR (Es/N0) [dB]   (Eb/N0 = Es/N0 {offset_db:+.2f} dB)")

# -------------------
# baselines (uncoded)
# -------------------
def run_trivial(bits, x, y, noise_var, h):
    pre = y.copy()
    s_hat, bits_hat = hard_slicer(pre, 'qpsk')
    b = ber(bits, bits_hat)
    snr_mfb_lin = matched_filter_bound(h, noise_var, Es=1.0)
    snr_lin = snr_receiver(x, pre)
    return dict(
        ber=b, ber_dd=b,
        snr_ff_db=10*np.log10(snr_lin+1e-15),
        snr_fb_db=np.nan, snr_or_db=np.nan,
        mfb_db=10*np.log10(snr_mfb_lin+1e-15),
        gap_ff_db=gap_to_mfb_db(snr_lin, snr_mfb_lin),
        gap_or_db=np.nan, gap_fb_db=np.nan
    )

def run_mmse_le(bits, x, y, noise_var, h, Lw=9, delay=None):
    pre, w, d = mmse_le(y, h, noise_var, Lw=Lw, delay=delay)
    s_hat, bits_hat = hard_slicer(pre, 'qpsk')
    b = ber(bits, bits_hat)
    snr_mfb_lin = matched_filter_bound(h, noise_var, Es=1.0)
    snr_lin = snr_receiver(x, pre)
    return dict(
        ber=b, ber_dd=b,
        snr_ff_db=10*np.log10(snr_lin+1e-15),
        snr_fb_db=np.nan, snr_or_db=np.nan,
        mfb_db=10*np.log10(snr_mfb_lin+1e-15),
        gap_ff_db=gap_to_mfb_db(snr_lin, snr_mfb_lin),
        gap_or_db=np.nan, gap_fb_db=np.nan
    )

def run_mlsd(bits, x, y, noise_var, h):
    s_hat = viterbi_mlsd(y, h, modulation='qpsk')
    _, bits_hat = hard_slicer(s_hat, 'qpsk')
    b = ber(bits, bits_hat)
    snr_mfb_lin = matched_filter_bound(h, noise_var, Es=1.0)
    return dict(
        ber=b, ber_dd=b,
        snr_ff_db=np.nan, snr_fb_db=np.nan, snr_or_db=np.nan,
        mfb_db=10*np.log10(snr_mfb_lin+1e-15),
        gap_ff_db=np.nan, gap_or_db=np.nan, gap_fb_db=np.nan
    )

def run_mmse_dfe(bits, x, y, noise_var, h,
                 Lw=DFE_LW, delay=DFE_DELAY, Nb=DFE_NB, warmup=DFE_WARM, gate_tau=None, lam=DFE_LAM):
    # Train (pilot) then DD
    y_tr, s_tr = y[:warmup], x[:warmup]
    w, b, d = mmse_dfe_design_train(y_tr, s_tr, Lw=Lw, Nb=Nb, delay=delay, lam=lam)
    def slicer_one(z):
        s, _ = hard_slicer(np.array([z]), 'qpsk'); return s[0]
    pre, s_hat, pre_fb = mmse_dfe_detect(y, w, b, d, slicer_one,
                                         s_true=x, warmup=warmup, gate_tau=gate_tau, soft_seq=None)
    # Oracle-linearized diagnostic (FB from true symbols)
    pre_or, _, pre_fb_or = mmse_dfe_detect(y, w, b, d, slicer_one,
                                           s_true=x, warmup=len(x), gate_tau=None, soft_seq=x)

    _, bits_hat = hard_slicer(s_hat, 'qpsk')
    b_total = ber(bits, bits_hat)
    b_dd = ber(bits[warmup:], bits_hat[warmup:])

    snr_mfb_lin = matched_filter_bound(h, noise_var, Es=1.0)
    snr_ff = snr_receiver(x[warmup:], pre[warmup:])
    snr_or = snr_receiver(x[warmup:], pre_fb_or[warmup:])
    snr_fb = snr_receiver(x[warmup:], pre_fb[warmup:])

    return dict(
        ber=b_total, ber_dd=b_dd,
        snr_ff_db=10*np.log10(snr_ff+1e-15),
        snr_or_db=10*np.log10(snr_or+1e-15),
        snr_fb_db=10*np.log10(snr_fb+1e-15),
        mfb_db=10*np.log10(snr_mfb_lin+1e-15),
        gap_ff_db=gap_to_mfb_db(snr_ff, snr_mfb_lin),
        gap_or_db=gap_to_mfb_db(snr_or, snr_mfb_lin),
        gap_fb_db=gap_to_mfb_db(snr_fb, snr_mfb_lin)
    )

# -------------------
# coded (LDPC) baseline
# -------------------
def run_mmse_dfe_ca_ldpc(snr_db, seed, channel=CHANNEL,
                         k=LDPC_K, dv=LDPC_DV, alpha=LDPC_ALPHA, ldpc_iters=LDPC_ITERS,
                         Lw=DFE_LW, delay=DFE_DELAY, Nb=DFE_NB, warmup=DFE_WARM, lam=DFE_LAM,
                         turbo_iters=TURBO_ITERS, pi_rows=None):
    rng = np.random.default_rng(seed)
    code = LDPCCode(k=k, dv=dv, seed=seed, alpha=alpha)

    bits_msg = rng.integers(0, 2, size=k, dtype=int)
    bits_coded = code.encode(bits_msg)
    n = len(bits_coded)
    pi = make_block_interleaver(n, nrows=(pi_rows or max(2, int(round(np.sqrt(n))))))

    bits_tx = interleave(bits_coded, pi)
    x = qpsk_mod(bits_tx)

    # channel
    if channel.upper() == "PR1D":
        h = pr1d()
    elif channel.upper() == "EPR4":
        h = epr4()
    elif channel.upper() == "RANDOM":
        h = random_fir(5, seed=seed)
    else:
        raise ValueError("Unknown channel")
    ch = make_channel(h)
    y, noise_var, _h = ch(x, snr_db, Es=1.0, rng=rng)

    # DFE train
    y_tr, s_tr = y[:warmup], x[:warmup]
    w, b, d = mmse_dfe_design_train(y_tr, s_tr, Lw=Lw, Nb=Nb, delay=delay, lam=lam)

    apriori_code = np.zeros(n, dtype=float)
    soft_seq = None
    pre = None; pre_fb = None

    for _ in range(turbo_iters):
        def slicer_one(z):
            s, _ = hard_slicer(np.array([z]), 'qpsk'); return s[0]
        pre, s_hat, pre_fb = mmse_dfe_detect(y, w, b, d, slicer_one,
                                             s_true=x, warmup=warmup, gate_tau=None, soft_seq=soft_seq)

        sigma2_eff = noise_var * float(np.vdot(w, w).real) + 1e-12
        LI, LQ = qpsk_llrs(pre_fb, sigma2_eff)
        L_ch_eq = np.empty(n, dtype=float); L_ch_eq[0::2]=LI; L_ch_eq[1::2]=LQ
        L_ch_dec = deinterleave(L_ch_eq, pi)

        L_post_dec = L_ch_dec + apriori_code
        L_msg_post, L_ext_code = code.decode_extrinsic(L_post_dec, iters=ldpc_iters, damping=0.2, early_stop=False)

        L_ext_eq = interleave(L_ext_code, pi)
        LI_next, LQ_next = L_ext_eq[0::2], L_ext_eq[1::2]
        soft_seq = soft_symbol_from_llrs(LI_next, LQ_next)
        apriori_code = L_ext_code

    # final decision
    L_post_dec = deinterleave(L_ch_eq, pi) + apriori_code
    L_msg_post, _ = code.decode_extrinsic(L_post_dec, iters=ldpc_iters, damping=0.2, early_stop=False)
    bits_msg_hat = (L_msg_post < 0).astype(int)
    msg_ber = ber(bits_msg, bits_msg_hat)

    snr_mfb_lin = matched_filter_bound(h, noise_var, Es=1.0)
    snr_ff = snr_receiver(x[warmup:], pre[warmup:])
    snr_fb = snr_receiver(x[warmup:], pre_fb[warmup:])
    return dict(
        msg_ber=msg_ber,
        snr_ff_db=10*np.log10(snr_ff+1e-15),
        snr_fb_db=10*np.log10(snr_fb+1e-15),
        mfb_db=10*np.log10(snr_mfb_lin+1e-15),
        gap_ff_db=gap_to_mfb_db(snr_ff, snr_mfb_lin),
        gap_fb_db=gap_to_mfb_db(snr_fb, snr_mfb_lin),
    )


# -------------------
# main sweep
# -------------------
def main():
    ensure_dirs()
    rows = []
    csv_path = "results/snr_sweep_ldpc.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method","snr_db","seed","ber","ber_dd","msg_ber",
                    "snr_ff_db","snr_or_db","snr_fb_db","mfb_db","gap_ff_db","gap_or_db","gap_fb_db"])

        for snr in SNRs:
            for sd in SEEDS:
                bits, x, y, noise_var, h = gen_data_qpsk(snr, N_BITS, sd, CHANNEL)

                rec = run_trivial(bits, x, y, noise_var, h)
                w.writerow(["trivial", snr, sd, rec["ber"], rec["ber_dd"], "",
                            rec["snr_ff_db"], "", "", rec["mfb_db"], rec["gap_ff_db"], "", ""])
                rows.append({"method":"trivial","snr_db":snr,"seed":sd, **rec})

                rec = run_mmse_le(bits, x, y, noise_var, h, Lw=9)
                w.writerow(["mmse_le", snr, sd, rec["ber"], rec["ber_dd"], "",
                            rec["snr_ff_db"], "", "", rec["mfb_db"], rec["gap_ff_db"], "", ""])
                rows.append({"method":"mmse_le","snr_db":snr,"seed":sd, **rec})

                rec = run_mlsd(bits, x, y, noise_var, h)
                w.writerow(["mlsd", snr, sd, rec["ber"], rec["ber_dd"], "",
                            "", "", "", rec["mfb_db"], "", "", ""])
                rows.append({"method":"mlsd","snr_db":snr,"seed":sd, **rec})

                rec = run_mmse_dfe(bits, x, y, noise_var, h)  # uses globals
                w.writerow(["mmse_dfe", snr, sd, rec["ber"], rec["ber_dd"], "",
                            rec["snr_ff_db"], rec["snr_or_db"], rec["snr_fb_db"],
                            rec["mfb_db"], rec["gap_ff_db"], rec["gap_or_db"], rec["gap_fb_db"]])
                rows.append({"method":"mmse_dfe","snr_db":snr,"seed":sd, **rec})

                rec = run_mmse_dfe_ca_ldpc(snr, sd)  # LDPC coded baseline
                w.writerow(["mmse_dfe_ca_ldpc", snr, sd, "", "", rec["msg_ber"],
                            rec["snr_ff_db"], "", rec["snr_fb_db"], rec["mfb_db"], rec["gap_ff_db"], "", rec["gap_fb_db"]])
                rows.append({"method":"mmse_dfe_ca_ldpc","snr_db":snr,"seed":sd, **rec})

    # ---- aggregate over seeds ----
    def agg(method, field): return mean_std([r for r in rows if r["method"]==method], "snr_db", field)

    ber_triv = agg("trivial", "ber")
    ber_le   = agg("mmse_le", "ber")
    ber_dfe  = agg("mmse_dfe", "ber_dd")       # DD-only
    ber_mlsd = agg("mlsd", "ber")
    mber_ld  = agg("mmse_dfe_ca_ldpc", "msg_ber")

    snrff_triv = agg("trivial", "snr_ff_db")
    snrff_le   = agg("mmse_le", "snr_ff_db")
    snr_or_dfe = agg("mmse_dfe", "snr_or_db")
    mfb_avg    = agg("trivial", "mfb_db")
    snrfb_dfe  = agg("mmse_dfe", "snr_fb_db")

    gap_triv = agg("trivial", "gap_ff_db")
    gap_le   = agg("mmse_le", "gap_ff_db")
    gap_dfe  = agg("mmse_dfe", "gap_or_db")    # use oracle-linearized

    # ---- plots ----
    off_db = ebn0_offset_db(mod_bits=2, rate=0.5)

    # Uncoded BER
    plt.figure()
    for name, data in [("trivial", ber_triv), ("mmse_le", ber_le), ("mmse_dfe (DD only)", ber_dfe), ("mlsd", ber_mlsd)]:
        xs = sorted(data.keys())
        ys = [data[x][0] for x in xs]   # mean only
        plt.semilogy(xs, ys, marker='o', label=name)
    ax = plt.gca(); ax.set_yscale('log'); ax.grid(True, which='both')
    annotate_ebn0(ax, off_db)
    plt.ylabel("BER")
    plt.title(f"Uncoded BER vs SNR — {CHANNEL}")
    plt.legend()
    plt.savefig("figures/snr_ber_uncoded.png", dpi=300)


    # Coded Message BER (LDPC 1/2)
    plt.figure()
    xs = sorted(mber_ld.keys())
    ys = [mber_ld[x][0] for x in xs]
    plt.semilogy(xs, ys, marker='o', label="mmse_dfe_ca (LDPC 1/2)")
    ax = plt.gca(); ax.set_yscale('log'); ax.grid(True, which='both')
    annotate_ebn0(ax, off_db)
    plt.ylabel("Message BER (after LDPC decode)")
    plt.title(f"Code-aided DFE (LDPC 1/2, {TURBO_ITERS} turbo iters) — {CHANNEL}")
    plt.legend()
    plt.savefig("figures/snr_ber_coded_ldpc.png", dpi=300)


    # Linearized receiver SNR vs MFB
    plt.figure()
    for (name, data) in [("trivial", snrff_triv), ("mmse_le", snrff_le), ("mmse_dfe (oracle-linearized)", snr_or_dfe)]:
        xs = sorted(data.keys())
        ys = [data[x][0] for x in xs]
        plt.plot(xs, ys, marker='o', label=name)
    xs = sorted(mfb_avg.keys())
    ys = [mfb_avg[x][0] for x in xs]
    plt.plot(xs, ys, linestyle='--', label="MFB")
    annotate_ebn0(plt.gca(), off_db)
    plt.ylabel("SNR_R (dB)")
    plt.title(f"Receiver SNR (apples-to-apples, linearized) — {CHANNEL}")
    plt.grid(True, which='both'); plt.legend()
    plt.savefig("figures/snr_snrr_linearized.png", dpi=300)


    # Nonlinear DFE SNR (FF+FB)
    plt.figure()
    xs = sorted(snrfb_dfe.keys())
    ys = [snrfb_dfe[x][0] for x in xs]
    plt.plot(xs, ys, marker='o', label="mmse_dfe (FF+FB, nonlinear)")
    annotate_ebn0(plt.gca(), off_db)
    plt.ylabel("SNR_R (dB)")
    plt.title(f"DFE Receiver SNR (nonlinear FF+FB) — {CHANNEL}")
    plt.grid(True, which='both'); plt.legend()
    plt.savefig("figures/snr_snrr_dfe_nonlinear.png", dpi=300)


    # Gap to MFB (apples-to-apples)
    plt.figure()
    for (name, data) in [("trivial", gap_triv), ("mmse_le", gap_le), ("mmse_dfe (oracle-linearized)", gap_dfe)]:
        xs = sorted(data.keys())
        ys = [data[x][0] for x in xs]
        plt.plot(xs, ys, marker='o', label=name)
    annotate_ebn0(plt.gca(), off_db)
    plt.ylabel("Gap to MFB (dB)")
    plt.title(f"Gap-to-MFB (linearized) — {CHANNEL}")
    plt.grid(True, which='both'); plt.legend()
    plt.savefig("figures/snr_gap_linearized.png", dpi=300)


    print("\nSaved CSV -> results/snr_sweep_ldpc.csv")
    print("Saved figures ->")
    print("  figures/snr_ber_uncoded.png")
    print("  figures/snr_ber_coded_ldpc.png")
    print("  figures/snr_snrr_linearized.png")
    print("  figures/snr_snrr_dfe_nonlinear.png")
    print("  figures/snr_gap_linearized.png")

if __name__ == "__main__":
    main()
