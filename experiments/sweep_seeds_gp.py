import os, sys, argparse, csv, time, yaml
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.modem import qpsk_mod
from utils.interleave import make_block_interleaver, interleave, deinterleave
import channels.linear_isi as chlin
from metrics.ber_fer import ber
from metrics.snrr_mfb import matched_filter_bound, snr_receiver
from dp_gp.interface import rbpf_detect

# Prefer JAX LDPC
try:
    from channel_codes.ldpc_jax import LDPCJAX as LDPCCode
except Exception:
    from channel_codes.ldpc import LDPCCode


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def make_channel(cfg, rng):
    ch_cfg = cfg.get('channel', {'type': 'PR1D'})
    ch_type = ch_cfg.get('type', 'PR1D').upper()
    if ch_type == 'PR1D':
        h = chlin.pr1d()
    elif ch_type == 'EPR4':
        h = chlin.epr4()
    elif ch_type == 'RANDOM':
        h = chlin.random_fir(int(ch_cfg.get('L', 5)), seed=cfg.get('seed', 0))
    else:
        raise ValueError(f"Unknown channel {ch_type}")
    return ch_type, h, chlin.make_channel(h)


def run_once(cfg, seed, code_seed):
    # Pull knobs from YAML (RBPF code-aided branch)
    eq = cfg['equalizer']
    snr_db = float(cfg.get('snr_db', 8.0))

    # Model + kwargs
    model = str(eq.get('model', 'ar1')).lower()
    if model == 'ar1':
        model_kwargs = dict(
            rho=float(eq.get('ar_rho', 0.999)),
            q_var=float(eq.get('q_var', 1e-6)),
        )
    else:
        model_kwargs = dict(
            ell=float(eq.get('ell', 12.0)),
            sigma_h2=float(eq.get('sigma_h2', 1.0)),
            dt=float(eq.get('dt', 1.0)),
            q_scale=float(eq.get('q_scale', 1.0)),
        )

    # RBPF/LDPC knobs
    Np          = int(eq.get('Np', 512))
    warm        = int(eq.get('warmup', 1000))
    iters       = int(eq.get('iterations', 2))
    ess_thr     = float(eq.get('ess_thresh', 0.7))
    k           = int(eq.get('ldpc_k', 4000))
    dv          = int(eq.get('ldpc_dv', 3))
    alpha       = float(eq.get('alpha', 0.9))
    ldpc_it     = int(eq.get('ldpc_iters', 30))
    prior_scale = float(eq.get('prior_scale', 1.0))  # DEC->EQ damping
    llr_scale   = float(eq.get('llr_scale',   1.0))  # EQ->DEC scaling

    rng = np.random.default_rng(seed)

    # Code, interleaver, mapping 
    code = LDPCCode(k=k, dv=dv, seed=code_seed, alpha=alpha)
    bits_msg   = rng.integers(0, 2, size=k, dtype=int)
    bits_coded = code.encode(bits_msg)                 # n = 2k
    n = bits_coded.size

    pi_rows = eq.get('pi_rows', None)
    pi = make_block_interleaver(n, nrows=pi_rows if pi_rows is not None else max(2, int(round(np.sqrt(n)))))
    bits_tx = interleave(bits_coded, pi)
    x = qpsk_mod(bits_tx)

    # Channel
    ch_type, h, channel = make_channel(cfg, rng)
    y, noise_var, h_used = channel(x, snr_db, Es=1.0, rng=rng)
    Lch = len(h)

    # Turbo loop (EQ<->DEC, extrinsic schedule)
    apriori_code = np.zeros(n, dtype=float)
    soft_seq = np.zeros_like(x, dtype=np.complex128)
    L_ext_eq_last = None

    for t in range(iters):
        # DEC -> EQ prior (interleaved) with optional damping & clip
        apriori_eq = interleave(apriori_code, pi) * prior_scale
        np.clip(apriori_eq, -16.0, 16.0, out=apriori_eq)

        L_ext_eq, soft_seq, aux = rbpf_detect(
            y=y, L=Lch, sigma_v2=noise_var,
            Np=Np, model=model, model_kwargs=model_kwargs,
            apriori_llr_bits=apriori_eq,
            pilot_sym=x, pilot_len=warm,
            ess_thresh=ess_thr, seed=seed
        )

        # --- sanity prints ---
        if t == 0:
            ess_med = float(np.median(aux.get('ess', []))) if isinstance(aux, dict) else float('nan')
            print(f"[seed {seed:>2}] iter1: apriori LLR min/max={apriori_eq.min():.2f}/{apriori_eq.max():.2f} | ESS_med={ess_med:.1f}")

        # EQ -> DEC: deinterleave + adaptive normalization + gentle scaling + wider clip
        L_ext_dec = deinterleave(L_ext_eq, pi)
        target_std = 3.0
        std = float(np.std(L_ext_dec)) + 1e-9
        L_ext_dec *= (target_std / std)
        if llr_scale != 1.0:
            L_ext_dec *= llr_scale
        np.clip(L_ext_dec, -24.0, 24.0, out=L_ext_dec)

        if t == 1:
            print(f"[seed {seed:>2}] iter2: L_ext_dec std={np.std(L_ext_dec):.2f}, min/max={L_ext_dec.min():.2f}/{L_ext_dec.max():.2f}")

        L_post_dec = L_ext_dec + apriori_code

        # LDPC: extrinsic back to equalizer (NMS)
        L_msg_post, apriori_code = code.decode_extrinsic(
            L_post_dec, iters=80, mode="nms", alpha=0.85, damping=0.0, early_stop=True
        )

        L_ext_eq_last = L_ext_eq

    # Final decode
    if L_ext_eq_last is None:  
        apriori_eq = interleave(apriori_code, pi) * prior_scale
        np.clip(apriori_eq, -16.0, 16.0, out=apriori_eq)
        L_ext_eq_last, soft_seq, _ = rbpf_detect(
            y=y, L=Lch, sigma_v2=noise_var,
            Np=Np, model=model, model_kwargs=model_kwargs,
            apriori_llr_bits=apriori_eq,
            pilot_sym=x, pilot_len=warm,
            ess_thresh=ess_thr, seed=seed
        )

    L_post_dec = deinterleave(L_ext_eq_last, pi)
    target_std = 3.0
    std = float(np.std(L_post_dec)) + 1e-9
    L_post_dec *= (target_std / std)
    if llr_scale != 1.0:
        L_post_dec *= llr_scale
    np.clip(L_post_dec, -24.0, 24.0, out=L_post_dec)
    L_post_dec += apriori_code

    L_msg_post, _ = code.decode_extrinsic(
        L_post_dec, iters=80, mode="nms", alpha=0.85, damping=0.0, early_stop=True
    )
    bits_msg_hat = (L_msg_post < 0).astype(int)

    # Metrics
    msg_ber = ber(bits_msg, bits_msg_hat)
    snr_lin = snr_receiver(x[warm:], soft_seq[warm:])
    snr_mfb = matched_filter_bound(h_used, noise_var, Es=1.0)

    return dict(
        seed=seed, ber=float(msg_ber),
        snr_db=snr_db, ch=ch_type, model=model, model_kwargs=str(model_kwargs),
        Np=Np, warmup=warm, ess_thresh=ess_thr, iters=iters,
        ldpc_k=k, ldpc_iters=ldpc_it, dv=dv, alpha=alpha,
        prior_scale=prior_scale, llr_scale=llr_scale,
        snr_R_lin_db=10*np.log10(snr_lin+1e-15), mfb_db=10*np.log10(snr_mfb+1e-15),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config (your rbpf_*_ca.yaml)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--start", type=int, default=0, help="starting seed value")
    ap.add_argument("--out", default="results/rbpf_gp_seed_sweep.csv")
    ap.add_argument("--code-seed", type=int, default=0, help="fixed LDPC code graph seed")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cfg = load_yaml(args.config)
    rows = []
    t0 = time.time()
    for s in range(args.start, args.start + args.seeds):
        rec = run_once(cfg, seed=s, code_seed=args.code_seed)
        rows.append(rec)
        print(f"[{len(rows)}/{args.seeds}] seed={s} -> BER={rec['ber']:.5f} | SNR_R={rec['snr_R_lin_db']:.2f} dB")

    bers = np.array([r['ber'] for r in rows], dtype=float)
    mean_ber = float(bers.mean())
    std_ber  = float(bers.std(ddof=1)) if len(bers) > 1 else 0.0

    print("\n=== Seed sweep summary ===")
    print(f"config: {args.config}")
    print(f"seeds:  {args.start}..{args.start+args.seeds-1}")
    print(f"BER mean = {mean_ber:.6f} | std = {std_ber:.6f}")

    # write CSV
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
        f.write(f"# mean_ber,{mean_ber}\n")
        f.write(f"# std_ber,{std_ber}\n")

    print(f"Saved CSV to {args.out} | elapsed {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
