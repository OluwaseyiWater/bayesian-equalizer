# experiments/sweep_rbpf_gp.py
import argparse, os, csv, time
import numpy as np

# --- local imports (match your tree) ---
import channels.linear_isi as chlin
from utils.modem import qpsk_mod
from utils.interleave import make_block_interleaver, interleave, deinterleave
from channel_codes.ldpc_jax import LDPCJAX as LDPCCode  # or channel_codes.ldpc if you prefer
from dp_gp.interface import rbpf_detect                  # <-- requires your recent patches
from metrics.ber_fer import ber
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_one(cfg):
    """Run a single coded RBPF turbo loop and return message BER and a few diagnostics."""
    seed       = cfg.get('seed', 0)
    snr_db     = float(cfg.get('snr_db', 8.0))
    ch_type    = cfg.get('channel', 'PR1D').upper()
    model      = cfg.get('model', 'matern32').lower()
    mkws       = cfg.get('model_kwargs', {}) or {}
    Np         = int(cfg.get('Np', 512))
    warm       = int(cfg.get('warmup', 1000))
    ess_thr    = float(cfg.get('ess_thresh', 0.7))
    iters      = int(cfg.get('iterations', 3))
    k          = int(cfg.get('ldpc_k', 8000))
    dv         = int(cfg.get('ldpc_dv', 3))
    alpha      = float(cfg.get('alpha', 0.9))
    ldpc_it    = int(cfg.get('ldpc_iters', 80))
    prior_scale= float(cfg.get('prior_scale', 1.0))
    llr_scale  = float(cfg.get('llr_scale', 1.0))
    pi_rows    = cfg.get('pi_rows', None)

    rng = np.random.default_rng(seed)

    # --- build LDPC, bits, interleaver, symbols ---
    code = LDPCCode(k=k, dv=dv, seed=seed, alpha=alpha)
    bits_msg   = rng.integers(0, 2, size=k, dtype=int)
    bits_coded = code.encode(bits_msg)              # n = 2k
    n = bits_coded.size
    pi = make_block_interleaver(n, nrows=pi_rows if pi_rows is not None else max(2, int(round(np.sqrt(n)))))
    bits_tx = interleave(bits_coded, pi)
    x = qpsk_mod(bits_tx)                           # Es=1

    # --- channel ---
    if ch_type == 'PR1D':
        h = chlin.pr1d()
    elif ch_type == 'EPR4':
        h = chlin.epr4()
    elif ch_type == 'RANDOM':
        h = chlin.random_fir(5, seed=seed)
    else:
        raise ValueError(f'Unknown channel: {ch_type}')
    channel = chlin.make_channel(h)
    y, noise_var, h_used = channel(x, snr_db, Es=1.0, rng=rng)
    Lch = len(h)

    # --- turbo loop (extrinsic schedule) ---
    apriori_code = np.zeros(n, dtype=float)
    soft_seq = np.zeros_like(x, dtype=np.complex128)
    L_ext_eq_last = None

    for _ in range(iters):
        # DEC -> EQ prior …
        apriori_eq = interleave(apriori_code, pi)
        apriori_eq *= prior_scale
        np.clip(apriori_eq, -16.0, 16.0, out=apriori_eq)

        L_ext_eq, soft_seq, aux = rbpf_detect(
            y=y, L=Lch, sigma_v2=noise_var,
            Np=Np, model=model, model_kwargs=mkws,
            apriori_llr_bits=apriori_eq,
            pilot_sym=x, pilot_len=warm,
            ess_thresh=ess_thr, seed=seed
        )
        # remember last EQ->DEC extrinsic for final decode
        L_ext_eq_last = L_ext_eq

        L_ext_dec = deinterleave(L_ext_eq, pi) * llr_scale
        np.clip(L_ext_dec, -16.0, 16.0, out=L_ext_dec)
        L_post_dec = L_ext_dec + apriori_code

        # NMS decoder (normalized min-sum)
        L_msg_post, apriori_code = code.decode_extrinsic(
            L_post_dec, iters=ldpc_it, mode="nms", alpha=0.85, damping=0.0, early_stop=True
        )

        # quick sanity check after first turbo pass
        if _ == 0:
            meta = dict(
                snr_db=snr_db,
                ch=ch_type,
                model=model,
                Np=Np,
                warmup=warm,
                ess=ess_thr,
                ldpcI=ldpc_it,
                k=k,
                dv=dv,
                prior_scale=prior_scale,
                llr_scale=llr_scale,
                mkws=str(mkws),
            )
            if float(np.mean(L_ext_dec * L_ext_dec)) < 0.05:
                return dict(ber=0.5, **meta)

    # final decode to get bits
    if L_ext_eq_last is None:  # iters==0 fallback
        apriori_eq = interleave(apriori_code, pi) * prior_scale
        np.clip(apriori_eq, -16.0, 16.0, out=apriori_eq)
        L_ext_eq_last, soft_seq, aux = rbpf_detect(
            y=y, L=Lch, sigma_v2=noise_var,
            Np=Np, model=model, model_kwargs=mkws,
            apriori_llr_bits=apriori_eq,
            pilot_sym=x, pilot_len=warm,
            ess_thresh=ess_thr, seed=seed
        )
    L_post_dec = deinterleave(L_ext_eq_last, pi) * llr_scale
    np.clip(L_post_dec, -16.0, 16.0, out=L_post_dec)
    L_post_dec += apriori_code
    L_msg_post, _ = code.decode_extrinsic(
        L_post_dec, iters=ldpc_it, mode="nms", alpha=0.85, damping=0.0, early_stop=True
    )
    bits_msg_hat = (L_msg_post < 0).astype(int)
    msg_ber = ber(bits_msg, bits_msg_hat)

    return dict(
        ber=msg_ber,
        snr_db=snr_db,
        ch=ch_type,
        model=model,
        Np=Np,
        warmup=warm,
        ess=ess_thr,
        ldpcI=ldpc_it,
        k=k,
        dv=dv,
        prior_scale=prior_scale,
        llr_scale=llr_scale,
        mkws=str(mkws),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", default="PR1D", choices=["PR1D","EPR4","RANDOM"])
    ap.add_argument("--snr", type=float, default=8.0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default="results/rbpf_gp_sweep.csv")
    ap.add_argument("--fast", action="store_true", help="smaller grid")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # --- grid ---
    models = ["matern32"]
    if args.fast:
        warms   = [800, 1200]
        Nps     = [256, 512]
        esss    = [0.7]
        llr_sc  = [1.0, 1.1]
        pr_sc   = [0.9, 1.0]
        mkwargs = [dict(ell=10.0, sigma_h2=1.0), dict(ell=14.0, sigma_h2=1.0)]
        iters, ldpc_k, ldpc_it = 2, 4000, 30   # lighter per-run workload
    else:
        warms   = [800, 1200, 2000]
        Nps     = [256, 512, 768]
        esss    = [0.5, 0.7]
        llr_sc  = [0.9, 1.0, 1.1]
        pr_sc   = [0.8, 0.9, 1.0]
        mkwargs = [dict(ell=v, sigma_h2=1.0) for v in (8.0, 12.0, 16.0)]
        iters, ldpc_k, ldpc_it = 2, 4000, 30

    # build job list
    jobs = []
    for sd in range(args.seeds):
        for model in models:
            mkws_list = mkwargs
            for warm, Np, ess, g, b, mkws in itertools.product(
                warms, Nps, esss, llr_sc, pr_sc, mkws_list
            ):
                cfg = dict(
                    seed=sd, snr_db=args.snr, channel=args.channel,
                    model=model, model_kwargs=mkws,
                    Np=Np, warmup=warm, ess_thresh=ess,
                    iterations=iters,
                    ldpc_k=ldpc_k, ldpc_dv=3, ldpc_iters=ldpc_it, alpha=0.9,
                    prior_scale=b, llr_scale=g,
                )
                jobs.append(cfg)

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.procs) as ex:
        futs = [ex.submit(run_one, cfg) for cfg in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            rows.append(rec)
            print(f"[{i}/{len(futs)}] BER={rec['ber']:.4f} | {rec}")

    # write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)

        rows_sorted = sorted(rows, key=lambda r: r['ber'])
        print("\nTOP-10 (lowest BER):")
        for r in rows_sorted[:10]:
            print(f"BER={r['ber']:.6f} | model={r['model']} mkws={r['mkws']} "
                  f"Np={r['Np']} warmup={r['warmup']} ess={r['ess']} "
                  f"llr_scale={r['llr_scale']} prior_scale={r['prior_scale']} seed={r.get('seed',0)}")

    print(f"\nSaved CSV to {args.out} | total runs: {len(rows)} | elapsed: {time.time()-t0:.1f}s")
