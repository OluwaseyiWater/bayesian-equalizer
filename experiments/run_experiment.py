# experiments/run_experiment.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # allow running file directly

import argparse
import yaml
import numpy as np

from utils.modem import qpsk_mod, es_of_constellation
from channels.linear_isi import pr1d, epr4, random_fir, make_channel
from metrics.ber_fer import ber
from metrics.snrr_mfb import snr_receiver, matched_filter_bound, gap_to_mfb_db
from equalizers.base import TrivialDetector
from equalizers.mmse_le import mmse_le
from baselines.mlsd_viterbi.viterbi import viterbi_mlsd
from utils.detection import hard_slicer
from equalizers.zf_dfe import zf_dfe_design, zf_dfe_detect
from equalizers.mmse_dfe import mmse_dfe_design_train, mmse_dfe_detect
from utils.llr import qpsk_llrs, soft_symbol_from_llrs
from channel_codes.ldpc import LLR_MAX, LDPCCode
from utils.interleave import make_block_interleaver, interleave, deinterleave
import channels.linear_isi as chlin
from dp_gp.interface import rbpf_detect
import channels.time_varying as chtv



def _ls_channel_estimate(y_pil, x_pil, L):
    """
    Least-squares FIR estimate: y ≈ conv(x, h), h length L.
    Uses a Toeplitz-style regression on the pilot segment.
    """
    y_pil = np.asarray(y_pil, dtype=np.complex128)
    x_pil = np.asarray(x_pil, dtype=np.complex128)
    N = len(y_pil)
    # Build X (N x L): X[n, k] = x[n-k], zero for out-of-range
    X = np.zeros((N, L), dtype=np.complex128)
    for k in range(L):
        X[k:, k] = x_pil[:N-k]
    h_ls, *_ = np.linalg.lstsq(X, y_pil, rcond=None)
    return h_ls



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    # ---- load config ----
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg.get('seed', 0))
    n_bits = int(cfg.get('n_bits', 10000))
    modulation = cfg.get('modulation', 'qpsk').lower()
    snr_db = float(cfg.get('snr_db', 8.0))

    # ---- bits & modulation ----
    bits = rng.integers(0, 2, size=n_bits, dtype=int)
    if modulation != 'qpsk':
        raise NotImplementedError('Only QPSK is wired in the minimal example')
    x = qpsk_mod(bits)
    Es = es_of_constellation('qpsk')

    # ---- channel ----
    ch_cfg = cfg.get('channel', {'type': 'PR1D'})
    ch_type = ch_cfg.get('type', 'PR1D').upper()

    if ch_type == 'PR1D':
        h = chlin.pr1d()
        channel = chlin.make_channel(h)
    elif ch_type == 'EPR4':
        h = chlin.epr4()
        channel = chlin.make_channel(h)
    elif ch_type == 'RANDOM':
        L = int(ch_cfg.get('L', 5))
        h = chlin.random_fir(L, seed=cfg.get('seed', 0))
        channel = chlin.make_channel(h)
    elif ch_type == 'TV_AR1':
        Ltv   = int(ch_cfg.get('L', 2))
        rho   = float(ch_cfg.get('rho', 0.9995))
        q_var = float(ch_cfg.get('q_var', 1e-6))
        channel = chtv.tv_ar1_fir(L=Ltv, rho=rho, q_var=q_var, seed=cfg.get('seed', 0))
        h = None  # time-varying; length is Ltv (use from config)
    else:
        raise ValueError(f'Unknown channel type {ch_type}')

    y, noise_var, h_used = channel(x, snr_db, Es=Es, rng=rng)

    # ---- equalizer choice ----
    eq_cfg = cfg.get('equalizer', {'type': 'trivial'})
    eq_type = eq_cfg.get('type', 'trivial').lower()

    # default to avoid NameError in tail metrics if a branch doesn't set it
    warm = 0

    pre = None
    if eq_type == 'trivial':
        eq = TrivialDetector(modulation='qpsk')
        out = eq.detect(y)
        pre = out.get('pre', None)        # pre-slicer signal (here == y)
        s_hat, bits_hat = out['hard'], out['bits']

    elif eq_type == 'mlsd':
        # default pilot length if not provided
        warm = int(eq_cfg.get('warmup', 1600))
        if ch_type == 'TV_AR1':
            # fixed-tap MLSD: estimate static h from pilot
            Lhat = int(ch_cfg.get('L', 2))  # same length as TV model
            h_hat = _ls_channel_estimate(y[:warm], x[:warm], Lhat)
        else:
            # static channels: use the true h_used returned by the channel wrapper
            h_hat = h_used

        s_hat = viterbi_mlsd(y, h_hat, modulation='qpsk')
        _, bits_hat = hard_slicer(s_hat, 'qpsk')
        pre = None

    elif eq_type == 'mmse_le':
        Lw = int(eq_cfg.get('Lw', 7))
        delay = eq_cfg.get('delay', None)
        pre, w, dly = mmse_le(y, h_used, noise_var, Lw=Lw, delay=delay)
        s_hat, bits_hat = hard_slicer(pre, 'qpsk')

    elif eq_type == 'zf_dfe':
        Lw = int(eq_cfg.get('Lw', 7))
        delay = eq_cfg.get('delay', None)
        reg = float(eq_cfg.get('reg', 0.0))
        w, b, dly, g = zf_dfe_design(h_used, Lw=Lw, delay=delay, reg=reg)
        def slicer_one(z):
            s, _ = hard_slicer(np.array([z]), 'qpsk')
            return s[0]
        pre, s_hat = zf_dfe_detect(y, w, b, dly, slicer_one)
        _, bits_hat = hard_slicer(s_hat, 'qpsk')
        pre_for_snr = pre  # linear pre-slicer is 'pre' here

    elif eq_type == 'mmse_dfe':
        Lw    = int(eq_cfg.get('Lw', 11))
        delay = int(eq_cfg.get('delay', 1))
        Nb    = int(eq_cfg.get('Nb', 1))
        warm  = int(eq_cfg.get('warmup', 500))
        gate  = eq_cfg.get('gate_tau', None)
        gate  = float(gate) if gate is not None else None
        lam   = float(eq_cfg.get('lambda', 0.0))

        # train on first 'warm' symbols (pilot)
        y_train, s_train = y[:warm], x[:warm]
        w, b, dly = mmse_dfe_design_train(y_train, s_train, Lw=Lw, Nb=Nb, delay=delay, lam=lam)

        # detect (decision-directed after warmup)
        def slicer_one(z):
            s, _ = hard_slicer(np.array([z]), 'qpsk'); return s[0]
        pre, s_hat, pre_fb = mmse_dfe_detect(y, w, b, dly, slicer_one, s_true=x, warmup=warm, gate_tau=gate)
        _, bits_hat = hard_slicer(s_hat, 'qpsk')
        pre_for_snr = pre_fb  # include FB path for SNR_R

    elif eq_type == 'mmse_dfe_ca':
        eq_cfg = cfg['equalizer']
        snr_db = float(cfg.get('snr_db', 8.0))
        rng = np.random.default_rng(cfg.get('seed', 0))

        # DFE + LDPC params
        Lw     = int(eq_cfg.get('Lw', 13))
        delay  = int(eq_cfg.get('delay', 1))
        Nb     = int(eq_cfg.get('Nb', 1))
        warm   = int(eq_cfg.get('warmup', 500))
        lam    = float(eq_cfg.get('lambda', 0.0))
        iters  = int(eq_cfg.get('iterations', 2))
        ldpc_k   = int(eq_cfg.get('ldpc_k', 8000))
        ldpc_dv  = int(eq_cfg.get('ldpc_dv', 3))
        ldpc_it  = int(eq_cfg.get('ldpc_iters', 50))
        alpha    = float(eq_cfg.get('alpha', 0.9))

        # --- LDPC encoder (systematic rate-1/2) ---
        code = LDPCCode(k=ldpc_k, dv=ldpc_dv, seed=cfg.get('seed', 0), alpha=alpha)
        bits_msg   = rng.integers(0, 2, size=ldpc_k, dtype=int)
        bits_coded = code.encode(bits_msg)               # len n = 2*k

        # --- Interleaver Π over coded bits (default near-square block) ---
        n = len(bits_coded)
        pi_rows = int(eq_cfg.get('pi_rows', max(2, int(round(np.sqrt(n))))))
        pi = make_block_interleaver(n, nrows=pi_rows)

        bits_tx = interleave(bits_coded, pi)             # Π(coded)
        x = qpsk_mod(bits_tx); Es = 1.0

        # --- Channel ---
        ch_cfg  = cfg.get('channel', {'type': 'PR1D'})
        ch_type = str(ch_cfg.get('type', 'PR1D')).upper()

        # static channels
        if ch_type in ('PR1D',):
            h = chlin.pr1d()
            channel = chlin.make_channel(h)

        elif ch_type in ('EPR4',):
            h = chlin.epr4()
            channel = chlin.make_channel(h)

        elif ch_type in ('RANDOM', 'RAND'):
            L = int(ch_cfg.get('L', 5))
            h = chlin.random_fir(L, seed=cfg.get('seed', 0))
            channel = chlin.make_channel(h)

        # time-varying AR(1) FIR channel
        elif ch_type in ('TV_AR1', 'TV-AR1', 'TVAR1'):
            Ltv   = int(ch_cfg.get('L', 2))
            rho   = float(ch_cfg.get('rho', 0.9995))
            q_var = float(ch_cfg.get('q_var', 1e-6))
            channel = chtv.tv_ar1_fir(L=Ltv, rho=rho, q_var=q_var, seed=cfg.get('seed', 0))
            h = None  # no static h when channel is time-varying

        else:
            raise ValueError(f'Unknown channel type {ch_type}')

        # simulate once you've selected the correct channel object
        y, noise_var, h_used = channel(x, snr_db, Es=Es, rng=rng)


        # --- Train DFE on pilots, then turbo iterations with LDPC (+Π) ---
        y_train, s_train = y[:warm], x[:warm]
        w, b, dly = mmse_dfe_design_train(y_train, s_train, Lw=Lw, Nb=Nb, delay=delay, lam=lam)

        # apriori LLRs are kept in the **decoder (deinterleaved) domain**
        apriori_code = np.zeros(n, dtype=float)
        soft_seq = None
        pre = None; pre_fb = None

        for _ in range(iters):
            def slicer_one(z):
                s, _ = hard_slicer(np.array([z]), 'qpsk'); return s[0]
            pre, s_hat, pre_fb = mmse_dfe_detect(
                y, w, b, dly, slicer_one, s_true=x, warmup=warm, gate_tau=None, soft_seq=soft_seq
            )

            # Channel LLRs are in **interleaved** order → deinterleave before LDPC
            sigma2_eff = noise_var * float(np.vdot(w, w).real) + 1e-12
            LI, LQ = qpsk_llrs(pre_fb, sigma2_eff)
            L_ch_eq = np.empty(n, dtype=float); L_ch_eq[0::2] = LI; L_ch_eq[1::2] = LQ
            L_ch_dec = deinterleave(L_ch_eq, pi)   # Π^{-1}(LLR)

            # EQ → DEC a-posteriori (code domain), LDPC decode, get **extrinsic** back
            L_post_dec = L_ch_dec + apriori_code
            L_msg_post, L_ext_code = code.decode_extrinsic(
                L_post_dec, iters=ldpc_it, mode="nms", alpha=0.85, damping=0.0, early_stop=True
            )

            # DEC → EQ: re-interleave extrinsic to build soft symbols for next pass
            L_ext_eq = interleave(L_ext_code, pi)
            LI_next, LQ_next = L_ext_eq[0::2], L_ext_eq[1::2]
            soft_seq = soft_symbol_from_llrs(LI_next, LQ_next)

            apriori_code = L_ext_code  # next a-priori for decoder

        # Final message decision (decoder domain)
        L_post_dec = deinterleave(L_ch_eq, pi) + apriori_code
        L_msg_post, _ = code.decode_extrinsic(
            L_post_dec, iters=ldpc_it, mode="nms", alpha=0.85, damping=0.0, early_stop=True
        )
        bits_msg_hat = (L_msg_post < 0).astype(int)

        # --- Metrics ---
        b_msg = ber(bits_msg, bits_msg_hat)
        snr_mfb_lin = matched_filter_bound(h_used, noise_var, Es=Es)
        snr_ff = snr_receiver(x[warm:], pre[warm:])
        snr_fb = snr_receiver(x[warm:], pre_fb[warm:])

        print('--- Code-aided DFE (LDPC + interleaver) run complete ---')
        print(f'Channel: {ch_type}, SNR(dB): {snr_db:.2f}, k={ldpc_k}, rate=1/2, iters={iters}, ldpcIters={ldpc_it}')
        print(f'Message BER (after LDPC decode): {b_msg:.4e}')
        print(f'MFB (dB): {10*np.log10(snr_mfb_lin+1e-15):.2f}')
        print(f'SNR_R (FF only, dB): {10*np.log10(snr_ff+1e-15):.2f} | Gap to MFB: {gap_to_mfb_db(snr_ff, snr_mfb_lin):.2f} dB')
        print(f'SNR_R (FF+FB, dB):   {10*np.log10(snr_fb+1e-15):.2f} | Gap to MFB: {gap_to_mfb_db(snr_fb, snr_mfb_lin):.2f} dB (nonlinear)')
        return

    elif eq_type == 'rbpf_ca':
        eq = cfg['equalizer']
        snr_db = float(cfg.get('snr_db', 8.0))
        rng = np.random.default_rng(cfg.get('seed', 0))

        # --- RBPF hyperparams & model selection ---
        Np       = int(eq.get('Np', 256))
        warm     = int(eq.get('warmup', 800))
        iters    = int(eq.get('iterations', 3))
        ess_thr  = float(eq.get('ess_thresh', 0.5))

        model = str(eq.get('model', 'ar1')).lower()
        if model in ('ar1',):
            model_kwargs = dict(
                rho   = float(eq.get('ar_rho', 0.999)),
                q_var = float(eq.get('q_var', 1e-6)),
            )
        elif model in ('matern32','m32','matern_32'):
            model_kwargs = dict(
                ell      = float(eq.get('ell', 12.0)),
                sigma_h2 = float(eq.get('sigma_h2', 1.0)),
                dt       = float(eq.get('dt', 1.0)),
                q_scale  = float(eq.get('q_scale', 1.0)),
            )
        else:
            raise ValueError(f"Unknown RBPF state model '{model}'")

        # soft LLR knobs
        prior_scale = float(eq.get('prior_scale', 1.0))  # DEC->EQ damping
        llr_scale   = float(eq.get('llr_scale',   1.0))  # EQ->DEC scaling

        # --- LDPC params ---
        k        = int(eq.get('ldpc_k', 8000))
        dv       = int(eq.get('ldpc_dv', 3))
        alpha    = float(eq.get('alpha', 0.9))
        ldpc_it  = int(eq.get('ldpc_iters', 80))

        # --- Build code, interleave, map ---
        code = LDPCCode(k=k, dv=dv, seed=cfg.get('seed', 0), alpha=alpha)
        bits_msg   = rng.integers(0, 2, size=k, dtype=int)
        bits_coded = code.encode(bits_msg)          # n = 2k
        n = bits_coded.size

        pi_rows = eq.get('pi_rows', None)
        pi = make_block_interleaver(n, nrows=pi_rows if pi_rows is not None else max(2, int(round(np.sqrt(n)))))

        bits_tx = interleave(bits_coded, pi)
        x = qpsk_mod(bits_tx)                       # Es = 1

        # --- Channel (static and time-varying) ---
        ch_cfg  = cfg.get('channel', {'type': 'PR1D'})
        ch_type = str(ch_cfg.get('type', 'PR1D')).upper()
        if ch_type in ('PR1D',):
            h = chlin.pr1d(); channel = chlin.make_channel(h)
        elif ch_type in ('EPR4',):
            h = chlin.epr4(); channel = chlin.make_channel(h)
        elif ch_type in ('RANDOM','RAND'):
            L = int(ch_cfg.get('L', 5))
            h = chlin.random_fir(L, seed=cfg.get('seed', 0))
            channel = chlin.make_channel(h)
        elif ch_type in ('TV_AR1','TV-AR1','TVAR1'):
            Ltv   = int(ch_cfg.get('L', 2))
            rho   = float(ch_cfg.get('rho', 0.9995))
            q_var = float(ch_cfg.get('q_var', 1e-6))
            channel = chtv.tv_ar1_fir(L=Ltv, rho=rho, q_var=q_var, seed=cfg.get('seed', 0))
            h = None
        else:
            raise ValueError(f'Unknown channel type {ch_type}')

        y, noise_var, h_used = channel(x, snr_db, Es=Es, rng=rng)
        Lch = len(h) if h is not None else int(ch_cfg.get('L', 2))

        # --- Turbo EQ <-> LDPC (EXTRINSIC schedule) ---
        apriori_code  = np.zeros(n, dtype=float)
        L_ext_eq_last = None
        soft_seq = np.zeros_like(x, dtype=np.complex128)

        # pilots (EQ domain) for *delay only*
        pilot_bits_eq = bits_tx[: 2*warm] if warm > 0 else None
        max_lag = int(eq.get('calib_max_lag', 6))
        did_delay_align = False

        for t in range(iters):
            # DEC -> EQ prior (damped & clipped)
            apriori_eq = interleave(apriori_code, pi) * prior_scale
            np.clip(apriori_eq, -16.0, 16.0, out=apriori_eq)

            # RBPF pass
            L_ext_eq, soft_seq, aux = rbpf_detect(
                y=y, L=Lch, sigma_v2=noise_var,
                Np=Np, model=model, model_kwargs=model_kwargs,
                apriori_llr_bits=apriori_eq,
                pilot_sym=x, pilot_len=warm,
                ess_thresh=ess_thr, seed=cfg.get('seed', 0)
            )

            # -------- delay calibration ONCE at t==0 (no lane swap, no polarity flips) --------
            if (t == 0) and (warm > 0) and (not did_delay_align):
                pb = pilot_bits_eq.astype(float)
                sI = 1.0 - 2.0 * pb[0::2]  # +1 for bit 0, -1 for bit 1
                sQ = 1.0 - 2.0 * pb[1::2]

                best_d, best_score = 0, -1e9
                for d in range(-max_lag, max_lag + 1):
                    Lp = np.roll(L_ext_eq[: 2*warm], 2*d)
                    I, Q = Lp[0::2], Lp[1::2]
                    # Keep I->I, Q->Q only (no swapping)
                    score = float(np.mean(I * sI) + np.mean(Q * sQ))
                    if score > best_score:
                        best_score, best_d = score, d
                if best_d != 0:
                    L_ext_eq = np.roll(L_ext_eq, 2*best_d)
                    soft_seq = np.roll(soft_seq, best_d)
                    print(f"[calib] Aligned delay: d={best_d:+d} symbols (score={best_score:.3f}).")
                did_delay_align = True

            L_ext_eq_last = L_ext_eq

            # EQ -> DEC: normalize & scale gently, clip tighter
            L_ext_dec = deinterleave(L_ext_eq, pi)
            target_std = 2.0
            std0 = float(np.std(L_ext_dec)) + 1e-9
            L_ext_dec *= (target_std / std0)
            if llr_scale != 1.0:
                L_ext_dec *= llr_scale
            np.clip(L_ext_dec, -12.0, 12.0, out=L_ext_dec)

            # quick health prints
            if t == 0:
                ess_med = float(np.median(aux.get('ess', []))) if isinstance(aux, dict) and 'ess' in aux else float('nan')
                print(f"[health] iter1: ESS_med={ess_med:.1f} / Np={Np}, L_ext_dec std={np.std(L_ext_dec):.2f}, "
                      f"min/max={L_ext_dec.min():.2f}/{L_ext_dec.max():.2f}")
            elif t == 1:
                print(f"[health] iter2: L_ext_dec std={np.std(L_ext_dec):.2f}, "
                      f"min/max={L_ext_dec.min():.2f}/{L_ext_dec.max():.2f}")

            # LDPC: DEC a-posteriori, return extrinsic
            L_post_dec = L_ext_dec + apriori_code
            L_msg_post, apriori_code = code.decode_extrinsic(
                L_post_dec, iters=ldpc_it, mode="nms", alpha=0.85, damping=0.0, early_stop=True
            )

        # ---- Final LDPC decode (normalize same way) ----
        if L_ext_eq_last is None:
            apriori_eq = interleave(apriori_code, pi) * prior_scale
            np.clip(apriori_eq, -16.0, 16.0, out=apriori_eq)
            L_ext_eq_last, soft_seq, _ = rbpf_detect(
                y=y, L=Lch, sigma_v2=noise_var,
                Np=Np, model=model, model_kwargs=model_kwargs,
                apriori_llr_bits=apriori_eq,
                pilot_sym=x, pilot_len=warm,
                ess_thresh=ess_thr, seed=cfg.get('seed', 0)
            )

        L_post_dec = deinterleave(L_ext_eq_last, pi)
        target_std = 2.0
        std0 = float(np.std(L_post_dec)) + 1e-9
        L_post_dec *= (target_std / std0)
        if llr_scale != 1.0:
            L_post_dec *= llr_scale
        np.clip(L_post_dec, -12.0, 12.0, out=L_post_dec)
        L_post_dec += apriori_code

        L_msg_post, _ = code.decode_extrinsic(
            L_post_dec, iters=ldpc_it, mode="nms", alpha=0.85, damping=0.0, early_stop=True
        )
        bits_msg_hat = (L_msg_post < 0).astype(int)

        # --- Metrics / prints ---
        msg_ber = ber(bits_msg, bits_msg_hat)
        snr_mfb_lin = matched_filter_bound(h_used, noise_var, Es=Es)
        snr_lin = snr_receiver(x[warm:], soft_seq[warm:])

        print('--- RBPF code-aided run complete ---')
        print(f'Channel: {ch_type}, L={Lch}, model={model}, SNR(dB): {snr_db:.2f}, Np={Np}, iters={iters}, ldpcIters={ldpc_it}')
        print(f'Message BER (after LDPC decode): {msg_ber:.4e}')
        print(f'MFB (dB): {10*np.log10(snr_mfb_lin+1e-15):.2f}')
        print(f'SNR_R (linear diagnostic, dB): {10*np.log10(snr_lin+1e-15):.2f}')
        return


    else:
        raise NotImplementedError(f'Unknown equalizer type {eq_type}')

    # ---- metrics (for baseline branches) ----
    N = min(len(x), len(s_hat)) if 's_hat' in locals() else len(x)
    b = ber(bits[warm:], bits_hat[warm:]) if 'bits_hat' in locals() else float('nan')
    snr_mfb_lin = matched_filter_bound(h_used, noise_var, Es=Es)

    snr_r_db = None
    gap_db   = None
    snr_ff_db = None

    if 'pre_for_snr' in locals() and pre_for_snr is not None:
        snr_r_lin = snr_receiver(x[warm:N], pre_for_snr[warm:N])
        snr_r_db  = 10*np.log10(snr_r_lin + 1e-15)
        gap_db    = gap_to_mfb_db(snr_r_lin, snr_mfb_lin)

    if 'pre' in locals() and pre is not None:
        snr_ff_db = 10*np.log10(snr_receiver(x[warm:N], pre[warm:N]) + 1e-15)

    print('--- Minimal run complete ---')
    print(f'Channel: {ch_type}, SNR(dB): {snr_db:.2f}, bits: {n_bits}')
    print(f'BER: {b:.4e}')
    print(f'MFB (dB):   {10*np.log10(snr_mfb_lin+1e-15):.2f}')
    if snr_ff_db is not None:
        print(f'SNR_R (FF only, dB): {snr_ff_db:.2f}')
    if snr_r_db is None:
        print('SNR_R (dB): n/a (no linear pre-slicer output)')
        print('Gap to MFB (dB): n/a')
    else:
        print(f'SNR_R (dB): {snr_r_db:.2f}')
        print(f'Gap to MFB (dB): {gap_db:.2f}')


if __name__ == '__main__':
    main()
