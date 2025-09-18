import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 
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

    # ---- equalizer choice ----
    eq_cfg = cfg.get('equalizer', {'type': 'trivial'})
    eq_type = eq_cfg.get('type', 'trivial').lower()

    # ---- bits & modulation (centralized) ----
    if eq_type in ('mmse_dfe_ca', 'rbpf_ca'):
        # --- LDPC encoder and Interleaver Setup for Code-Aided Equalizers ---
        ldpc_k = int(eq_cfg.get('ldpc_k', 8000))
        ldpc_dv = int(eq_cfg.get('ldpc_dv', 3))
        alpha = float(eq_cfg.get('alpha', 0.9))
        code = LDPCCode(k=ldpc_k, dv=ldpc_dv, seed=cfg.get('seed', 0), alpha=alpha)
        bits_msg = rng.integers(0, 2, size=ldpc_k, dtype=int)
        bits_coded = code.encode(bits_msg)
        n = len(bits_coded)
        pi_rows = eq_cfg.get('pi_rows', None)
        pi = make_block_interleaver(n, nrows=pi_rows if pi_rows is not None else max(2, int(round(np.sqrt(n)))))
        bits_tx = interleave(bits_coded, pi)
        x = qpsk_mod(bits_tx)
        bits = bits_tx # For compatibility with non-coded metrics
    else:
        # --- Standard Bit Generation for Non-Coded Equalizers ---
        bits = rng.integers(0, 2, size=n_bits, dtype=int)
        if modulation != 'qpsk':
            raise NotImplementedError('Only QPSK is wired in the minimal example')
        x = qpsk_mod(bits)

    Es = es_of_constellation('qpsk')

    # ---- channel (centralized) ----
    ch_cfg = cfg.get('channel', {'type': 'PR1D'})
    ch_type = ch_cfg.get('type', 'PR1D').upper()

    if ch_type == 'PR1D':
        h = chlin.pr1d()
        channel = chlin.make_channel(h)
    elif ch_type == 'EPR4':
        h = chlin.epr4()
        channel = chlin.make_channel(h)
    elif ch_type in ('RANDOM', 'RAND'):
        L = int(ch_cfg.get('L', 5))
        h = chlin.random_fir(L, seed=cfg.get('seed', 0))
        channel = chlin.make_channel(h)
    elif ch_type in ('TV_AR1', 'TV-AR1', 'TVAR1'):
        Ltv = int(ch_cfg.get('L', 2))
        rho = float(ch_cfg.get('rho', 0.9995))
        q_var = float(ch_cfg.get('q_var', 1e-6))
        channel = chtv.tv_ar1_fir(L=Ltv, rho=rho, q_var=q_var, seed=cfg.get('seed', 0))
        h = None  # time-varying; length is Ltv
    else:
        raise ValueError(f'Unknown channel type {ch_type}')

    y, noise_var, h_used = channel(x, snr_db, Es=Es, rng=rng)

    # ---- equalizer detection ----
    warm = 0
    pre = None
    pre_slicer_signal_for_snr = None
    bits_to_check = None
    bits_hat_to_check = None

    if eq_type == 'trivial':
        eq = TrivialDetector(modulation='qpsk')
        out = eq.detect(y)
        pre = out.get('pre', None)
        s_hat, bits_hat = out['hard'], out['bits']
        pre_slicer_signal_for_snr = pre

    elif eq_type == 'mlsd':
        warm = int(eq_cfg.get('warmup', 1600))
        if ch_type in ('TV_AR1', 'TV-AR1', 'TVAR1'):
            Lhat = int(ch_cfg.get('L', 2))
            h_hat = _ls_channel_estimate(y[:warm], x[:warm], Lhat)
        else:
            h_hat = h_used
        s_hat = viterbi_mlsd(y, h_hat, modulation='qpsk')
        _, bits_hat = hard_slicer(s_hat, 'qpsk')

    elif eq_type == 'mmse_le':
        Lw = int(eq_cfg.get('Lw', 7))
        delay = eq_cfg.get('delay', None)
        pre, w, dly = mmse_le(y, h_used, noise_var, Lw=Lw, delay=delay)
        s_hat, bits_hat = hard_slicer(pre, 'qpsk')
        pre_slicer_signal_for_snr = pre

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
        # The true pre-slicer signal in a DFE includes the feedback.
        # This implementation's `zf_dfe_detect` doesn't seem to return it, so we use `pre` as is.
        pre_slicer_signal_for_snr = pre

    elif eq_type == 'mmse_dfe':
        Lw = int(eq_cfg.get('Lw', 11))
        delay = int(eq_cfg.get('delay', 1))
        Nb = int(eq_cfg.get('Nb', 1))
        warm = int(eq_cfg.get('warmup', 500))
        gate = eq_cfg.get('gate_tau', None)
        gate = float(gate) if gate is not None else None
        lam = float(eq_cfg.get('lambda', 0.0))
        y_train, s_train = y[:warm], x[:warm]
        w, b, dly = mmse_dfe_design_train(y_train, s_train, Lw=Lw, Nb=Nb, delay=delay, lam=lam)
        def slicer_one(z):
            s, _ = hard_slicer(np.array([z]), 'qpsk'); return s[0]
        pre, s_hat, pre_fb = mmse_dfe_detect(y, w, b, dly, slicer_one, s_true=x, warmup=warm, gate_tau=gate)
        _, bits_hat = hard_slicer(s_hat, 'qpsk')
        pre_slicer_signal_for_snr = pre_fb

    elif eq_type == 'mmse_dfe_ca':
        Lw = int(eq_cfg.get('Lw', 13))
        delay = int(eq_cfg.get('delay', 1))
        Nb = int(eq_cfg.get('Nb', 1))
        warm = int(eq_cfg.get('warmup', 500))
        lam = float(eq_cfg.get('lambda', 0.0))
        iters = int(eq_cfg.get('iterations', 2))
        ldpc_it = int(eq_cfg.get('ldpc_iters', 50))
        alpha_ldpc = float(eq_cfg.get('alpha', 0.9)) 

        # --- Train DFE on pilots, then turbo iterations with LDPC ---
        y_train, s_train = y[:warm], x[:warm]
        w, b, dly = mmse_dfe_design_train(y_train, s_train, Lw=Lw, Nb=Nb, delay=delay, lam=lam)
        apriori_code = np.zeros(n, dtype=float)
        soft_seq = None
        for _ in range(iters):
            def slicer_one(z):
                s, _ = hard_slicer(np.array([z]), 'qpsk'); return s[0]
            pre, s_hat, pre_fb = mmse_dfe_detect(
                y, w, b, dly, slicer_one, s_true=x, warmup=warm, gate_tau=None, soft_seq=soft_seq
            )
            sigma2_eff = noise_var * float(np.vdot(w, w).real) + 1e-12
            LI, LQ = qpsk_llrs(pre_fb, sigma2_eff)
            L_ch_eq = np.empty(n, dtype=float); L_ch_eq[0::2] = LI; L_ch_eq[1::2] = LQ
            L_ch_dec = deinterleave(L_ch_eq, pi)
            L_post_dec = L_ch_dec + apriori_code
            L_msg_post, L_ext_code = code.decode_extrinsic(
                L_post_dec, iters=ldpc_it, mode="nms", alpha=0.85, damping=0.0, early_stop=True
            )
            L_ext_eq = interleave(L_ext_code, pi)
            LI_next, LQ_next = L_ext_eq[0::2], L_ext_eq[1::2]
            soft_seq = soft_symbol_from_llrs(LI_next, LQ_next)
            apriori_code = L_ext_code
        L_post_dec = deinterleave(L_ch_eq, pi) + apriori_code
        L_msg_post, _ = code.decode_extrinsic(
            L_post_dec, iters=ldpc_it, mode="nms", alpha=0.85, damping=0.0, early_stop=True
        )
        bits_msg_hat = (L_msg_post < 0).astype(int)
        bits_to_check, bits_hat_to_check = bits_msg, bits_msg_hat
        pre_slicer_signal_for_snr = pre_fb

    elif eq_type == 'rbpf_ca':
        print("[rbpf] patch active")
        # --- RBPF hyperparams & model selection ---
        Np       = int(eq_cfg.get('Np', 256))
        warm     = int(eq_cfg.get('warmup', 800))
        iters    = int(eq_cfg.get('iterations', 3))
        ess_thr  = float(eq_cfg.get('ess_thresh', 0.5))
        ldpc_it  = int(eq_cfg.get('ldpc_iters', 80))

        model = str(eq_cfg.get('model', 'ar1')).lower()
        if model in ('ar1',):
            model_kwargs = dict(
                rho   = float(eq_cfg.get('ar_rho', 0.999)),
                q_var = float(eq_cfg.get('q_var', 1e-6)),
            )
        elif model in ('matern32','m32','matern_32'):
            model_kwargs = dict(
                ell      = float(eq_cfg.get('ell', 12.0)),
                sigma_h2 = float(eq_cfg.get('sigma_h2', 1.0)),
                dt       = float(eq_cfg.get('dt', 1.0)),
                q_scale  = float(eq_cfg.get('q_scale', 1.0)),
            )
        else:
            raise ValueError(f"Unknown RBPF state model '{model}'")

        # soft LLR knobs
        prior_scale = float(eq_cfg.get('prior_scale', 1.0))  # DEC->EQ damping
        llr_scale   = float(eq_cfg.get('llr_scale',   1.0))  # EQ->DEC scaling

        # RBPF channel length
        Lch = len(h) if h is not None else int(ch_cfg.get('L', 2))

        # --- Turbo EQ <-> LDPC (EXTRINSIC schedule) ---
        apriori_code  = np.zeros(n, dtype=float)
        L_ext_eq_last = None
        soft_seq = np.zeros_like(x, dtype=np.complex128)

        # pilots (EQ domain)
        pilot_bits_eq = bits_tx[: 2*warm] if warm > 0 else None
        max_lag = int(eq_cfg.get('calib_max_lag', 6))
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

            # -------- Robust pilot alignment (delay only, first pass) --------
            if (t == 0) and (warm > 0) and (not did_delay_align):
                # integer symbol delay via soft-symbol correlation on pilots
                xp = x[:warm]                 # known pilot symbols
                sp = soft_seq[:warm]          # RBPF soft symbols on pilot span
                max_lag_sym = min(max_lag, warm - 1)
                best_d, best_score = 0, -1e18
                for d in range(-max_lag_sym, max_lag_sym + 1):
                    if d >= 0:
                        s_sub = sp[d:warm]
                        x_sub = xp[: warm - d]
                    else:
                        s_sub = sp[: warm + d]
                        x_sub = xp[-d:warm]
                    if len(s_sub) == 0:
                        continue
                    score = float(np.real(np.vdot(x_sub, s_sub)))
                    if score > best_score:
                        best_score, best_d = score, d
                if best_d != 0:
                    L_ext_eq = np.roll(L_ext_eq, 2 * best_d)  # 2 bits / QPSK symbol
                    soft_seq = np.roll(soft_seq, best_d)
                    print(f"[calib] Aligned delay: d={best_d:+d} symbols (score={best_score:.3f}).")
                did_delay_align = True

                # pilot-lane agreement diagnostic after delay
                if warm > 0:
                    pb  = pilot_bits_eq.astype(float)
                    sI  = 1.0 - 2.0 * pb[0::2]
                    sQ  = 1.0 - 2.0 * pb[1::2]
                    Lp  = L_ext_eq[: 2*warm].astype(float)
                    scI = float(np.mean(Lp[0::2]*sI))
                    scQ = float(np.mean(Lp[1::2]*sQ))
                    print(f"[calib] Pilot agreement after fix: I={scI:.3f}, Q={scQ:.3f}.")

            # --- Posterior->extrinsic correction if RBPF returned posterior ---
            if warm > 0:
                pb  = pilot_bits_eq.astype(float)
                sI  = 1.0 - 2.0 * pb[0::2]
                sQ  = 1.0 - 2.0 * pb[1::2]
                Lp_cur = L_ext_eq[: 2*warm].astype(float)
                sc_cur = float(np.mean(Lp_cur[0::2]*sI) + np.mean(Lp_cur[1::2]*sQ))
                apr_eq = apriori_eq[: 2*warm].astype(float)
                Lp_try = (Lp_cur - apr_eq)
                sc_try = float(np.mean(Lp_try[0::2]*sI) + np.mean(Lp_try[1::2]*sQ))
                if sc_try > sc_cur + 1e-6:
                    L_ext_eq = L_ext_eq - apriori_eq
                    print(f"[rbpf] Using posterior->extrinsic correction (pilot score {sc_try:.3f} > {sc_cur:.3f}).")

            # pilot BER diagnostic after calibration
            if t == 0 and warm > 0:
                Lp_bits = (L_ext_eq[: 2*warm] < 0).astype(int)
                pber = float(np.mean(Lp_bits != bits_tx[: 2*warm]))
                print(f"[diag] Pilot bit-error rate (EQ-domain): {pber:.3f}")

            L_ext_eq_last = L_ext_eq

            # EQ -> DEC: deinterleave, normalize, scale, clip (stronger than 1.6/±7.5)
            L_ext_dec = deinterleave(L_ext_eq, pi)
            std0 = float(np.std(L_ext_dec)) + 1e-9
            target_std = 2.2
            L_ext_dec *= (target_std / std0)
            if llr_scale != 1.0:
                L_ext_dec *= llr_scale
            np.clip(L_ext_dec, -12.0, 12.0, out=L_ext_dec)

            # health prints (first two iters)
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

        # ---- Final LDPC decode----
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
        std0 = float(np.std(L_post_dec)) + 1e-9
        target_std = 2.2
        L_post_dec *= (target_std / std0)
        if llr_scale != 1.0:
            L_post_dec *= llr_scale
        np.clip(L_post_dec, -12.0, 12.0, out=L_post_dec)
        L_post_dec += apriori_code

        L_msg_post, _ = code.decode_extrinsic(
            L_post_dec, iters=ldpc_it, mode="nms", alpha=0.85, damping=0.0, early_stop=True
        )
        bits_msg_hat = (L_msg_post < 0).astype(int)

        # hand results to unified metrics/prints
        bits_to_check      = bits_msg
        bits_hat_to_check  = bits_msg_hat
        pre_slicer_signal_for_snr = soft_seq

    else:
        raise NotImplementedError(f'Unknown equalizer type {eq_type}')

    if bits_to_check is None and 'bits_hat' in locals():
        bits_to_check = bits[2*warm:] # Factor of 2 for QPSK
        bits_hat_to_check = bits_hat[warm:]

    # ---- metrics (unified for all branches) ----
    print(f'--- Run complete for EQ: {eq_type} ---')
    print(f'Channel: {ch_type}, SNR(dB): {snr_db:.2f}, Symbols: {len(x)}')

    if bits_to_check is not None and bits_hat_to_check is not None:
        min_len = min(len(bits_to_check), len(bits_hat_to_check))
        b = ber(bits_to_check[:min_len], bits_hat_to_check[:min_len])
        ber_type = "Message BER" if eq_type in ('mmse_dfe_ca', 'rbpf_ca') else "BER"
        print(f'{ber_type}: {b:.4e}')
    else:
        print('BER: n/a')

    snr_mfb_lin = float('nan')
    if h_used is not None and np.ndim(h_used) == 1:
        snr_mfb_lin = matched_filter_bound(h_used, noise_var, Es=Es)

    if np.isnan(snr_mfb_lin):
        print('MFB (dB): n/a (not applicable for time-varying channels)')
    else:
        print(f'MFB (dB):   {10 * np.log10(snr_mfb_lin + 1e-15):.2f}')

    if pre_slicer_signal_for_snr is not None:
        N = min(len(x), len(pre_slicer_signal_for_snr))
        snr_r_lin = snr_receiver(x[warm:N], pre_slicer_signal_for_snr[warm:N])
        snr_r_db = 10 * np.log10(snr_r_lin + 1e-15)
        print(f'SNR_R (dB): {snr_r_db:.2f}')
        if not np.isnan(snr_mfb_lin):
            gap_db = gap_to_mfb_db(snr_r_lin, snr_mfb_lin)
            print(f'Gap to MFB (dB): {gap_db:.2f}')
    else:
        print('SNR_R (dB): n/a (no linear pre-slicer output)')

if __name__ == '__main__':
    main()