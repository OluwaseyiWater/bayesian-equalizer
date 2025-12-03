import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import yaml
import numpy as np
from utils.modem import qpsk_mod, es_of_constellation
from metrics.ber_fer import ber
from metrics.snrr_mfb import snr_receiver, matched_filter_bound, gap_to_mfb_db
from channel_codes.qcldpc import QCLDPCCode
from utils.interleave import make_block_interleaver, interleave, deinterleave
from dp_gp.interface import rbpf_detect
from channels.time_varying import tv_ar1
import inspect

# This is the diagnostic print
if 'rbpf_detect' in globals():
    print(f"--> [DIAGNOSTIC] Running rbpf_detect from: {inspect.getfile(rbpf_detect)}")

def _ls_channel_estimate(y_pil, x_pil, L):
    y_pil = np.asarray(y_pil, dtype=np.complex128); x_pil = np.asarray(x_pil, dtype=np.complex128)
    N = len(y_pil)
    X = np.zeros((N, L), dtype=np.complex128)
    for k in range(L): X[k:, k] = x_pil[:N-k]
    h_ls, *_ = np.linalg.lstsq(X, y_pil, rcond=None)
    return h_ls

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f: cfg = yaml.safe_load(f)

    rng = np.random.default_rng(cfg.get('seed', 0))
    snr_db = float(cfg.get('snr_db', 8.0))
    eq_cfg = cfg.get('equalizer', {})
    eq_type = eq_cfg.get('type').lower()
    
    # ---- Channel code setup ----
    ldpc_k_config = int(eq_cfg.get('ldpc_k', 8000))
    code = QCLDPCCode(k=ldpc_k_config)

    ldpc_k = code.k
    n = code.n

    # ---- TX side ----
    bits_msg = rng.integers(0, 2, size=ldpc_k_config)
    bits_tx_coded = code.encode(bits_msg)
    pi = make_block_interleaver(n, 256)
    bits_tx = interleave(bits_tx_coded, pi)
    x = qpsk_mod(bits_tx)
    Es = es_of_constellation('qpsk')
    h_used = None

    # ---- Channel simulation ----
    noise_var_scaling = (ldpc_k / n)
    noise_var = Es / (2 * (10**(snr_db / 10.0))) * noise_var_scaling
    noise = np.sqrt(noise_var) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    
    ch_cfg = cfg.get('channel', {})
    ch_type = ch_cfg.get('type')
    if ch_type == 'TV_AR1':
        _, y = tv_ar1(x, noise, **ch_cfg, seed=cfg.get('seed', 0))
        h_used = None
    else:
        raise ValueError(f"This script is configured for TV_AR1 channel, but got {ch_type}")

    # ---- RX side: RBPF ----
    if eq_type != 'rbpf_ca':
        raise ValueError("This script is configured for 'rbpf_ca' equalizer.")

    Np, warm, iters, ess_thr, ldpc_it = (int(eq_cfg.get(p, d)) for p, d in [('Np', 1024), ('warmup', 1200), ('iterations', 8), ('ess_thresh', 0.6), ('ldpc_iters', 80)])
    model = str(eq_cfg.get('model', 'ar1')).lower()
    model_kwargs = {'rho': float(eq_cfg.get('ar_rho', 0.9995)), 'q_var': float(eq_cfg.get('q_var', 1e-6))}
    prior_scale = float(eq_cfg.get('prior_scale', 0.8))
    Lch = int(ch_cfg.get('L', 2))
    
    apriori_code = np.zeros(n, dtype=float)
    
    m0, P0 = None, None
    if warm > 0:
        print("[rbpf] Performing pilot-based warm start for channel state.")
        h_ls = _ls_channel_estimate(y[:warm], x[:warm], Lch)
        if model == 'ar1': m0, state_dim = h_ls, Lch
        else: state_dim = 2 * Lch; m0 = np.zeros(state_dim, dtype=np.complex128); m0[0::2] = h_ls
        P0 = (1e-3) * np.eye(state_dim, dtype=np.complex128)
        
    did_calibrate, lane_swap, signI, signQ, delay_sym = False, False, 1.0, 1.0, 0
    max_lag = int(eq_cfg.get('calib_max_lag', 4))
    ldpc_it_fast = min(20, ldpc_it)
    
    for t in range(iters):
        apriori_eq = interleave(apriori_code, pi) * prior_scale
        L_post_eq, soft_seq, aux = rbpf_detect(y=y, L=Lch, sigma_v2=noise_var, Np=Np, model=model, model_kwargs=model_kwargs, apriori_llr_bits=apriori_eq, pilot_sym=x, pilot_len=warm, ess_thresh=ess_thr, seed=cfg.get('seed', 0) + t, m0=m0, P0=P0)
        
        if t == 0 and warm > 0:
            pb = bits_tx[:2*warm].astype(float); sI, sQ = 1-2*pb[0::2], 1-2*pb[1::2]
            xp, sp = x[:warm], soft_seq[:warm]
            best_d, best_score = 0, -1e18
            for d in range(-max_lag, max_lag + 1):
                if d >= 0: s_sub, x_sub = sp[d:warm], xp[:warm-d]
                else: s_sub, x_sub = sp[:warm+d], xp[-d:warm]
                if not len(s_sub): continue
                score = float(np.real(np.vdot(x_sub, s_sub)))
                if score > best_score: best_score, best_d = score, d
            delay_sym = best_d
            if delay_sym != 0: L_post_eq = np.roll(L_post_eq, 2*delay_sym)
            
            Lp = L_post_eq[:2*warm]
            def score_combo(Lp, swap, fI, fQ):
                Le, Lo = (Lp[1::2], Lp[0::2]) if swap else (Lp[0::2], Lp[1::2])
                return np.mean((1-2*fI)*Le*sI) + np.mean((1-2*fQ)*Lo*sQ)
            
            best = (-1e18, False, False, False)
            for swap, fI, fQ in [(s, i, q) for s in [False, True] for i in [False, True] for q in [False, True]]:
                sc = score_combo(Lp, swap, fI, fQ)
                if sc > best[0]: best = (sc, swap, fI, fQ)
            
            _, lane_swap, flipI, flipQ = best
            signI, signQ = -1.0 if flipI else 1.0, -1.0 if flipQ else 1.0
            did_calibrate = True
        
        if did_calibrate:
            if delay_sym != 0: L_post_eq = np.roll(L_post_eq, 2*delay_sym)
            Le_full, Lo_full = L_post_eq[0::2].copy(), L_post_eq[1::2].copy()
            if signI < 0: Le_full *= -1.0
            if signQ < 0: Lo_full *= -1.0
            L_post_eq[0::2], L_post_eq[1::2] = (Lo_full, Le_full) if lane_swap else (Le_full, Lo_full)
        
        L_extr_eq = L_post_eq if t == 0 else L_post_eq - apriori_eq
        L_ext_dec = deinterleave(L_extr_eq, pi)
        L_ext_dec *= (3.5 / (np.std(L_ext_dec) + 1e-9))
        np.clip(L_ext_dec, -14.0, 14.0, out=L_ext_dec)
        
        L_post_dec = L_ext_dec + apriori_code
        it_this = ldpc_it_fast if t < iters -1 else ldpc_it
        L_msg_post, apriori_code = code.decode_extrinsic(L_post_dec, iters=it_this, mode="nms", alpha=0.85)

    bits_msg_hat = (L_msg_post < 0).astype(int)
    bits_to_check = bits_msg
    bits_hat_to_check = bits_msg_hat[:len(bits_msg)]

    # ---- metrics ----
    print(f'--- Run complete for EQ: {eq_type} ---')
    print(f'Channel: {ch_type}, SNR(dB): {snr_db:.2f}, Symbols: {len(x)}')
    b = ber(bits_to_check, bits_hat_to_check)
    print(f'Message BER: {b:.4e}')
    
    snr_r_lin = snr_receiver(x, soft_seq)
    print(f'SNR_R (dB): {10*np.log10(snr_r_lin):.2f}') if not np.isnan(snr_r_lin) else print('SNR_R (dB): n/a')

if __name__ == '__main__':
    main()