import numpy as np
from .ssm import build_state_model, predict, update

SQRT2 = np.sqrt(2.0)

def systematic_resample(w, rng):
    """
    Systematic resampling for particle filters.
    w: (Np,) nonnegative, sum to 1
    returns integer indices of length Np
    """
    N = w.size
    cdf = np.cumsum(w)
    u0 = rng.uniform(0.0, 1.0 / N)
    u = u0 + (np.arange(N) / N)
    return np.searchsorted(cdf, u, side="right")


class RBPF:
    """
    Rao-Blackwellized Particle Filter for ISI with linear-Gaussian tap dynamics.

    - State space is built by `build_state_model(model, L, **model_kwargs)`.
      For AR(1): state_dim = L, layout is [h0,...,h_{L-1}]
      For Matérn-3/2: state_dim = 2L, layout is [h0, h0(z^-1), h1, h1(z^-1), ...]

    - Measurement at time n for candidate symbol s and ISI memory mem:
        y[n] = phi^T h + v,  where phi depends on (s, mem) and is expanded
        to the chosen state layout via expand_phi(...).

    - This RBPF outputs POSTERIOR bit LLRs (decoder a-priori is consumed inside
      the filter). Your interface should subtract the a-priori to form EXTRINSIC.
    """

    def __init__(self, L, Np=128, noise_var=1e-2, model="ar1", model_kwargs=None, rng=None):
        self.L = int(L)
        self.Np = int(Np)
        self.noise_var = float(noise_var)
        self.rng = np.random.default_rng() if rng is None else rng

        # QPSK symbols (Gray mapping): (+,+), (+,-), (-,+), (-,-)
        self.syms = np.array([(1+1j), (1-1j), (-1+1j), (-1-1j)], dtype=np.complex128) / SQRT2

        #linear-Gaussian state model and regressor expander
        self.F, self.Q, self.state_dim, self.expand_phi = build_state_model(
            model, self.L, **(model_kwargs or {})
        )

    def run(self, y, pri_llr_bits=None, pilots=None, pilot_len=0, ess_thresh=0.5, m0=None, P0=None):
        """
        Run the RBPF forward pass.

        Parameters
        ----------
        y            : (N,) complex np.ndarray, received symbols
        pri_llr_bits : (2N,) float or None, decoder a-priori LLRs (EQ order)
        pilots       : (N,) complex or None, transmitted symbols (pilots in front)
        pilot_len    : int, number of pilot symbols at the start
        ess_thresh   : float in (0,1], resample when ESS < ess_thresh * Np
        m0, P0       : optional initial mean/covariance in the chosen state space

        Returns
        -------
        L_post_bits  : (2N,) float, posterior bit LLRs in EQ order (NOT extrinsic)
        soft_seq     : (N,) complex, E[s_n | y_1:n] (diagnostic)
        aux          : dict, e.g., {"ess": array of ESS per step}
        """
        y = np.asarray(y, dtype=np.complex128)
        N = y.size
        L = self.L
        D = self.state_dim
        rng = self.rng
        nv = self.noise_var

        # Init particle means/covs in state space
        if (m0 is not None) and (P0 is not None):
            m = np.tile(np.asarray(m0, dtype=np.complex128).ravel()[None, :], (self.Np, 1))
            P = np.tile(np.asarray(P0, dtype=np.complex128)[None, :, :], (self.Np, 1, 1))
        else:
            m = np.zeros((self.Np, D), dtype=np.complex128)
            P = np.tile(10.0 * np.eye(D, dtype=np.complex128)[None, :, :], (self.Np, 1, 1))

        # ISI memory in symbol space (length L-1)
        mem = np.zeros((self.Np, max(0, L-1)), dtype=np.complex128)

        # Particle weights
        w = np.ones(self.Np, dtype=float) / self.Np

        # Outputs
        soft_seq = np.zeros(N, dtype=np.complex128)
        LLR = np.zeros(2 * N, dtype=float)
        ess_hist = []

        # Helper for bit priors from LLRs (Gray mapping)
        def symbol_prior_from_llrs(LI, LQ):
            # P(b=+1) = 1 / (1 + e^{-L})
            pI_plus = 1.0 / (1.0 + np.exp(-LI))
            pQ_plus = 1.0 / (1.0 + np.exp(-LQ))
            # syms order: s0=(+,+), s1=(+,-), s2=(-,+), s3=(-,-)
            p = np.array([
                pI_plus * pQ_plus,
                pI_plus * (1.0 - pQ_plus),
                (1.0 - pI_plus) * pQ_plus,
                (1.0 - pI_plus) * (1.0 - pQ_plus)
            ], dtype=float)
            s = p.sum()
            return p / (s + 1e-300)

        for n in range(N):
            # 1) Predict each particle's linear-Gaussian state
            for i in range(self.Np):
                m[i], P[i] = predict(m[i], P[i], self.F, self.Q)

            # 2) Build symbol prior p_s (from decoder LLRs or pilots)
            if (pri_llr_bits is not None) and (2*n + 1 < len(pri_llr_bits)):
                LI = float(pri_llr_bits[2*n])
                LQ = float(pri_llr_bits[2*n + 1])
                p_s = symbol_prior_from_llrs(LI, LQ)
            else:
                p_s = np.full(4, 0.25, dtype=float)

            use_pil = (n < int(pilot_len)) and (pilots is not None)
            if use_pil:
                # Override with one-hot prior on the known pilot symbol
                diffs = np.abs(self.syms - pilots[n])
                idx_true = int(np.argmin(diffs))
                p_s = np.zeros(4, dtype=float)
                p_s[idx_true] = 1.0

            # 3) Per-particle, per-symbol likelihoods p(y_n | s, particle i)
            like_i_s = np.zeros((self.Np, 4), dtype=float)
            temp = 1.2 
            


            for i in range(self.Np):
                for s_idx, s in enumerate(self.syms):
                    # Construct phi over taps in symbol space
                    if L > 0:
                        phi_L = np.empty(L, dtype=np.complex128)
                        phi_L[0] = s
                        if L > 1:
                            phi_L[1:] = mem[i, :L-1]
                    else:
                        phi_L = np.zeros(0, dtype=np.complex128)

                    phi = self.expand_phi(phi_L)
                    phi_col = phi.reshape(-1, 1)

                    # Predictive variance and residual
                    S = (phi_col.T @ P[i] @ phi_col.conj()).real.item() + nv
                    mu = (phi_col.T @ m[i].reshape(-1, 1)).item()
                    e = y[n] - mu

                    # Complex Gaussian likelihood (scalar)
                    like = np.exp(- (np.abs(e) ** 2) / S) / (np.pi * S)
                    like_i_s[i, s_idx] = float(like)

            if temp != 1.0:
                like_i_s **= temp

            # 4) Posterior over s_n: mixture across particles WITH prior p_s
            num = (w[:, None] * like_i_s) * p_s[None, :]
            post_s = num.sum(axis=0)
            post_s /= (post_s.sum() + 1e-300)

            # 5) Soft symbol and POSTERIOR bit LLRs (EQ order)
            soft_seq[n] = np.dot(post_s, self.syms)
            # LLRs from post_s (Gray mapping)
            pI_pos = post_s[0] + post_s[1]  # I=+1
            pI_neg = post_s[2] + post_s[3]  # I=-1
            pQ_pos = post_s[0] + post_s[2]  # Q=+1
            pQ_neg = post_s[1] + post_s[3]  # Q=-1
            LLR[2*n]   = np.log((pI_pos + 1e-300) / (pI_neg + 1e-300))
            LLR[2*n+1] = np.log((pQ_pos + 1e-300) / (pQ_neg + 1e-300))

            # 6) Particle weight update: multiply by prior-weighted likelihood
            #    inc_i = sum_s like_i_s[i, s] * p_s[s]
            inc = like_i_s @ p_s
            w *= inc
            w /= (w.sum() + 1e-300)

            # 7) MAP symbol advance per particle (low-variance), then KF update
            for i in range(self.Np):
                p_i = like_i_s[i] * p_s
                p_i /= (p_i.sum() + 1e-300)
                s_idx = self.rng.choice(4, p=p_i)
                s = self.syms[s_idx]


                # Measurement vector for chosen s
                if L > 0:
                    phi_L = np.empty(L, dtype=np.complex128)
                    phi_L[0] = s
                    if L > 1:
                        phi_L[1:] = mem[i, :L-1]
                else:
                    phi_L = np.zeros(0, dtype=np.complex128)
                phi = self.expand_phi(phi_L)

                # Kalman measurement update
                m[i], P[i], _, _ = update(m[i], P[i], phi, y[n], nv)

                # Roll ISI memory
                if L > 1:
                    mem[i, 1:] = mem[i, :-1]
                    mem[i, 0]  = s

            # 8) ESS check & resample (+ tiny covariance inflation)
            ess = 1.0 / (np.sum(w * w) + 1e-300)
            ess_hist.append(ess)
            if ess < float(ess_thresh) * self.Np:
                idx = systematic_resample(w, rng)
                m = m[idx]
                P = P[idx]
                mem = mem[idx]
                w = np.ones_like(w) / self.Np
                # small inflation to avoid degeneracy after resampling
                P += (1e-6) * np.eye(D, dtype=np.complex128)[None, :, :]

        aux = {"ess": np.asarray(ess_hist, dtype=float)}
        return LLR, soft_seq, aux
