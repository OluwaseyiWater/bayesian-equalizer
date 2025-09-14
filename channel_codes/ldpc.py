# channel_codes/ldpc.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np

"""
Plain NumPy LDPC (rate-1/2, systematic) with normalized min-sum (NMS) decoding.
API matches what's used in run_experiment.py / sweep_*:
  - class LDPCCode(k, dv, seed, alpha)
  - encode(msg_bits) -> np.uint8[n]  (n = 2k)
  - decode_extrinsic(L_apriori, iters=..., mode="nms", alpha=..., damping=..., early_stop=True)
      returns (L_post_msg, L_ext_bits)
    * L_post_msg: size k (a-posteriori LLRs of the first k = message bits)
    * L_ext_bits: size n (C->V extrinsic for all code bits)
"""

LLR_MAX = 24.0  # generous clip to avoid NaNs/Infs but keep stability


def _clip_arr(a, lim=LLR_MAX, out=None):
    return np.clip(a, -lim, lim, out=out)


def _build_systematic_ldpc(k=8000, dv=3, seed=0):
    """
    Build a very simple systematic LDPC: H = [A | I_k], n = 2k, m = k, rate = 1/2.
      - A is (k x k) sparse with ~dv ones per column (chosen uniformly without replacement).
      - Encoding is trivial: p = (A @ m) mod 2, x = [m | p].
    NOTE: This is NOT a standardized code; it's a light baseline suitable for simulations.
    """
    rng = np.random.default_rng(seed)
    m = k
    n = 2 * k

    A = np.zeros((m, k), dtype=np.uint8)
    for j in range(k):
        rows = rng.choice(m, size=dv, replace=False)
        A[rows, j] ^= 1

    H = np.hstack([A, np.eye(m, dtype=np.uint8)])
    return H, A


class LDPCCode:
    """
    Systematic rate-1/2 LDPC with normalized min-sum message passing.
    """
    def __init__(self, k=8000, dv=3, seed=0, alpha=0.85):
        self.k = int(k)
        self.n = 2 * self.k
        self.m = self.k
        self.H, self.A = _build_systematic_ldpc(self.k, dv=dv, seed=seed)
        self.alpha_default = float(alpha)

        # adjacency as compact integer arrays (for fast Python loops)
        H = self.H
        self.vars_of_check = [np.nonzero(H[i, :])[0].astype(np.int32) for i in range(self.m)]
        self.checks_of_var = [np.nonzero(H[:, j])[0].astype(np.int32) for j in range(self.n)]

    # ---------- encoder ----------
    def encode(self, msg_bits):
        """
        msg_bits: np.uint8/np.int array of length k in {0,1}
        returns codeword x = [m | p] with length n=2k
        """
        m = (np.asarray(msg_bits, dtype=np.uint8) & 1)
        if m.size != self.k:
            raise ValueError(f"LDPC.encode: msg length {m.size} != k={self.k}")
        # parity
        p = (self.A @ m) & 1
        return np.concatenate([m, p], axis=0).astype(np.uint8)

    # ---------- decoder ----------
    def decode_extrinsic(
        self,
        L_apriori,
        iters=50,
        mode="nms",
        alpha=None,
        damping=0.0,
        early_stop=True,
    ):
        """
        Normalized min-sum (alpha-scaled) belief-propagation decoder.

        Inputs:
          L_apriori : size n a-priori LLRs for the code bits
          iters     : max BP iterations
          mode      : "nms" (only one supported; others raise if requested)
          alpha     : NMS normalization factor in (0,1]. If None, use self.alpha_default.
          damping   : optional message damping factor in [0,1)
          early_stop: stop when syndrome == 0

        Returns:
          L_post_msg : size k a-posteriori LLRs for message bits (first k positions)
          L_ext_bits : size n extrinsic LLRs = L_post - L_apriori (for all code bits)
        """
        if mode.lower() != "nms":
            raise NotImplementedError("Only mode='nms' is supported in this LDPC baseline.")
        a = float(self.alpha_default if alpha is None else alpha)
        dmp = float(damping)

        n, m = self.n, self.m
        L_a = np.asarray(L_apriori, dtype=float)
        if L_a.size != n:
            raise ValueError(f"decode_extrinsic: L_apriori length {L_a.size} != n={n}")
        _clip_arr(L_a, out=L_a)

        # Initialize messages: dictionaries (check->var) and (var->check) stored sparsely
        L_vc = [{int(ci): float(L_a[j]) for ci in self.checks_of_var[j]} for j in range(n)]
        L_cv = [{int(vj): 0.0 for vj in self.vars_of_check[i]} for i in range(m)]

        for it in range(int(iters)):
            # ----- check node update (C -> V) -----
            for i in range(m):
                vs = self.vars_of_check[i]
                if vs.size == 0:
                    continue
                msgs = np.array([L_vc[v][i] for v in vs], dtype=float)
                _clip_arr(msgs, out=msgs)

                sgn = np.sign(msgs); sgn[sgn == 0] = 1.0
                prod_sgn = np.prod(sgn)
                aabs = np.abs(msgs)

                idx_min = int(np.argmin(aabs))
                min1 = aabs[idx_min]
                # second min
                tmp = aabs.copy(); tmp[idx_min] = np.inf
                min2 = float(np.min(tmp))

                # write outgoing messages
                for t, v in enumerate(vs):
                    mag = min2 if t == idx_min else min1
                    msg_new = a * prod_sgn * sgn[t] * mag
                    msg_new = float(np.clip(msg_new, -LLR_MAX, LLR_MAX))
                    if dmp > 0.0:
                        L_cv[i][int(v)] = (1 - dmp) * msg_new + dmp * L_cv[i][int(v)]
                    else:
                        L_cv[i][int(v)] = msg_new

            # ----- variable node update (V -> C) -----
            for j in range(n):
                cs = self.checks_of_var[j]
                if cs.size == 0:
                    continue
                incoming = [np.clip(L_cv[ci][j], -LLR_MAX, LLR_MAX) for ci in cs]
                total = float(np.clip(L_a[j] + sum(incoming), -LLR_MAX, LLR_MAX))
                for ci in cs:
                    # extrinsic to check ci
                    val = float(np.clip(total - np.clip(L_cv[ci][j], -LLR_MAX, LLR_MAX), -LLR_MAX, LLR_MAX))
                    if dmp > 0.0:
                        L_vc[j][int(ci)] = (1 - dmp) * val + dmp * L_vc[j][int(ci)]
                    else:
                        L_vc[j][int(ci)] = val

            # ----- a-posteriori and early stop -----
            L_post = np.zeros(n, dtype=float)
            for j in range(n):
                L_post[j] = float(np.clip(L_a[j] + sum(np.clip(L_cv[ci][j], -LLR_MAX, LLR_MAX)
                                                       for ci in self.checks_of_var[j]), -LLR_MAX, LLR_MAX))

            if early_stop:
                hard = (L_post < 0).astype(np.uint8)
                synd = (self.H @ hard) & 1
                if not np.any(synd):
                    # valid codeword estimate found
                    break

        L_ext = np.clip(L_post - L_a, -LLR_MAX, LLR_MAX)
        return L_post[:self.k], L_ext
