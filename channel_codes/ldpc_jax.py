import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np

"""
JAX-flavored LDPC with the SAME public API as channel_codes.ldpc.LDPCCode.
We keep Python loops for clarity/compatibility (so jitting is optional),
but store/operate in jnp where helpful. If JAX is unavailable, this file
should not be imported (callers already try/except and fall back to ldpc.py).
"""

import jax
import jax.numpy as jnp

LLR_MAX = 24.0


def _clip(x):
    return jnp.clip(x, -LLR_MAX, LLR_MAX)


def _build_systematic_ldpc(k=8000, dv=3, seed=0):
    """
    Identical construction to the NumPy version:
      H = [A | I_k], n=2k, m=k, rate=1/2
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


class LDPCJAX:
    """
    Same interface as LDPCCode, but with jax.numpy arrays internally.
    """
    def __init__(self, k=8000, dv=3, seed=0, alpha=0.85):
        self.k = int(k)
        self.n = 2 * self.k
        self.m = self.k
        H_np, A_np = _build_systematic_ldpc(self.k, dv=dv, seed=seed)
        self.H = jnp.array(H_np, dtype=jnp.uint8)
        self.A = jnp.array(A_np, dtype=jnp.uint8)

        self.alpha_default = float(alpha)
        H = H_np
        self.vars_of_check = [np.nonzero(H[i, :])[0].astype(np.int32) for i in range(self.m)]
        self.checks_of_var = [np.nonzero(H[:, j])[0].astype(np.int32) for j in range(self.n)]

    # ---------- encoder ----------
    def encode(self, msg_bits):
        """
        msg_bits: length k array in {0,1}
        returns np.uint8[n] codeword [m | p]
        (We use NumPy for parity mod-2 arithmetic for simplicity.)
        """
        m = (np.asarray(msg_bits, dtype=np.uint8) & 1)
        if m.size != self.k:
            raise ValueError(f"LDPCJAX.encode: msg length {m.size} != k={self.k}")
        p = (np.asarray(self.A) @ m) & 1
        x = np.concatenate([m, p], axis=0).astype(np.uint8)
        return x

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
        Normalized min-sum (alpha-scaled) message passing.
        Returns (L_post_msg[:k], L_ext[:n]).
        Python loops (easy to read/maintain); data moved as jnp where convenient.
        """
        if mode.lower() != "nms":
            raise NotImplementedError("LDPCJAX: only mode='nms' implemented.")
        a = float(self.alpha_default if alpha is None else alpha)
        dmp = float(damping)

        n, m = self.n, self.m
        L_a = jnp.array(L_apriori, dtype=jnp.float32)
        if L_a.size != n:
            raise ValueError(f"decode_extrinsic: L_apriori length {L_a.size} != n={n}")
        L_a = _clip(L_a)

        L_a_np = np.asarray(L_a)

        L_vc = [{int(ci): float(L_a_np[j]) for ci in self.checks_of_var[j]} for j in range(n)]
        L_cv = [{int(vj): 0.0 for vj in self.vars_of_check[i]} for i in range(m)]

        H_np = np.asarray(self.H)

        for it in range(int(iters)):
            # ----- check update -----
            for i in range(m):
                vs = self.vars_of_check[i]
                if vs.size == 0:
                    continue
                msgs = np.array([L_vc[v][i] for v in vs], dtype=float)
                msgs = np.clip(msgs, -LLR_MAX, LLR_MAX)

                sgn = np.sign(msgs); sgn[sgn == 0] = 1.0
                prod_sgn = np.prod(sgn)
                aabs = np.abs(msgs)

                idx_min = int(np.argmin(aabs))
                min1 = aabs[idx_min]
                tmp = aabs.copy(); tmp[idx_min] = np.inf
                min2 = float(np.min(tmp))

                for t, v in enumerate(vs):
                    mag = min2 if t == idx_min else min1
                    msg_new = a * prod_sgn * sgn[t] * mag
                    msg_new = float(np.clip(msg_new, -LLR_MAX, LLR_MAX))
                    if dmp > 0.0:
                        L_cv[i][int(v)] = (1 - dmp) * msg_new + dmp * L_cv[i][int(v)]
                    else:
                        L_cv[i][int(v)] = msg_new

            # ----- variable update -----
            for j in range(n):
                cs = self.checks_of_var[j]
                if cs.size == 0:
                    continue
                incoming = [np.clip(L_cv[ci][j], -LLR_MAX, LLR_MAX) for ci in cs]
                total = float(np.clip(L_a_np[j] + sum(incoming), -LLR_MAX, LLR_MAX))
                for ci in cs:
                    val = float(np.clip(total - np.clip(L_cv[ci][j], -LLR_MAX, LLR_MAX), -LLR_MAX, LLR_MAX))
                    if dmp > 0.0:
                        L_vc[j][int(ci)] = (1 - dmp) * val + dmp * L_vc[j][int(ci)]
                    else:
                        L_vc[j][int(ci)] = val

            # ----- a-posteriori & early stop -----
            L_post = np.zeros(n, dtype=float)
            for j in range(n):
                L_post[j] = float(np.clip(L_a_np[j] + sum(np.clip(L_cv[ci][j], -LLR_MAX, LLR_MAX)
                                                          for ci in self.checks_of_var[j]), -LLR_MAX, LLR_MAX))

            if early_stop:
                hard = (L_post < 0).astype(np.uint8)
                synd = (H_np @ hard) & 1
                if not np.any(synd):
                    break

        L_ext = np.clip(L_post - L_a_np, -LLR_MAX, LLR_MAX)
        return L_post[:self.k], L_ext
