import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np
from itertools import product

def qpsk_constellation():
    a = 1/np.sqrt(2)
    return np.array([ a+1j*a,  a-1j*a, -a+1j*a, -a-1j*a ], dtype=np.complex128)

def _build_states(M, Lm1):
    if Lm1 == 0:
        return [()], {(): 0}
    states = list(product(range(M), repeat=Lm1))
    idx = {tuple(s): i for i, s in enumerate(states)}
    return states, idx

def viterbi_mlsd(y, h, modulation='qpsk'):
    y = np.asarray(y, dtype=np.complex128)
    h = np.asarray(h, dtype=np.complex128)
    L = len(h); Lm1 = max(0, L-1)

    if modulation.lower() != 'qpsk':
        raise NotImplementedError("Only QPSK wired.")
    S = qpsk_constellation(); M = len(S)

    states, state_to_idx = _build_states(M, Lm1)
    n_states = len(states) if Lm1 > 0 else 1
    N = len(y)

    INF = 1e300
    pm = np.zeros(n_states)
    bp_state = np.full((N, n_states), -1, dtype=int)
    bp_sym   = np.full((N, n_states), -1, dtype=int)

    for t in range(N):
        new_pm = np.full(n_states, INF)
        new_bs = np.full(n_states, -1, dtype=int)
        new_ba = np.full(n_states, -1, dtype=int)

        for ps in range(n_states):
            prev = states[ps] if Lm1 > 0 else ()
            for a_idx in range(M):
                a = S[a_idx]
                if Lm1 > 0:
                    nxt = (prev[1:] + (a_idx,)) if Lm1 > 1 else (a_idx,)
                    ns = state_to_idx[nxt]
                else:
                    ns = 0
                yhat = h[0]*a
                for k in range(1, L):
                    if Lm1 == 0: break
                    s_idx = prev[-k] if k <= Lm1 else None
                    if s_idx is not None:
                        yhat += h[k] * S[s_idx]
                bm = np.abs(y[t] - yhat)**2
                cand = pm[ps] + bm
                if cand < new_pm[ns]:
                    new_pm[ns] = cand
                    new_bs[ns] = ps
                    new_ba[ns] = a_idx

        pm = new_pm
        bp_state[t, :] = new_bs
        bp_sym[t, :]   = new_ba

    end_state = int(np.argmin(pm))
    sym_idx = np.empty(N, dtype=int)
    s = end_state
    for t in range(N-1, -1, -1):
        sym_idx[t] = bp_sym[t, s]
        s = bp_state[t, s] if t > 0 else s

    return S[sym_idx]
