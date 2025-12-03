# import numpy as np
# from itertools import product


# def qpsk_constellation():
#     """
#     QPSK constellation in *Gray order*:
#       bits 00 ->  +1 + j*1   ( + + )
#       bits 01 ->  +1 - j*1   ( + - )
#       bits 11 ->  -1 - j*1   ( - - )
#       bits 10 ->  -1 + j*1   ( - + )
#     All points are normalized to unit energy (1/sqrt(2) per component).
#     """
#     a = 1.0 / np.sqrt(2.0)
#     # Gray order: [++, +-, --, -+]
#     return np.array([a + 1j * a,  a - 1j * a,  -a - 1j * a,  -a + 1j * a], dtype=np.complex128)


# def _build_states(M: int, Lm1: int):
#     """
#     Build the trellis state space for a channel of memory L-1 (Lm1).
#     Each state is a tuple of length Lm1 containing *symbol indices* in [0..M-1].
#     Returns:
#         states: list[tuple[int]], len = M**Lm1
#         state_to_idx: dict[tuple[int] -> int]
#     """
#     if Lm1 == 0:
#         # Memoryless channel: single 'empty' state
#         states = [tuple()]
#         return states, {tuple(): 0}
#     states = list(product(range(M), repeat=Lm1))
#     state_to_idx = {st: i for i, st in enumerate(states)}
#     return states, state_to_idx


# def viterbi_mlsd(y, h, modulation: str = 'qpsk'):
#     """
#     Maximum-Likelihood Sequence Detection (Viterbi) for a linear ISI channel.

#     Args:
#         y : array-like, complex
#             Received samples at SYMBOL rate: y[t] = sum_{k=0..L-1} h[k] * s[t-k] + n[t]
#             with s[t] in the chosen constellation and s[t]=0 for t<0 (free-running init here).
#         h : array-like, complex
#             Channel impulse response (length L).
#         modulation : str
#             Only 'qpsk' is implemented in this version.

#     Returns:
#         s_hat : np.ndarray (complex128, shape (N,))
#             MLSD-decoded *symbols* (complex QPSK points in Gray order).
#             Map to bits with your existing hard slicer if needed.
#     """
#     # --- 0) Setup ---
#     y = np.asarray(y, dtype=np.complex128)
#     h = np.asarray(h, dtype=np.complex128)

#     if modulation.lower() != 'qpsk':
#         raise NotImplementedError("Only QPSK is implemented in this viterbi_mlsd.")

#     S = qpsk_constellation()     # constellation points in Gray order
#     M = len(S)                   # M = 4 for QPSK
#     L = int(len(h))              # channel length
#     Lm1 = max(L - 1, 0)
#     N = int(len(y))              # number of received symbols

#     # Trellis states encode the previous L-1 symbol indices
#     states, state_to_idx = _build_states(M, Lm1)
#     n_states = len(states)

#     # --- 1) Initialization (free-running) ---
#     # All states are equally likely at t=0
#     INF = 1e300
#     path_metrics = np.zeros(n_states, dtype=np.float64)
#     bp_state = np.zeros((N, n_states), dtype=np.int32)   # backpointer: previous state index
#     bp_sym   = np.zeros((N, n_states), dtype=np.int32)   # backpointer: decided *current* symbol index

#     # --- 2) Recursion (Add-Compare-Select) ---
#     # y_hat(t) = h[0]*s[t] + h[1]*s[t-1] + ... + h[L-1]*s[t-L+1]
#     # where s[t-k] for k>=1 are encoded in the 'prev state'
#     for t in range(N):
#         y_t = y[t]
#         next_path_metrics = np.full(n_states, INF, dtype=np.float64)

#         for prev_state_idx, prev_state_tuple in enumerate(states):
#             pm = path_metrics[prev_state_idx]
#             # If a path is impossible, skip it
#             if pm >= INF:
#                 continue

#             for sym_idx in range(M):
#                 # Determine destination state by shifting in the new symbol index
#                 if Lm1 > 0:
#                     # state tuple is (..., s_{t-2}, s_{t-1}); append current sym
#                     next_state_tuple = prev_state_tuple[1:] + (sym_idx,)
#                     next_state_idx = state_to_idx[next_state_tuple]
#                 else:
#                     # memoryless channel: single state
#                     next_state_idx = 0

#                 # Compute noiseless output y_hat for this branch
#                 # current symbol contribution
#                 y_hat = h[0] * S[sym_idx]
#                 # past symbols from the previous-state tuple
#                 # prev_state_tuple is ordered oldest..newest = (s_{t-L+1}, ..., s_{t-1})
#                 # we need h[1]*s_{t-1} + h[2]*s_{t-2} + ...
#                 for k in range(Lm1):
#                     # index from newest backward: prev_state_tuple[-1 - k] = s_{t-1-k}
#                     s_prev_idx = prev_state_tuple[-1 - k]
#                     y_hat += h[k + 1] * S[s_prev_idx]

#                 # Branch metric (AWGN, white)
#                 metric = pm + np.abs(y_t - y_hat)**2

#                 # ACS: keep best path into next_state_idx
#                 if metric < next_path_metrics[next_state_idx]:
#                     next_path_metrics[next_state_idx] = metric
#                     bp_state[t, next_state_idx] = prev_state_idx
#                     bp_sym[t,   next_state_idx] = sym_idx

#         path_metrics = next_path_metrics

#     # --- 3) Termination ---
#     # Choose the best final state
#     current_state_idx = int(np.argmin(path_metrics))

#     # --- 4) Backtracking ---
#     decoded_indices = np.empty(N, dtype=np.int32)
#     for t in range(N - 1, -1, -1):
#         decoded_indices[t] = bp_sym[t, current_state_idx]
#         current_state_idx = bp_state[t, current_state_idx]

#     # Map symbol indices to complex constellation points (Gray order)
#     s_hat = S[decoded_indices]
#     return s_hat

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
    # The first state from product() corresponds to the all-zeros symbol index state.
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
    
    # --- 1. Initialization ---
    # Path metrics start at infinity, except for the known start state (state 0).
    path_metrics = np.full(n_states, INF)
    if n_states > 0:
        path_metrics[0] = 0.0
    
    # Backpointers store the index of the best previous state and the symbol that led there.
    bp_state = np.zeros((N, n_states), dtype=int)
    bp_sym   = np.zeros((N, n_states), dtype=int)

    # --- 2. Recursion (Add-Compare-Select) ---
    for t, y_t in enumerate(y):
        new_path_metrics = np.full(n_states, INF)
        
        # For each possible PREVIOUS state at time t-1
        for prev_state_idx, prev_state_tuple in enumerate(states):
            
            # Skip paths that are already "infinitely" bad
            if path_metrics[prev_state_idx] == INF:
                continue
            
            # For each possible transition from this previous state
            for current_sym_idx in range(M):
                
                # Determine the destination state for this transition
                if Lm1 > 0:
                    next_state_tuple = prev_state_tuple[1:] + (current_sym_idx,)
                    next_state_idx = state_to_idx[next_state_tuple]
                else:
                    next_state_idx = 0
                
                # Calculate the expected, noise-free output (y_hat) for this transition
                y_hat = h[0] * S[current_sym_idx]
                if Lm1 > 0:
                    for i in range(Lm1):
                        y_hat += h[i+1] * S[prev_state_tuple[-(i+1)]]
                
                # Add: Calculate the new path's total metric
                branch_metric = np.abs(y_t - y_hat)**2
                total_metric = path_metrics[prev_state_idx] + branch_metric
                
                # Compare & Select: If this is the best path found so far to the next_state, update it.
                if total_metric < new_path_metrics[next_state_idx]:
                    new_path_metrics[next_state_idx] = total_metric
                    bp_state[t, next_state_idx] = prev_state_idx
                    bp_sym[t, next_state_idx] = current_sym_idx
        
        path_metrics = new_path_metrics

    # --- 3. Termination & 4. Backtracking ---
    sym_indices = np.empty(N, dtype=int)
    
    # Find the most likely final state
    current_state_idx = np.argmin(path_metrics)

    # Trace backward through the trellis using the stored backpointers
    for t in reversed(range(N)):
        sym_indices[t] = bp_sym[t, current_state_idx]
        current_state_idx = bp_state[t, current_state_idx]
        
    return S[sym_indices]