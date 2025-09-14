import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np

def encode_repeat(msg_bits, R=3):
    msg_bits = np.asarray(msg_bits).astype(int)
    return np.repeat(msg_bits, R)

def decode_repeat_extrinsic(apriori_llr, R=3):
    """
    Turbo-style sum-product for a repetition code (rate 1/R):
    - Input: a priori LLRs for *coded* bits (length N=R*M)
    - Output:
      * a posteriori LLRs for message bits (length M)
      * extrinsic LLRs per coded bit for *next* equalizer iteration (length N)
    """
    apriori_llr = np.asarray(apriori_llr, dtype=float)
    assert len(apriori_llr) % R == 0, "coded length not divisible by R"
    M = len(apriori_llr)//R
    L_groups = apriori_llr.reshape(M, R)

    # Posterior for message bit = sum of its replicas
    L_msg_post = np.sum(L_groups, axis=1)

    # Extrinsic for each replica = (sum of others)
    L_groups_ext = (L_msg_post[:,None] - L_groups)

    return L_msg_post, L_groups_ext.ravel()

def hard_from_llr(L):
    return (np.asarray(L) < 0).astype(int)
