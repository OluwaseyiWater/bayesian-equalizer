import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__))) 
import numpy as np

def qpsk_llrs(z, sigma2):
    """
    LLRs per bit for unit-energy Gray QPSK used in this repo:
      b0 (I-bit): 0 if Re>0 else 1  → LLR_I = 2*Re(z)/sigma2
      b1 (Q-bit): 0 if Im>0 else 1  → LLR_Q = 2*Im(z)/sigma2
    Returns (LLR_I, LLR_Q) each shape (N,)
    """
    z = np.asarray(z)
    s2 = float(sigma2) + 1e-15
    return 2.0*np.real(z)/s2, 2.0*np.imag(z)/s2

def soft_symbol_from_llrs(LI, LQ):
    """
    Map per-bit LLRs to a soft complex symbol estimate:
      E[s | LLRs] = (tanh(LI/2) + j*tanh(LQ/2)) / sqrt(2)
    """
    LI = np.asarray(LI); LQ = np.asarray(LQ)
    return (np.tanh(LI/2.0) + 1j*np.tanh(LQ/2.0))/np.sqrt(2.0)
