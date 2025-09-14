
import numpy as np
def qpsk_mod(bits):
    bits = np.asarray(bits).astype(int)
    assert bits.ndim == 1 and (len(bits) % 2 == 0)
    b0 = bits[0::2]; b1 = bits[1::2]
    re = 1 - 2*b0; im = 1 - 2*b1
    return (re + 1j*im)/np.sqrt(2.0)

def qpsk_demod_nearest(y):
    re = np.sign(np.real(y)); im = np.sign(np.imag(y))
    re[re==0] = 1; im[im==0] = 1
    s_hat = (re + 1j*im)/np.sqrt(2.0)
    b0 = (re < 0).astype(int); b1 = (im < 0).astype(int)
    bits_hat = np.empty(2*len(b0), dtype=int)
    bits_hat[0::2] = b0; bits_hat[1::2] = b1
    return s_hat, bits_hat

def es_of_constellation(name:str):
    if name.lower() == 'qpsk':
        return 1.0
    raise NotImplementedError
