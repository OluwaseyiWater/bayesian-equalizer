
from .modem import qpsk_demod_nearest
def hard_slicer(y, modulation:str):
    if modulation.lower() == 'qpsk':
        return qpsk_demod_nearest(y)
    raise NotImplementedError
