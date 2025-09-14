import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # keep this if you want to run the file directly
from typing import Dict, Any, Optional
import numpy as np
from utils.detection import hard_slicer

class Equalizer:
    def reset(self, ch_conf: Dict[str, Any], mod_conf: Dict[str, Any]) -> None: ...
    def warmstart(self, pilots: np.ndarray, rx_pilots: np.ndarray) -> None: ...
    def detect(self, rx_block: np.ndarray, soft_in: Optional[np.ndarray]=None) -> Dict[str, Any]:
        raise NotImplementedError

class TrivialDetector(Equalizer):
    def __init__(self, modulation='qpsk'):
        self.modulation = modulation
    def detect(self, rx_block, soft_in=None):
        pre = rx_block.copy()                 # linear pre-slicer value
        s_hat, bits_hat = hard_slicer(pre, self.modulation)
        return {'pre': pre, 'hard': s_hat, 'bits': bits_hat, 'llr': None}
