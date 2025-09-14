
class DPGPSMC:
    def __init__(self, n_particles=256, dp_alpha=1.0):
        self.np = n_particles
        self.alpha = dp_alpha
    def detect(self, rx_block, soft_in=None):
        return {'hard': rx_block, 'bits': None, 'llr': None, 'snr_r': None}
