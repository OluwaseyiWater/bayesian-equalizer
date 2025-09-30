import numpy as np
from dp_gp.rbpf import RBPF

def rbpf_detect(
    y,
    L: int,
    sigma_v2: float,
    Np: int,
    model: str,
    model_kwargs: dict,
    apriori_llr_bits=None,
    pilot_sym=None,
    pilot_len=0,
    ess_thresh=0.5,
    seed=0,
    m0=None,
    P0=None
):
    """
    Wrapper function to instantiate and run the RBPF.
    This function now accepts and passes m0 and P0 for warm-starting.
    """
    rng = np.random.default_rng(seed)

    # Instantiate the Rao-Blackwellized Particle Filter
    filt = RBPF(
        L=L,
        Np=Np,
        noise_var=sigma_v2,
        model=model,
        model_kwargs=model_kwargs,
        rng=rng
    )

    LLR, soft_seq, aux = filt.run(
        y=y,
        pri_llr_bits=apriori_llr_bits,
        pilots=pilot_sym,
        pilot_len=pilot_len,
        ess_thresh=ess_thresh,
        m0=m0,      
        P0=P0      
    )

    return LLR, soft_seq, aux