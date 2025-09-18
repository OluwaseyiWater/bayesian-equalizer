import numpy as np
import inspect
from dp_gp.rbpf import RBPF

def _normalize_outputs(ret):
    """Accept tuple/list or dict and return (L_ext_eq, soft_seq, aux_dict)."""
    if isinstance(ret, (tuple, list)) and len(ret) >= 2:
        L_ext_eq = np.asarray(ret[0], dtype=float)
        soft_seq = np.asarray(ret[1])
        aux = ret[2] if len(ret) >= 3 and isinstance(ret[2], dict) else {}
        return L_ext_eq, soft_seq, aux
    if isinstance(ret, dict):
        L_ext_eq = ret.get("L_ext_eq", ret.get("L_ext"))
        soft_seq = ret.get("soft_seq", ret.get("soft"))
        if L_ext_eq is None or soft_seq is None:
            raise KeyError("RBPF return dict missing 'L_ext_eq'/'L_ext' or 'soft_seq'/'soft'.")
        return np.asarray(L_ext_eq, dtype=float), np.asarray(soft_seq), dict(ret.get("aux", {}))
    raise TypeError("Unexpected RBPF return type; expected (L_ext_eq, soft_seq, aux) or dict.")

def _call_with_signature(obj, method_name, base_kwargs):
    """Call obj.method_name with kwargs filtered/mapped to its signature."""
    func = getattr(obj, method_name)
    sig_params = set(inspect.signature(func).parameters.keys())

    kwargs = {k: v for k, v in base_kwargs.items() if k in sig_params}

    # Alias mapping when the method uses different names
    def map_if_needed(src_key, candidates):
        if src_key in base_kwargs and not any(k in kwargs for k in candidates):
            for alt in candidates:
                if alt in sig_params:
                    kwargs[alt] = base_kwargs[src_key]
                    break

    # Common aliases seen across RBPF implementations
    map_if_needed("apriori_llr_bits", ("pri_llr_bits", "prior_llr", "L_apriori", "apriori"))
    map_if_needed("pilot_sym",        ("pilot_sym", "pilot", "x_pilot", "pilot_symbols"))
    map_if_needed("pilot_len",        ("pilot_len", "n_pilot", "warmup"))

    return func(**kwargs)

def rbpf_detect(
    y,
    L,
    sigma_v2,
    Np=256,
    model="ar1",
    model_kwargs=None,
    apriori_llr_bits=None,
    pilot_sym=None,
    pilot_len=0,
    ess_thresh=0.5,
    seed=0,
):
    """
    Run one RBPF pass and return (L_ext_eq, soft_seq, aux).
    Tries RBPF.detect(...), RBPF.run(...), then RBPF.infer(...), adapting kwargs to each.
    """
    rng = np.random.default_rng(seed)
    model_kwargs = model_kwargs or {}

    rbpf = RBPF(
        L=int(L),
        Np=int(Np),
        noise_var=float(sigma_v2),
        model=str(model).lower(),
        model_kwargs=model_kwargs,
        rng=rng,
    )

    base_kwargs = dict(
        y=y,
        apriori_llr_bits=apriori_llr_bits,
        pilot_sym=pilot_sym,
        pilot_len=int(pilot_len),
        ess_thresh=float(ess_thresh),
    )

    for name in ("detect", "run", "infer"):
        if hasattr(rbpf, name) and callable(getattr(rbpf, name)):
            ret = _call_with_signature(rbpf, name, base_kwargs)
            L_ext_eq, soft_seq, aux = _normalize_outputs(ret)
            return L_ext_eq, soft_seq, aux

    # error listing callables
    callables = [n for n in dir(rbpf) if callable(getattr(rbpf, n)) and not n.startswith("_")]
    raise AttributeError(
        "RBPF object has no method 'detect', 'run', or 'infer'. "
        f"Available callables: {callables}"
    )
