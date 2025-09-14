import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import numpy as np

def make_block_interleaver(n: int, nrows: int | None = None):
    """
    Row-write / column-read interleaver over length-n vector.
    Returns permutation pi of length n such that y = x[pi].
    """
    if nrows is None:
        nrows = max(2, int(round(np.sqrt(n))))  # near-square by default
    ncols = int(np.ceil(n / nrows))

    # fill row-major with indices; pad with -1
    M = -np.ones((nrows, ncols), dtype=int)
    idx = np.arange(n, dtype=int)
    M.flat[:n] = idx

    # read column-major, skip padding
    order = []
    for c in range(ncols):
        for r in range(nrows):
            k = M[r, c]
            if k >= 0:
                order.append(k)
    pi = np.array(order, dtype=int)
    assert pi.size == n
    return pi

def interleave(x, pi):
    x = np.asarray(x)
    return x[pi]

def deinterleave(x, pi):
    x = np.asarray(x)
    inv = np.empty_like(pi)
    inv[pi] = np.arange(len(pi))
    return x[inv]
