import jax
import jax.numpy as jnp

SQRT2 = jnp.sqrt(2.0)
SYMS = jnp.array([1+1j, 1-1j, -1+1j, -1-1j], dtype=jnp.complex64) / SQRT2

@jax.jit
def like_matrix_step(m, P, mem, y_n, nv):
    """
    Vectorized likelihoods for ALL particles and 4 QPSK symbols at time n.
    Args:
      m:  (Np, D) complex
      P:  (Np, D, D) complex (Hermitian PSD)
      mem:(Np, L-1) complex  (ISI memory)
      y_n:() complex
      nv: () real
    Returns:
      like_i_s: (Np, 4) real
      phi_L_s:  (4, L) complex (for reuse if you build phi in Python)
    """
    Np, D = m.shape
    Lm1 = mem.shape[1]
    L = Lm1 + 1

    def phi_for_s(s):
        return s

  
    head = jnp.tile(SYMS[None, :, None], (Np, 1, 1))           
    tail = jnp.tile(mem[:, None, :], (1, 4, 1)) if Lm1>0 else jnp.zeros((Np,4,0), dtype=jnp.complex64)
    phiL = jnp.concatenate([head, tail], axis=-1)               

    phi = phiL  
    phiH = jnp.conj(phi)

    S = jnp.einsum('nsl,nlm,nsm->ns', phiH, P, phi) + nv
    mu = jnp.einsum('nsl,nl->ns', phiH, m)
    e2 = jnp.abs(y_n - mu) ** 2
    like = jnp.exp(-e2 / S) / (jnp.pi * S)
    return like.real
