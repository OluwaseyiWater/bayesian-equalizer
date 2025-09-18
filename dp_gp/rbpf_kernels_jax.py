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

    # phi_L for 4 symbols, shared across particles
    def phi_for_s(s):
        return s

    # phi (Np, 4, L) in-place: first element is symbol, tail is mem
    head = jnp.tile(SYMS[None, :, None], (Np, 1, 1))           # (Np,4,1)
    tail = jnp.tile(mem[:, None, :], (1, 4, 1)) if Lm1>0 else jnp.zeros((Np,4,0), dtype=jnp.complex64)
    phiL = jnp.concatenate([head, tail], axis=-1)               # (Np,4,L)

    # Here we assume AR(1). For 2L, interleave zeros on odd slots.
    phi = phiL  # (Np,4,L) -> used as (D=L)
    phiH = jnp.conj(phi)

    # S = phi^H P phi + nv, and mu = phi^H m
    # einsum: (Np,4,L),(Np,L,L),(Np,4,L*) -> (Np,4)
    S = jnp.einsum('nsl,nlm,nsm->ns', phiH, P, phi) + nv
    mu = jnp.einsum('nsl,nl->ns', phiH, m)
    e2 = jnp.abs(y_n - mu) ** 2
    like = jnp.exp(-e2 / S) / (jnp.pi * S)
    return like.real
