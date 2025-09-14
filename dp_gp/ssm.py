# dp_gp/ssm.py
import numpy as np

# ---------- AR(1) on each tap ----------
def ar1_params(L: int, rho: float, q_var: float):
    F = np.eye(L, dtype=np.complex128) * rho
    Q = np.eye(L, dtype=np.complex128) * q_var
    return F, Q, L  # state_dim = L

def predict(m, P, F, Q):
    m_pred = F @ m
    P_pred = F @ P @ F.conj().T + Q
    return m_pred, P_pred

def update(m_pred, P_pred, phi, y, noise_var):
    """
    Complex KF update for scalar measurement y = phi^T h + v.
    phi: (state_dim,) already 'expanded' to match state.
    """
    phi = phi.reshape(-1, 1)
    S = (phi.T @ P_pred @ phi.conj()).item() + noise_var
    e = y - (phi.T @ m_pred.reshape(-1,1)).item()
    K = (P_pred @ phi.conj()) / S
    m_new = m_pred.reshape(-1,1) + K * e
    P_new = P_pred - K @ (phi.T @ P_pred)
    return m_new.ravel(), P_new, e, S


# ---------- Matérn-3/2 per tap as AR(2) with repeated pole ----------
def _solve_lyapunov_2x2(F, Q):
    """
    Solve P = F P F^H + Q for 2x2 blocks (complex allowed).
    Vectorized via kron solve: vec(P) = (I - F⊗F*)^{-1} vec(Q).
    """
    I4 = np.eye(4, dtype=np.complex128)
    A = I4 - np.kron(F, F.conj())
    q = Q.reshape(-1, 1)
    p = np.linalg.solve(A, q)
    return p.reshape(2, 2)

def matern32_ar2_params(L: int, ell: float, sigma_h2: float, dt: float = 1.0):
    """
    Matérn-3/2 kernel k(τ)=σ^2 (1 + λ|τ|) exp(-λ|τ|), λ=√3/ell.
    Discrete-time exact-mean AR(2) with repeated pole ρ = exp(-λ dt):
      h_n = 2ρ h_{n-1} - ρ^2 h_{n-2} + ε_n ,  ε_n ~ N(0, q)
    We realize it in 2x2 state form per tap:
      x_n = [h_n, h_{n-1}]^T,  F = [[2ρ, -ρ^2],[1, 0]],  Q = [[q,0],[0,0]]
    We choose q so that the steady-state Var(h_n) = sigma_h2 exactly.
    """
    lam = np.sqrt(3.0) / float(ell)
    rho = np.exp(-lam * float(dt))
    a1, a2 = 2.0 * rho, - (rho ** 2)

    # single-tap 2x2 F and Q (with q=1 initially)
    Ftap = np.array([[a1, a2], [1.0, 0.0]], dtype=np.complex128)
    Qtap_1 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)

    # find scaling q so that steady-state P[0,0] = sigma_h2
    P_ss_for_q1 = _solve_lyapunov_2x2(Ftap, Qtap_1)
    scale_q = float(sigma_h2) / (P_ss_for_q1[0, 0].real + 1e-18)
    Qtap = Qtap_1 * scale_q

    # block-diagonalize across L taps
    # state layout per tap: [h_k(n), h_k(n-1)]
    state_dim = 2 * L
    F = np.zeros((state_dim, state_dim), dtype=np.complex128)
    Q = np.zeros_like(F)
    for k in range(L):
        sl = slice(2*k, 2*k+2)
        F[sl, sl] = Ftap
        Q[sl, sl] = Qtap
    return F, Q, state_dim, rho  # returning rho can be useful for init


def build_state_model(kind: str, L: int, **kwargs):
    """
    Factory for (F, Q, state_dim, expand_phi_fn).
    expand_phi_fn(φ) maps an L-length regressor into state_dim-length
    measurement vector for y = φ^T h, given the chosen state layout.
    """
    kind = (kind or "ar1").lower()
    if kind == "ar1":
        F, Q, d = ar1_params(L, rho=float(kwargs.get("rho", 0.98)),
                                q_var=float(kwargs.get("q_var", 1e-4)))
        def expand_phi(phi_L):
            # state is [h0,...,hL-1]; direct mapping
            return np.asarray(phi_L, dtype=np.complex128)
        return F, Q, d, expand_phi

    elif kind in ("matern32", "m32", "matern_32"):
        ell = float(kwargs.get("ell", 10.0))
        sigma_h2 = float(kwargs.get("sigma_h2", 1.0))
        dt = float(kwargs.get("dt", 1.0))
        F, Q, d, _ = matern32_ar2_params(L, ell, sigma_h2, dt)

        def expand_phi(phi_L):
            # state is [h0, h0(z^-1), h1, h1(z^-1), ...]
            # measurement depends only on the first component in each 2x2 block
            out = np.zeros(2*L, dtype=np.complex128)
            out[0::2] = np.asarray(phi_L, dtype=np.complex128)
            return out

        return F, Q, d, expand_phi

    else:
        raise ValueError(f"Unknown state model '{kind}'")
