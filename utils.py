import numpy as np
from numba import njit
from scipy.sparse.linalg import svds

import jax.numpy as jnp
import numba as nb
from numba import vectorize, njit, guvectorize

import torch

@njit
def NewtonSchulz(M, steps = 6):
    """
    Newton-Schulz method
    
    Parameters
    ----------
    M : array-like, shape (m, n)
        Input matrix.
    steps : int, optional
        Number of steps to perform. The default and maximum is 6.
    
    Returns
    -------
    M : array-like, shape (m, n)
        Output matrix.
    """
   # by @YouJiacheng (with stability loss idea from @leloykun)
   # https://twitter.com/YouJiacheng/status/1893704552689303901
   # https://gist.github.com/YouJiacheng/393c90cbdc23b09d5688815ba382288b/5bff1f7781cf7d062a155eecd2f13075756482ae

    abc_list = [
        (3955/1024, -8306/1024, 5008/1024),
        (3735/1024, -6681/1024, 3463/1024),
        (3799/1024, -6499/1024, 3211/1024),
        (4019/1024, -6385/1024, 2906/1024),
        (2677/1024, -3029/1024, 1162/1024),
        (2172/1024, -1833/1024,  682/1024)
    ]

    transpose = M.shape[1] > M.shape[0]
    if transpose:
        M = M.T
    M = M / np.linalg.norm(M)
    for a, b, c in abc_list[:steps]:
        A = M.T @ M
        I = np.eye(A.shape[0])
        M = M @ (a * I + b * A + c * A @ A)
    if transpose:
        M = M.T
    return M

#############################
## Vector lmo
#############################
@njit
def lmo_l2(g, r):
    # lmo to l2 ball of radius r
    norm = np.linalg.norm(g)
    if norm == 0:
        return np.zeros_like(g)
    return -r * g / norm

@njit
def lmo_l1(g, r):
    # lmo to l1 ball of radius r
    i = np.argmin(g)  # most negative direction
    x = np.zeros_like(g)
    x[i] = -r
    return x

@njit
def lmo_linf(g, r):
    # lmo to l-infinity ball of radius r
    return -r * np.sign(g)

#############################
## Matrix lmo
#############################

# @njit
def lmo_fro(G, r):
    # Frobenius norm ball
    norm = np.linalg.norm(G, 'fro')
    if norm == 0:
        return np.zeros_like(G)
    return -r * G / norm

# @njit
def lmo_nuclear(G, r):
    # Nuclear norm ball
    if np.allclose(G, 0):
        return np.zeros_like(G)
    u1, s1, v1 = svds(G, k=1)
    # u1 = U[:, 0]
    # v1 = Vt[0, :]
    return -r * np.outer(u1, v1)

# @njit
def lmo_spectral(G, r, steps = 6):
    # Spectral norm ball
    if np.allclose(G, 0):
        return np.zeros_like(G)
    
    return -r*NewtonSchulz(G, steps)

@njit
def lmo_entrywise_l1(G, r):
    # entrywise l1 ball
    flat_idx = np.argmin(G)  # index of most negative entry
    X = np.zeros_like(G)
    X.flat[flat_idx] = -r
    return X

@njit
def lmo_entrywise_linf(G, r):
    # entrywise l-infinity ball
    return -r * np.sign(G)

@njit
def lmo_block_l1(G, r = 1., quantile = .5):
    flat_G = G.flatten()
    idx = np.argsort(np.abs(flat_G))
    X = np.zeros_like(flat_G)
    # X[idx[int(quantile * flat_G.size):]] = -r
    selection = idx[int(quantile * flat_G.size):]
    X[selection] = -flat_G[selection]
    return X.reshape(G.shape)

#############################
## Functions
#############################
# ---- MCP penalty function ----
@vectorize(['float64(float64, float64, float64)'])
def _mcp(x, lam, mu):
    """Scalar MCP penalty function for vectorization."""
    abs_x = abs(x)
    threshold = mu * lam
    if abs_x <= threshold:
        return lam * abs_x - (x * x) / (2.0 * mu)
    else:
        return 0.5 * mu * lam * lam

def mcp(x, lam=1.0, mu=3.0):
    """
    Minimax Concave Penalty (MCP) function, Numba-compatible.
    
    Parameters:
    -----------
    x : array-like or scalar
        Input values.
    lam : float, optional (default=1.0)
        Regularization parameter.
    mu : float, optional (default=3.0)
        Tuning parameter (mu > 1).
    
    Returns:
    --------
    p : ndarray or scalar
        MCP penalty values. Returns a scalar if input is scalar,
        otherwise returns an array of the same shape as input.
    """
    return _mcp(x, lam, mu)

def mcp_old(x, lam = 1., mu = 3.):
    """MCP"""
    x = np.asarray(x)
    p = np.zeros_like(x)
    mask1 = np.abs(x) <= mu * lam
    mask2 = ~mask1
    # Region 1: |x| <= mu*lam
    p[mask1] = lam * np.abs(x[mask1]) - (x[mask1]**2) / (2 * mu)
    # Region 2: |x| > mu*lam
    p[mask2] = 0.5 * mu * lam**2
    return p

def mcp_torch(x, lam=1.0, mu=3.0):
    """MCP penalty (torch). Mirrors numpy mcp() and supports autograd."""
    import torch

    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
    lam_t = torch.as_tensor(lam, dtype=t.dtype, device=t.device)
    mu_t = torch.as_tensor(mu, dtype=t.dtype, device=t.device)

    mask = t.abs() <= mu_t * lam_t
    region1 = lam_t * t.abs() - (t ** 2) / (2 * mu_t)
    region2 = 0.5 * mu_t * (lam_t ** 2)

    return torch.where(mask, region1, region2)

def mcp_jnp(x, lam = 1., mu = 3.):
    """MCP"""
    x = jnp.asarray(x)
    p = jnp.zeros_like(x)
    x_mask1 = jnp.where(jnp.abs(x) <= mu * lam, p, 0)
    # mask2 = ~mask1
    # Region 1: |x| <= mu*lam
    # p[mask1] = lam * jnp.abs(x[mask1]) - (x[mask1]**2) / (2 * mu)
    p = jnp.where(jnp.abs(x) <= mu * lam, lam * jnp.abs(x_mask1) - (x_mask1**2) / (2 * mu), p)
    # Region 2: |x| > mu*lam
    # p[mask2] = 0.5 * mu * lam**2
    p = jnp.where(jnp.abs(x) > mu * lam, 0.5 * mu * lam**2, p)
    
    
    return p
#############################
## Proximals
#############################

def prox_l1(x, b):
    return np.sign(x)*np.maximum(np.abs(x)-b, 0)

# @njit
def spectral_prox_l1_old(W, b):
    O = NewtonSchulz(W)
    res = .5*(
        (W - b*O)@O.T
        @(
            O + 
            NewtonSchulz(W@O.T@O - b*O)
        )
    )
    return res

def spectral_prox_l1(W: np.ndarray, beta: float, num_iters: int = 6):
    """
    Spectral soft thresholding function that applies soft thresholding to the singular values of W.
    
    S_spectral(beta, W) = 1/2[(W - beta·msign(W)) + 
                             (W - beta·msign(W))·msign(msign(W)^T·W - beta·I)]
    
    Args:
        W: Input matrix of shape (m, n)
        beta: Threshold parameter (soft thresholding parameter)
        num_iters: Number of Newton-Schulz iterations for msign computation
    
    Returns:
        Matrix with soft-thresholded singular values
    """
    # Handle the case where beta is zero or negative
    if beta <= 0:
        return W
    
    # Compute matrix sign function of W
    OW = NewtonSchulz(W, num_iters)
    
    # Compute the term (W - beta·msign(W))
    thresholded_term = W - beta * OW
    
    # Compute msign(W)^T·W - beta·I
    # Determine which dimension to use for the identity matrix
    m, n = W.shape
    if m <= n:
        # Use m x m identity
        I = np.eye(m)
        # msign(W)^T·W is m x m
        gram_matrix = OW.T @ W - beta * I
        # Compute the sign of the gram matrix
        sign_gram = NewtonSchulz(gram_matrix, num_iters)
        # Final result using the efficient formulation
        result = 0.5 * (thresholded_term + thresholded_term @ sign_gram)
    else:
        # Use n x n identity
        I = np.eye(n)
        # W·msign(W)^T is n x n
        gram_matrix = W @ OW.T - beta * I
        # Compute the sign of the gram matrix
        sign_gram = NewtonSchulz(gram_matrix, num_iters)
        # Final result using the efficient formulation
        result = 0.5 * (thresholded_term + sign_gram @ thresholded_term)
    
    return result

@njit
def _prox_mcp_scalar(x, b, lam, mu):
    """Numba-compatible scalar implementation of MCP proximal operator."""
    if mu <= b:
        raise ValueError("Need mu > b")
    
    lower = b * lam
    upper = mu * lam
    abs_x = np.abs(x)
    
    if abs_x <= lower:
        return 0.0
    elif abs_x <= upper:
        denom = 1.0 - b / mu
        return np.sign(x) * (abs_x - lower) / denom
    else:
        return x

@guvectorize([(nb.float64, nb.float64, nb.float64, nb.float64, nb.float64[:])],
                '(),(),(),()->()', nopython=True)
def _prox_mcp_gufunc(x, b, lam, mu, out):
    """Generalized ufunc for array inputs."""
    out[0] = _prox_mcp_scalar(x, b, lam, mu)

def prox_mcp(x, b, lam=1.0, mu=3.0):
    """
    Numba-compatible proximal operator for MCP penalty.
    
    Parameters:
    -----------
    x : scalar or ndarray
        Input value(s)
    b : float
        Step size parameter (must be < mu)
    lam : float, optional
        Regularization parameter (default=1.0)
    mu : float, optional
        Tuning parameter (default=3.0, must be > b)
    
    Returns:
    --------
    beta : scalar or ndarray
        Proximal mapping result with same shape as x
    """
    if mu <= b:
        raise ValueError("Need mu > b")
    
    # Handle scalar input directly
    if np.isscalar(x):
        return _prox_mcp_scalar(x, b, lam, mu)
    
    # Handle array inputs using guvectorize
    x_arr = np.asarray(x)
    out = np.empty_like(x_arr)
    _prox_mcp_gufunc(x_arr, b, lam, mu, out)
    return out

# @njit
def prox_mcp_old(x, b, lam = 1., mu = 3.):
    if mu <= b:
        raise ValueError("Need gamma > t")
    x = np.asarray(x)
    beta = np.zeros_like(x)
    lower = b * lam
    upper = mu * lam
    mask2 = (np.abs(x) > lower) & (np.abs(x) <= upper)
    mask3 = np.abs(x) > upper
    soft = np.sign(x[mask2]) * (np.abs(x[mask2]) - lower)
    beta[mask2] = soft / (1 - b / mu)
    beta[mask3] = x[mask3]
    return beta

def grad_gb(x, prox, b):
    return (x - prox(x, b))/b


def noisy_image(X, noise_level = 0.1, noise_type='gaussian', seed=None):
    if seed is not None:
        np.random.seed(seed)
    if noise_type == 'gaussian':
        return X + noise_level * np.random.randn(*X.shape)
    elif noise_type == 'salt_and_pepper':
        noise = np.random.rand(*X.shape)
        noisy_X = X.copy()
        noisy_X[noise < noise_level/2] = 0.0  # Salt (black)
        noisy_X[noise > 1 - noise_level/2] = 1.0  # Pepper (white)
        return noisy_X
    else:
        raise ValueError("noise_type must be 'gaussian' or 'salt_and_pepper'")

def TV(X):
    """
    Compute the discrete gradient ∇X of a 2D matrix X.
    
    Parameters:
        X : np.ndarray of shape (m, n)
    
    Returns:
        Px : np.ndarray of shape (m, n) — horizontal (x) differences
        Py : np.ndarray of shape (m, n) — vertical (y) differences
        
    Forward differences with zero padding on the right/bottom:
        Px[i, j] = X[i, j+1] - X[i, j]   for j < n-1, else 0
        Py[i, j] = X[i+1, j] - X[i, j]   for i < m-1, else 0
    """
    Px = np.zeros_like(X)
    Py = np.zeros_like(X)
    
    Px[:, :-1] = X[:, 1:] - X[:, :-1]   # forward diff in x (columns)
    Py[:-1, :] = X[1:, :] - X[:-1, :]   # forward diff in y (rows)
    
    return Px, Py

def TV_torch(X):
    Px = torch.zeros_like(X)
    Py = torch.zeros_like(X)
    
    Px[:, :-1] = X[:, 1:] - X[:, :-1]   # forward diff in x (columns)
    Py[:-1, :] = X[1:, :] - X[:-1, :]   # forward diff in y (rows)
    
    return Px, Py
    
def div(Px, Py):
    """
    Compute the discrete divergence of vector field (Px, Py).
    This is the NEGATIVE adjoint of grad, i.e., div = -grad^*.
    
    Parameters:
        Px : np.ndarray of shape (m, n)
        Py : np.ndarray of shape (m, n)
    
    Returns:
        div_P : np.ndarray of shape (m, n)
        
    Uses backward differences with zero padding on the left/top:
        div_P[i, j] = (Px[i, j] - Px[i, j-1]) + (Py[i, j] - Py[i-1, j])
    with Px[:, -1], Px[:, 0], etc. handled via zero padding.
    """
    m, n = Px.shape
    div_P = np.zeros_like(Px)
    
    # x-component (horizontal): backward difference
    div_P[:, 0] += Px[:, 0]                     # j=0: no left neighbor → assume 0
    div_P[:, 1:] += Px[:, 1:] - Px[:, :-1]     # j>=1
    
    # y-component (vertical): backward difference
    div_P[0, :] += Py[0, :]                     # i=0: no top neighbor → assume 0
    div_P[1:, :] += Py[1:, :] - Py[:-1, :]     # i>=1
    
    return div_P

def TV_adj(Px, Py):
    """
    Adjoint of the gradient operator: ∇^* (Px, Py) = -div(Px, Py)
    """
    return -div(Px, Py)