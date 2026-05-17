import numpy as np
from numba import njit

from utils import lmo_fro, lmo_spectral, grad_gb, mcp, prox_mcp

class GradientDescent:
    def __init__(
        self,
        gamma = 1e-3,
        lmo = None,
        ):
        """Gradient descent scheme and algorithm

        Parameters
        ----------
        gamma : float, optional
            step size (may be time dependent), by default 1e-3
        lmo : function, optional
            the linear minimization oracle function, by default None
        """
        self.gamma = gamma
        self.lmo = lmo
    
    def update(self, x, grad, gamma = None):
        if gamma is None:
            gamma = self.gamma
        g = grad(x)
            
        d = -g if self.lmo is None else self.lmo(g)
        return x + gamma * d
    
    
def _update_gamons(
    xt : np.ndarray,
    gt : np.ndarray,
    gamma : float,
    lmo : callable,
    ):
    dt = lmo(gt)
    return xt + gamma * dt

def melmo(
    x0,
    f : callable,
    grad_f : callable,
    g : callable = mcp,
    prox = prox_mcp,
    lmo = lmo_spectral,
    T : np.ndarray = None,
    gamma = 1/3,
    beta = 1.,
    p = 7/12,
    q = 1/3,
    max_iter = 1_000,
    store_xt_every = 0,
    ):
    # TODO
    
    if T is None:
        T = np.eye(x0.shape[0])
    
    grad_g = lambda x, b : grad_gb(x, prox, b)
    
    grad_norms = np.zeros(max_iter)
        
    xt = x0
    if store_xt_every > 0:
        xts = [xt.copy()]
    else:
        xts = None
    
    fs = np.zeros(max_iter)
    gs = np.zeros(max_iter)
    
    beta_t = beta
    gamma_t = gamma
    
    for t in tqdm(range(max_iter)):
        Txt = T@xt
        
        fs[t] = f(xt)
        gs[t] = g(Txt)
        
        gt = grad_f(xt) + T.T@grad_g(Txt, beta_t)
        grad_norms[t] = np.linalg.norm(gt)
        
        xt = _update_gamons(xt, gt, gamma_t, lmo)
        
        beta_t = beta_t / (t+1)**q
        gamma_t = gamma_t / (t+1)**p
        
        if store_xt_every > 0 and t % store_xt_every == 0:
            xts.append(xt.copy())
    
    result = {
        'x' : xt,
        'fs' : fs,
        'gs' : gs,
        'grad_norms' : grad_norms,
        'xs' : xts,
    }
    return result


def epoch_melmo(
    x0,
    f : callable,
    grad_f : callable,
    g : callable = mcp,
    prox = prox_mcp,
    lmo = lmo_spectral,
    epsilon = 1e-3,
    T : np.ndarray = None,
    gamma = 1/3,
    beta = 1.,
    p = 7/12,
    q = 1/3,
    max_K = 10,
    store_xt_every = 0,
    ):
    # TODO
    
    if T is None:
        T = np.eye(x0.shape[0])
    
    grad_g = lambda x, b : grad_gb(x, prox, b)
    
    grad_norms = np.zeros(max_iter)
        
    xt = x0
    if store_xt_every > 0:
        xts = [xt.copy()]
    else:
        xts = None
    
    fs = np.zeros(max_K)
    gs = np.zeros(max_K)
    
    beta_t = beta
    gamma_t = gamma
    
    for l in tqdm(range(max_K)):
        k0 = 2**l
        k_end = 2**(l+1)
        S = np.inf
        for t in tqdm(range(k0, k_end)):
            Txt = T@xt
            
            fs[t] = f(xt)
            gs[t] = g(Txt)
            
            gt = grad_f(xt) + T.T@grad_g(Txt, beta_t)
            grad_norms[t] = np.linalg.norm(gt)
            
            xt = _update_gamons(xt, gt, gamma_t, lmo)
            
            if grad_norms[t] < S:
                # TODO : complete it
                S = grad_norms[t]
                z = prox(xt, beta_t)
                break
            beta_t = beta_t / t**q
            gamma_t = gamma_t / t**p
            
            if store_xt_every > 0 and t % store_xt_every == 0:
                xts.append(xt.copy())
    
    result = {
        'x' : xt,
        'fs' : fs,
        'gs' : gs,
        'grad_norms' : grad_norms,
        'xs' : xts,
        'j' : t,
    }
    return result