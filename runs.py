from utils import lmo_fro, lmo_spectral, lmo_nuclear, prox_l1, grad_gb, prox_mcp, spectral_prox_l1
from BCD import load_dataset, Hadamard_BCD

from skimage.metrics import structural_similarity as ssim

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import Tensor, tensor

from tqdm import tqdm
from typing import Callable

def generateWH(m = 100, n = 100, r = 10):
    W = np.random.rand(m, r)
    H = np.random.rand(r, n)
    return W, H

def update(
    g,
    lmo,
    x : np.ndarray,
    gamma,
    ):
    # gt = grad_F(xt)
    d = lmo(g)
    return x + gamma * d

def run_melmo_c(
    Y,
    r : int,
    prox = prox_l1,
    lmo = lambda M : lmo_spectral(M, 1., 6),
    beta = 1.,
    e = 1e-3,
    max_iter = 1_000,
    ):
    
    grad_g = lambda x, b : grad_gb(x, prox, b)
        
    W, H = np.random.randn(Y.shape[0], r), np.random.randn(r, Y.shape[1])
    beta_t = beta/np.sqrt(2)
    gamma_t = 1
    WHs = [(W, H)]
    
    loss = np.zeros(max_iter)
    dist_W_prox = np.zeros(max_iter)
    
    for t in tqdm(range(max_iter)):
        D = (Y - W@H)
        loss[t] = np.linalg.norm(D, 'fro')**2
        g_W = -D@H.T + grad_g(W, beta_t)
        W = update(g_W, lmo, W, gamma_t)
        g_H = -W.T@D
        H = update(g_H, lmo, H, gamma_t)
        WHs.append((W, H))
        
        dist_W_prox[t] = np.linalg.norm(W - prox(W, beta_t), 'fro')
        
        beta_t = beta / np.sqrt(t+2)
        gamma_t = 2 / (t+2)
        
    # plt.semilogy(loss)
    # plt.show()
    return loss, dist_W_prox, WHs

def run_melmo(
    Y : np.ndarray,
    g  : Callable,
    r : int,
    prox = prox_mcp,
    lmo = lambda M : lmo_spectral(M, 1., 6),
    gamma = 1.,
    beta = 1.5, # 1.5 is the beta for the MCP as it is 1/mu weakly convex
    p = 2/3,
    q = 1/4,
    e = 1e-3,
    max_iter = 1_000,
    fixed_steps = False,
    # store_WH_every = 1,
    ):
    
    grad_g = lambda x, b : grad_gb(x, prox, b)
        
    W, H = np.random.randn(Y.shape[0], r), np.random.randn(r, Y.shape[1])
    beta_t = beta if not fixed_steps else beta / max_iter**q
    gamma_t = gamma if not fixed_steps else gamma / max_iter**p
    # WHs = [(W, H)]
    
    loss = np.zeros(max_iter)
    penalty = np.zeros(max_iter)
    dist_W_prox = np.zeros(max_iter)
    
    for t in tqdm(range(max_iter)):
        D = (Y - W@H)
        loss[t] = np.linalg.norm(D, 'fro')**2
        penalty[t] = g(W)
        g_W = -2*D@H.T + grad_g(W, beta_t)
        g_H = -2*W.T@D
        W = update(g_W, lmo, W, gamma_t)
        H = update(g_H, lmo, H, gamma_t)
        
        # if t % store_WH_every == 0:
        #     WHs.append((W, H))
        
        dist_W_prox[t] = np.linalg.norm(W - prox(W, beta_t), 'fro')
        
        beta_t = beta / (t+1)**q if not fixed_steps else beta_t
        gamma_t = gamma / (t+1)**p if not fixed_steps else gamma_t
        
    # plt.semilogy(loss)
    # plt.show()
    return loss, penalty, dist_W_prox, (W, H)

def run_melmo2(
    Y : np.ndarray,
    g  : Callable,
    T : Callable = lambda x : x,
    T_adj : Callable = lambda x : x,
    prox = spectral_prox_l1,
    lmo = lambda M : lmo_nuclear(M, 1.),
    gamma = 2.,
    beta = 1.5, # 1.5 is the beta for the MCP as it is 1/mu weakly convex
    p = 7/12,
    q = 1.,
    reg_coeff = 1,
    max_iter = 1_000,
    fixed_steps = False,
    original = None,
    norm_type = 2, # spectral norm by default
    dual_norm_type = 'nuc',
    # store_WH_every = 1,
    ):
    
    grad_g = lambda x, b : grad_gb(x, prox, b)
        
    # W = np.random.randn(Y.shape[0], Y.shape[1])
    W = Y.copy()
    beta_t = beta if not fixed_steps else beta / max_iter**q
    gamma_t = gamma if not fixed_steps else gamma / max_iter**p
    # WHs = [(W, H)]
    
    loss = np.zeros(max_iter)
    penalty = np.zeros(max_iter)
    ssims = np.zeros(max_iter)
    dist_W_prox = np.zeros(max_iter)
    grad_norms = np.zeros(max_iter)
    
    if original is None:
        original = Y
    for t in tqdm(range(max_iter)):
        D = (Y - W)
        loss[t] = np.linalg.norm(D, 'fro')**2
        TW = T(W)
        penalty[t] = g(TW)
        g_W = -2*D + reg_coeff * T_adj(grad_g(TW, beta_t))
        W = update(g_W, lmo, W, gamma_t)
        grad_norms[t] = np.linalg.norm(g_W, dual_norm_type)
        
        # if t % store_WH_every == 0:
        #     WHs.append((W, H))
        
        ssims[t] = ssim(original, W, full=True, data_range=1)[0]
        concatenate_W = np.concatenate((TW[0], TW[1]))
        dist_W_prox[t] = np.linalg.norm(concatenate_W - prox(concatenate_W, beta_t), 'fro')
        
        # dist_W_prox[t] = np.linalg.norm(TW[0] - prox(TW[0], beta_t), 'fro')
        # dist_W_prox[t] += np.linalg.norm(TW[1] - prox(TW[1], beta_t), 'fro')
        
        beta_t = beta / (t+1)**q if not fixed_steps else beta_t
        gamma_t = gamma / (t+1)**p if not fixed_steps else gamma_t
        
    # plt.semilogy(loss)
    # plt.show()
    return loss, penalty, ssims, dist_W_prox, W, grad_norms


def run_melmo_M(
    Y : np.ndarray,
    M : np.ndarray,
    g  : Callable,
    T : Callable = lambda x : x,
    T_adj : Callable = lambda x : x,
    prox = spectral_prox_l1,
    lmo = lambda M : lmo_nuclear(M, 1.),
    gamma = 2.,
    beta = 1.5, # 1.5 is the beta for the MCP as it is 1/mu weakly convex
    p = 7/12,
    q = 1/3,
    reg_coeff = 1,
    max_iter = 1_000,
    fixed_steps = False,
    original = None,
    norm_type = 2, # spectral norm by default
    dual_norm_type = 'nuc',
    # store_WH_every = 1,
    ):
    
    grad_g = lambda x, b : grad_gb(x, prox, b)
        
    X  = np.random.randn(Y.shape[0], Y.shape[1])
    # X = Y.copy()
    beta_t = beta if not fixed_steps else beta / max_iter**q
    gamma_t = gamma if not fixed_steps else gamma / max_iter**p
    # WHs = [(W, H)]
    
    loss = np.zeros(max_iter)
    held_out_loss = np.zeros(max_iter)
    penalty = np.zeros(max_iter)
    ssims = np.zeros(max_iter)
    dist_W_prox = np.zeros(max_iter)
    grad_norms = np.zeros(max_iter)
    
    if original is None:
        original = Y
    for t in tqdm(range(max_iter)):
        D = M*(X-Y)
        loss[t] = np.linalg.norm(D, 'fro')**2
        held_out_loss[t] = np.linalg.norm((1-M)*(X-Y), 'fro')
        TX = T(X)
        penalty[t] = g(TX)
        g_W = D + reg_coeff * T_adj(grad_g(TX, beta_t))
        X = update(g_W, lmo, X, gamma_t)
        grad_norms[t] = np.linalg.norm(g_W, dual_norm_type)
        
        # if t % store_WH_every == 0:
        #     WHs.append((W, H))
        
        ssims[t] = ssim(original, X, full=True, data_range=1)[0]
        # concatenate_W = np.concatenate((TW[0], TW[1]))
        dist_W_prox[t] = np.linalg.norm(X - prox(X, beta_t), 'fro')
        
        # dist_W_prox[t] = np.linalg.norm(TW[0] - prox(TW[0], beta_t), 'fro')
        # dist_W_prox[t] += np.linalg.norm(TW[1] - prox(TW[1], beta_t), 'fro')
        
        beta_t = beta / (t+1)**q if not fixed_steps else beta_t
        gamma_t = gamma / (t+1)**p if not fixed_steps else gamma_t
        
    # plt.semilogy(loss)
    # plt.show()
    return loss, held_out_loss, penalty, ssims, dist_W_prox, X, grad_norms

def run_melmo_M_epoch(
    Y : np.ndarray,
    M : np.ndarray,
    g  : Callable,
    T : Callable = lambda x : x,
    T_adj : Callable = lambda x : x,
    prox = spectral_prox_l1,
    lmo = lambda M : lmo_nuclear(M, 1.),
    gamma = 2.,
    beta = 1.5, # 1.5 is the beta for the MCP as it is 1/mu weakly convex
    p = 2/3,
    q = 1/3,
    reg_coeff = 1,
    max_K = 20,
    max_iter = None,
    fixed_steps = False,
    original = None,
    norm_type = 2, # spectral norm by default
    dual_norm_type = 'nuc',
    # store_WH_every = 1,
    ):
    
    grad_g = lambda x, b : grad_gb(x, prox, b)
        
    max_iter = 2**max_K if max_iter is None else max_iter
    # W = np.random.randn(Y.shape[0], Y.shape[1])
    X = np.random.randn(Y.shape[0], Y.shape[1])
    # X = Y.copy()
    beta_t = beta
    gamma_t = gamma
    # WHs = [(W, H)]
    
    loss = np.zeros(max_iter)
    held_out_loss = np.zeros(max_iter)
    penalty = np.zeros(max_iter)
    ssims = np.zeros(max_iter)
    dist_W_prox = np.zeros(max_iter)
    grad_norms = np.zeros(max_iter)
    
    if original is None:
        original = Y
    for l in tqdm(range(max_K)):
        beta_t = beta / 2**(l*q)
        gamma_t = gamma / 2**(l*p)
        for t in range(2**l, 2**(l+1)):
            D = M*(X - Y)
            loss[t] = np.linalg.norm(D, 'fro')**2
            held_out_loss[t] = np.linalg.norm((1-M)*(X-Y), 'fro')
            penalty[t] = g(T(X))
            g_W = D + reg_coeff * T_adj(grad_g(T(X), beta_t))
            X = update(g_W, lmo, X, gamma_t)
            grad_norms[t] = np.linalg.norm(g_W, dual_norm_type)
            
            dist_W_prox[t] = np.linalg.norm(X - prox(X, beta_t), 'fro')
            
            # if t % store_WH_every == 0:
            #     WHs.append((W, H))
            
            ssims[t] = ssim(original, X, full=True, data_range=1)[0]
            if (t+1) % max_iter == 0:
                break
        if (t+1) % max_iter == 0:
                break 
        
    # plt.semilogy(loss)
    # plt.show()
    return loss, held_out_loss, penalty, ssims, dist_W_prox, X, grad_norms


def run_melmo2_epoch(
    Y : np.ndarray,
    g  : Callable,
    T : Callable = lambda x : x,
    T_adj : Callable = lambda x : x,
    prox = spectral_prox_l1,
    lmo = lambda M : lmo_nuclear(M, 1.),
    gamma = 2.,
    beta = 1.5, # 1.5 is the beta for the MCP as it is 1/mu weakly convex
    p = 2/3,
    q = 1/3,
    reg_coeff = 1,
    max_K = 10,
    fixed_steps = False,
    original = None,
    norm_type = 2, # spectral norm by default
    dual_norm_type = 'nuc',
    # store_WH_every = 1,
    ):
    
    grad_g = lambda x, b : grad_gb(x, prox, b)
        
    max_iter = 2**max_K
    # W = np.random.randn(Y.shape[0], Y.shape[1])
    W = Y.copy()
    beta_t = beta
    gamma_t = gamma
    # WHs = [(W, H)]
    
    loss = np.zeros(max_iter)
    penalty = np.zeros(max_iter)
    ssims = np.zeros(max_iter)
    dist_W_prox = np.zeros(max_iter)
    grad_norms = np.zeros(max_iter)
    
    if original is None:
        original = Y
    for l in tqdm(range(max_K)):
        beta_t = beta / 2**(l*q)
        gamma_t = gamma / 2**(l*p)
        for t in range(2**l, 2**(l+1)):
            D = (Y - W)
            loss[t] = np.linalg.norm(D, 'fro')**2
            penalty[t] = g(T(W))
            g_W = -2*D + reg_coeff * T_adj(grad_g(T(W), beta_t))
            W = update(g_W, lmo, W, gamma_t)
            grad_norms[t] = np.linalg.norm(g_W, dual_norm_type)
            
            dist_W_prox[t] = np.linalg.norm(W - prox(W, beta_t), 'fro')
            
            # if t % store_WH_every == 0:
            #     WHs.append((W, H))
            
            ssims[t] = ssim(original, W, full=True, data_range=1)[0]
            
            
        
    # plt.semilogy(loss)
    # plt.show()
    return loss, penalty, ssims, dist_W_prox, W, grad_norms


def run_VS(
    Y,
    g  : Callable,
    r : int,
    prox = prox_mcp,
    L_gradf = 2.,
    beta = 1.5, # 1.5 is the beta for the MCP as it is 1/mu weakly convex
    e = 1e-3,
    max_iter = 1_000,
    # store_WH_every = 1,
    ):
    
    grad_g = lambda x, b : grad_gb(x, prox, b)
        
    W, H = np.random.randn(Y.shape[0], r), np.random.randn(r, Y.shape[1])
    # WHs = [(W, H)]
    
    beta_t = beta
    norm_T2 = 1 #np.max(W.shape)**2
    # gamma_t = 1/(L_gradf + norm_T2/beta_t)
    # gamma_tW = 1/(L_gradf*np.linalg.norm(H)**2 + norm_T2/beta_t)
    # gamma_tH = 1/(L_gradf*np.linalg.norm(W)**2 + norm_T2/beta_t)
    gamma_tW = 1/(L_gradf + norm_T2/beta_t)
    gamma_tH = 1/(L_gradf + norm_T2/beta_t)
    
    loss = np.zeros(max_iter)
    penalty = np.zeros(max_iter)
    dist_W_prox = np.zeros(max_iter)
    
    for t in tqdm(range(max_iter)):
        D = (Y - W@H)
        loss[t] = np.linalg.norm(D, 'fro')**2
        penalty[t] = g(W)
        g_W = -2*D@H.T + grad_g(W, beta_t)
        g_H = -2*W.T@D

        W = update(g_W, lambda x : lmo_fro(x, 1), W, gamma_tW)
        H = update(g_H, lambda x : lmo_fro(x, 1), H, gamma_tH)
        
        # if t % store_WH_every == 0:
        #     WHs.append((W, H))  
        dist_W_prox[t] = np.linalg.norm(W - prox(W, beta_t), 'fro')
        
        beta_t = beta / (t+1)**(1/3)
        # gamma_t = 1/(L_gradf + norm_T2/beta_t)
        gamma_tW = 1/(L_gradf + norm_T2/beta_t)
        gamma_tH = 1/(L_gradf + norm_T2/beta_t)
        
    # plt.semilogy(loss)
    # plt.show()
    return loss, penalty, dist_W_prox, (W, H)

def run_VS2(
    Y,
    g  : Callable,
    T : Callable = lambda x : x,
    T_adj : Callable = lambda x : x,
    prox = spectral_prox_l1,
    L_gradf = 2.,
    beta = 1., # 1.5 is the beta for the MCP as it is 1/mu weakly convex
    e = 1e-3,
    max_iter = 1_000,
    original = None,
    # store_WH_every = 1,
    ):
    
    grad_g = lambda x, b : grad_gb(x, prox, b)
        
    # W = np.random.randn(Y.shape[0], Y.shape[1])
    W = Y.copy()
    
    # WHs = [(W, H)]
    
    beta_t = beta
    norm_T2 = 1 #np.max(W.shape)**2
    gamma_t = 1/(L_gradf + norm_T2/beta_t)
    
    loss = np.zeros(max_iter)
    penalty = np.zeros(max_iter)
    ssims = np.zeros(max_iter)
    
    if original is None:
        original = Y
    for t in tqdm(range(max_iter)):
        D = (Y - W)
        loss[t] = np.linalg.norm(D, 'fro')**2
        penalty[t] = g(T(W))
        g_W = -2*D + T_adj(grad_g(T(W), beta_t))

        W = update(g_W, lambda x : lmo_fro(x, 1), W, gamma_t)
        
        # if t % store_WH_every == 0:
        #     WHs.append((W, H))  
        ssims[t] = ssim(original, W, full=True, data_range=1)[0]
        
        beta_t = beta / (t+1)**(1/3)
        gamma_t = 1/(L_gradf + norm_T2/beta_t)
        
    return loss, penalty, ssims, W


def run_VS_M(
    Y,
    M : np.ndarray,
    g  : Callable,
    T : Callable = lambda x : x,
    T_adj : Callable = lambda x : x,
    prox = spectral_prox_l1,
    L_gradf = 2.,
    beta = 1.5, # 1.5 is the beta for the MCP as it is 1/mu weakly convex
    e = 1e-3,
    max_iter = 1_000,
    reg_coeff = 1,
    original = None,
    # store_WH_every = 1,
    ):
    
    grad_g = lambda x, b : grad_gb(x, prox, b)
        
    X = np.random.randn(Y.shape[0], Y.shape[1])
    # W = Y.copy()
    
    # WHs = [(W, H)]
    
    beta_t = beta
    norm_T2 = 1 #np.max(W.shape)**2
    gamma_t = 1/(L_gradf + norm_T2/beta_t)
    
    loss = np.zeros(max_iter)
    held_out_loss = np.zeros(max_iter)
    penalty = np.zeros(max_iter)
    ssims = np.zeros(max_iter)
    dist_W_prox = np.zeros(max_iter)
    grad_norms = np.zeros(max_iter)
    
    if original is None:
        original = Y
    for t in tqdm(range(max_iter)):
        D = M*(X - Y)
        loss[t] = np.linalg.norm(D, 'fro')**2
        held_out_loss[t] = np.linalg.norm((1-M)*(X-Y), 'fro')
        penalty[t] = g(T(X))
        g_W = D + reg_coeff* T_adj(grad_g(T(X), beta_t))

        X = update(g_W, lambda x : lmo_fro(x, 1), X, gamma_t)
        
        # if t % store_WH_every == 0:
        #     WHs.append((W, H))  
        ssims[t] = ssim(original, X, full=True, data_range=1)[0]
        grad_norms[t] = np.linalg.norm(g_W, 'fro')
        
        beta_t = beta / (t+1)**(1/3)
        gamma_t = 1/(L_gradf + norm_T2/beta_t)
        
    return loss, held_out_loss, penalty, ssims, dist_W_prox, X, grad_norms



def run_subgradient_descent(
    Y,
    g,
    shapes,
    init='random',
    # step_size_rule=lambda k: 1.0 / (k + 1)**(1/4),
    step_size_rule=1e-3,
    max_iter=1000,
    # tol=1e-8,
    callback=None,
    device='cpu',
    dtype=torch.float32,
    seed=None,
    # store_WH_every = 1,
    ):
    """
    Subgradient steepest descent for matrix factorization problems F(W, H).
    
    Parameters:
    -----------
    f : Callable
        Objective function F(W, H) that accepts two PyTorch tensors and returns a scalar tensor
    shapes : tuple of tuples
        (shape_W, shape_H) where shape_W = (n_rows_W, n_cols_W), shape_H = (n_rows_H, n_cols_H)
    init : str or Callable, optional
        Initialization method for W and H:
        - 'random': Normal distribution (mean=0, std=0.1)
        - 'uniform': Uniform distribution [0, 1)
        - 'zeros': Zero matrices
        - Callable: Custom init function taking (shape_W, shape_H) and returning (W0, H0)
    step_size_rule : Callable or float, optional
        Step size rule (constant or function of iteration index k)
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Tolerance for stopping based on subgradient norm
    callback : Callable, optional
        Function called after each iteration with signature:
        callback(iteration, W, H, f_val, grad_norm)
    device : str or torch.device, optional
        Device to perform computations ('cpu' or 'cuda')
    dtype : torch.dtype, optional
        Data type for matrices
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    W : torch.Tensor
        Optimized W matrix
    H : torch.Tensor
        Optimized H matrix
    history : dict
        Optimization history containing:
        - 'f': list of function values
        - 'grad_norm': list of combined gradient norms
        - 'W': list of W matrices (if callback stores them)
        - 'H': list of H matrices (if callback stores them)
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)
    
    # Initialize matrices
    shape_W, shape_H = shapes
    if isinstance(init, Callable):
        W, H = init(shape_W, shape_H)
    elif init == 'random':
        W = torch.randn(*shape_W, device=device, dtype=dtype)
        H = torch.randn(*shape_H, device=device, dtype=dtype)
    elif init == 'uniform':
        W = torch.rand(*shape_W, device=device, dtype=dtype)
        H = torch.rand(*shape_H, device=device, dtype=dtype)
    elif init == 'zeros':
        W = torch.zeros(*shape_W, device=device, dtype=dtype)
        H = torch.zeros(*shape_H, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported initialization method: {init}")
    
    # Enable gradient tracking
    W: Tensor = W.detach().requires_grad_(True)
    H: Tensor = H.detach().requires_grad_(True)
    
    # History tracking
    loss = np.zeros(max_iter)
    penalty = np.zeros(max_iter)
    # WHs = [(W.detach().numpy(), H.detach().numpy())]
    
    Y = torch.from_numpy(Y).to(device, dtype)
    
    # history = {'f': [], 'grad_norm': [], 'WH' : [(W.item(), H.item())]}
    
    for k in tqdm(range(max_iter)):
        # Compute function value and gradients
        # f_val = f(W, H)
        D = (Y - W@H)
        f_val = torch.norm(D, 'fro')**2
        loss[k] = f_val.item()
        pen = g(W)
        f_val += pen
        penalty[k] = pen.item()
        f_val.backward()
        
        # Get gradients and compute norm
        gW: Tensor = W.grad.clone()
        gH: Tensor = H.grad.clone()
        grad_norm = torch.norm(torch.cat([gW.flatten(), gH.flatten()])).item()
        
        
        # Stopping criterion
        # if grad_norm < tol:
        #     print(f"Converged at iteration {k} with gradient norm {grad_norm:.2e}")
        #     loss[k:] = loss[k]
        #     break
        
        # Get step size
        step = step_size_rule(k) if isinstance(step_size_rule, Callable) else step_size_rule
        
        # Update matrices
        with torch.no_grad():
            W_new = W - step * gW
            H_new = H - step * gH
        
        # Prepare for next iteration (break computation graph)
        W = W_new.detach().requires_grad_(True)
        H = H_new.detach().requires_grad_(True)
        
        # Store history
        # if k % store_WH_every == 0:
        #     WHs.append((W.detach().numpy(), H.detach().numpy()))
        
        # Optional callback
        if callback is not None:
            callback(k, W, H, f_val.item(), grad_norm)
    
    return loss, penalty, None, (W.detach().numpy(), H.detach().numpy())


def run_subgradient_descent2(
    Y,
    g,
    shapes,
    T : Callable = lambda x : x,
    init='original',
    # step_size_rule=lambda k: 1.0 / (k + 1)**(1/4),
    step_size_rule=1e-3,
    max_iter=1000,
    # tol=1e-8,
    callback=None,
    device='cpu',
    dtype=torch.float32,
    seed=None,
    original=None,
    # store_WH_every = 1,
    ):
    """
    Subgradient steepest descent for matrix factorization problems F(W, H).
    
    Parameters:
    -----------
    f : Callable
        Objective function F(W, H) that accepts two PyTorch tensors and returns a scalar tensor
    shapes : tuple of tuples
        (shape_W, shape_H) where shape_W = (n_rows_W, n_cols_W), shape_H = (n_rows_H, n_cols_H)
    init : str or Callable, optional
        Initialization method for W and H:
        - 'random': Normal distribution (mean=0, std=0.1)
        - 'uniform': Uniform distribution [0, 1)
        - 'zeros': Zero matrices
        - Callable: Custom init function taking (shape_W, shape_H) and returning (W0, H0)
    step_size_rule : Callable or float, optional
        Step size rule (constant or function of iteration index k)
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Tolerance for stopping based on subgradient norm
    callback : Callable, optional
        Function called after each iteration with signature:
        callback(iteration, W, H, f_val, grad_norm)
    device : str or torch.device, optional
        Device to perform computations ('cpu' or 'cuda')
    dtype : torch.dtype, optional
        Data type for matrices
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    W : torch.Tensor
        Optimized W matrix
    H : torch.Tensor
        Optimized H matrix
    history : dict
        Optimization history containing:
        - 'f': list of function values
        - 'grad_norm': list of combined gradient norms
        - 'W': list of W matrices (if callback stores them)
        - 'H': list of H matrices (if callback stores them)
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)
    
    # Initialize matrices
    if original is None:
        original = Y
        
    shape_W = shapes
    if callable(init):
        # W, H = init(shape_W, shape_H)
        pass
    elif init == 'random':
        W = 0.1 * torch.randn(*shape_W, device=device, dtype=dtype)
        # H = 0.1 * torch.randn(*shape_H, device=device, dtype=dtype)
    elif init == 'uniform':
        W = torch.rand(*shape_W, device=device, dtype=dtype)
        # H = torch.rand(*shape_H, device=device, dtype=dtype)
    elif init == 'zeros':
        W = torch.zeros(*shape_W, device=device, dtype=dtype)
        # H = torch.zeros(*shape_H, device=device, dtype=dtype)
    elif init == 'original':
        W = torch.from_numpy(Y.copy()).to(device, dtype)
    else:
        raise ValueError(f"Unsupported initialization method: {init}")
    
    # Enable gradient tracking
    W = W.detach().requires_grad_(True)
    # H = H.detach().requires_grad_(True)
    
    # History tracking
    loss = np.zeros(max_iter)
    penalty = np.zeros(max_iter)
    ssims = np.zeros(max_iter)
    # WHs = [(W.detach().numpy(), H.detach().numpy())]
    
    Y = torch.from_numpy(Y).to(device, dtype)
    
    # history = {'f': [], 'grad_norm': [], 'WH' : [(W.item(), H.item())]}
    
    for k in tqdm(range(max_iter)):
        # Compute function value and gradients
        # f_val = f(W, H)
        D = (Y - W)
        f_val = torch.norm(D, 'fro')**2
        loss[k] = f_val.item()
        pen = g(T(W))
        f_val += pen
        penalty[k] = pen.item()
        f_val.backward()
        
        # Compute SSIM
        ssims[k] = ssim(original, W.detach().numpy(), full=True, data_range=1)[0]
        
        # Get gradients and compute norm
        gW = W.grad.clone()
        # gH = H.grad.clone()
        grad_norm = torch.norm(gW.flatten()).item()
        
        
        # Get step size
        step = step_size_rule(k) if callable(step_size_rule) else step_size_rule
        
        # Update matrices
        with torch.no_grad():
            W_new = W - step * gW
        
        # Prepare for next iteration (break computation graph)
        W = W_new.detach().requires_grad_(True)
        
        # Store history
        # if k % store_WH_every == 0:
        #     WHs.append((W.detach().numpy(), H.detach().numpy()))
        
        # Optional callback
        if callback is not None:
            callback(k, W, f_val.item(), grad_norm)
    
    return loss, penalty, ssims, W.detach().numpy()

def run_subgradient_descent_M(
    Y,
    M,
    g,
    shapes,
    T : Callable = lambda x : x,
    init='random',
    # step_size_rule=lambda k: 1.0 / (k + 1)**(1/4),
    step_size_rule=1e-3,
    reg_coeff=1,
    max_iter=1000,
    # tol=1e-8,
    callback=None,
    device='cpu',
    dtype=torch.float32,
    seed=None,
    original=None,
    # store_WH_every = 1,
    ):
    """
    Subgradient steepest descent for matrix factorization problems F(W, H).
    
    Parameters:
    -----------
    f : Callable
        Objective function F(W, H) that accepts two PyTorch tensors and returns a scalar tensor
    shapes : tuple of tuples
        (shape_W, shape_H) where shape_W = (n_rows_W, n_cols_W), shape_H = (n_rows_H, n_cols_H)
    init : str or Callable, optional
        Initialization method for W and H:
        - 'random': Normal distribution (mean=0, std=0.1)
        - 'uniform': Uniform distribution [0, 1)
        - 'zeros': Zero matrices
        - Callable: Custom init function taking (shape_W, shape_H) and returning (W0, H0)
    step_size_rule : Callable or float, optional
        Step size rule (constant or function of iteration index k)
    max_iter : int, optional
        Maximum number of iterations
    tol : float, optional
        Tolerance for stopping based on subgradient norm
    callback : Callable, optional
        Function called after each iteration with signature:
        callback(iteration, W, H, f_val, grad_norm)
    device : str or torch.device, optional
        Device to perform computations ('cpu' or 'cuda')
    dtype : torch.dtype, optional
        Data type for matrices
    seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    W : torch.Tensor
        Optimized W matrix
    H : torch.Tensor
        Optimized H matrix
    history : dict
        Optimization history containing:
        - 'f': list of function values
        - 'grad_norm': list of combined gradient norms
        - 'W': list of W matrices (if callback stores them)
        - 'H': list of H matrices (if callback stores them)
    """
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if device == 'cuda':
            torch.cuda.manual_seed(seed)
    
    # Initialize matrices
    if original is None:
        original = Y
        
    shape_W = shapes
    if callable(init):
        # W, H = init(shape_W, shape_H)
        pass
    elif init == 'random':
        X = torch.randn(*shape_W, device=device, dtype=dtype)
        # H = 0.1 * torch.randn(*shape_H, device=device, dtype=dtype)
    elif init == 'uniform':
        X = torch.rand(*shape_W, device=device, dtype=dtype)
        # H = torch.rand(*shape_H, device=device, dtype=dtype)
    elif init == 'zeros':
        X = torch.zeros(*shape_W, device=device, dtype=dtype)
        # H = torch.zeros(*shape_H, device=device, dtype=dtype)
    elif init == 'original':
        X = torch.from_numpy(Y.copy()).to(device, dtype)
    else:
        raise ValueError(f"Unsupported initialization method: {init}")
    
    # Enable gradient tracking
    X = X.detach().requires_grad_(True)
    # H = H.detach().requires_grad_(True)
    M = torch.from_numpy(M).to(device, dtype)
    
    # History tracking
    loss = np.zeros(max_iter)
    held_out_loss = np.zeros(max_iter)
    penalty = np.zeros(max_iter)
    ssims = np.zeros(max_iter)
    grad_norms = np.zeros(max_iter)

    # WHs = [(W.detach().numpy(), H.detach().numpy())]
    
    Y = torch.from_numpy(Y).to(device, dtype)
    
    # history = {'f': [], 'grad_norm': [], 'WH' : [(W.item(), H.item())]}
    
    for k in tqdm(range(max_iter)):
        # Compute function value and gradients
        # f_val = f(W, H)
        D = M*(X - Y)
        f_val = torch.norm(D, 'fro')**2
        loss[k] = f_val.item()
        held_out_loss[k] = torch.norm((1-M)*(X-Y), 'fro').item()
        pen = reg_coeff * g(T(X))
        f_val += pen
        penalty[k] = pen.item()
        f_val.backward()
        
        # Compute SSIM
        ssims[k] = ssim(original, X.detach().numpy(), full=True, data_range=1)[0]
        
        # Get gradients and compute norm
        gW = X.grad.clone()
        # gH = H.grad.clone()
        grad_norm = torch.norm(gW.flatten()).item()
        
        # Compute distance to proximal operator
        grad_norms[k] = grad_norm
        
        # Get step size
        step = step_size_rule(k) if callable(step_size_rule) else step_size_rule
        
        # Update matrices
        with torch.no_grad():
            W_new = X - step * gW
        
        # Prepare for next iteration (break computation graph)
        X = W_new.detach().requires_grad_(True)
        
        # Store history
        # if k % store_WH_every == 0:
        #     WHs.append((W.detach().numpy(), H.detach().numpy()))
        
        # Optional callback
        if callback is not None:
            callback(k, X, f_val.item(), grad_norm)
    
    return loss, held_out_loss, penalty, ssims, None,X.detach().numpy(), grad_norms
