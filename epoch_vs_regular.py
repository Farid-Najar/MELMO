import numpy as np
import torch
import os
import pickle
from typing import Callable

from utils import  prox_mcp, mcp, mcp_torch, lmo_spectral, lmo_l2, lmo_nuclear, noisy_image
from BCD import load_dataset

from runs import run_melmo2, run_melmo2_epoch

import seaborn as sns
sns.set_theme('paper', 'whitegrid')
import matplotlib.pyplot as plt

def run_experiment(
    dataset_name: str = 'camera',
    noise_type: str = 'salt_and_pepper',
    noise_level: float = 0.1,
    T : Callable = lambda x : x,
    T_torch : Callable = lambda x : x,
    T_adj : Callable = lambda x : x,
    K: int = 5_000,
    g: Callable = lambda W : np.sum(mcp(W)),
    g_torch: Callable = lambda W : torch.sum(mcp_torch(W)),
    prox: Callable = prox_mcp,
    lmo: Callable = lambda M : lmo_spectral(M, 1., 6),
    comment: str = '',
    save: bool = False,
    scale: bool = True,
    # store_WH_every: int = 1,
    ):
    """
    Run an experiment on a given dataset.

    Parameters
    ----------
    dataset_name : str, optional
        The name of the dataset to use, by default 'camera'
        Possible dataset : 
            'synthetic', 'olivetti', 'camera', 'spectrometer', 'football', 'miserables', 'low_rank_synthetic'
    rank : int, optional
        The rank of the low-rank approximation, by default 10
    K : int, optional
        The number of iterations to run, by default 5_000
    """
    D = load_dataset(dataset_name)
    
    results = {}
    results['K'] = K
    results['dataset_name'] = dataset_name

    if scale :
        # Min-Max Scaling
        min_vals = D.min()
        max_vals = D.max()

        Y = (D - min_vals) / (max_vals - min_vals)
    else:
        Y = D.copy()
    D = noisy_image(Y.copy(), noise_type=noise_type, noise_level=noise_level)
    results['original'] = D
    m, n = D.shape
    norm_D = np.linalg.norm(D, 'fro')**2
    results['norm'] = norm_D
    F_min = g(T(D))
    results['F_min'] = F_min


    # Running the experiments
    loss_NSD, penalty_NSD, ssims_NSD, dist_W_prox, WHs_NSD, grad_norms_NSD = run_melmo2(
        D, g = g, prox = prox, T = T, T_adj = T_adj, max_iter = 2**K, lmo = lambda M : lmo_spectral(M, 1., 6), 
        original = Y,
    )
    print(f'melmo (regular) loss: {loss_NSD[-1]}')
    results['melmo (regular)'] = {
        'loss': loss_NSD,
        'penalty': penalty_NSD,
        'ssims': ssims_NSD,
        'dist_W_prox': dist_W_prox,
        'WH': WHs_NSD,
        'grad_norms': grad_norms_NSD,
    }
    
    loss_NSD, penalty_NSD, ssims_NSD, dist_W_prox, WHs_NSD, grad_norms_NSD = run_melmo2_epoch(
        D, g = g, prox = prox, T = T, T_adj = T_adj, max_K = K, lmo = lambda M : lmo_l2(M, 1.), 
        original = Y,
    )
    print(f'melmo (epochs) loss: {loss_NSD[-1]}')
    results['melmo (epochs)'] = {
        'loss': loss_NSD,
        'penalty': penalty_NSD,
        'ssims': ssims_NSD,
        'dist_W_prox': dist_W_prox,
        'WH': WHs_NSD,
        'grad_norms': grad_norms_NSD,
    }
    

    
    if save :
        with open(f'denoising_results/{dataset_name}/{dataset_name}{"_" if comment else ""}{comment}.pkl', 'wb') as f:
            pickle.dump(results, f) 
    
    return results
    

def plot_scaling(dataset_name: str = 'camera'):
    D = load_dataset(dataset_name)
    img = plt.imshow(D)
    plt.title('Original')
    img.set_cmap('gray')
    plt.axis('off')
    plt.show()

    # Min-Max Scaling
    min_vals = D.min()
    max_vals = D.max()

    D = (D - min_vals) / (max_vals - min_vals)

    plt.title('After Min-Max scaling')
    img = plt.imshow(D)
    img.set_cmap('gray')
    plt.axis('off')
    plt.show()

    # # Normalization
    # D /= np.linalg.norm(D, 'fro')
    # plt.title('After normalization')
    # plt.imshow(D)
    # plt.show()

    print(f'Dataset shape: {D.shape}')
    
def plot_images(results: dict):
    D = results['original']
    norm_D = results['norm']
    
    # Plotting the results
    img = plt.imshow(D)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('Original')
    plt.show()
        
    W = results['melmo (regular)']['WH']
    img = plt.imshow(W)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('melmo (regular)')
    plt.show()
    
    W = results['melmo (epochs)']['WH']
    img = plt.imshow(W)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('melmo (epochs)')
    plt.show()
    
def plot_smoothing_loss(results: dict):
    K = results['K']
    scatter_period = 2**K // 20
    norm_D = results['norm']

    x = np.arange(len(results['melmo (regular)']['dist_W_prox']))[::scatter_period]
    x = np.logspace(0, np.log10(len(results['melmo (regular)']['dist_W_prox'])-1), num=len(x), endpoint=True).astype(int)
    
    plt.loglog(results['melmo (regular)']['dist_W_prox'])
    plt.scatter(x, results['melmo (regular)']['dist_W_prox'][x], label = 'melmo (regular)', marker="^")
    print(f"melmo (regular) : {results['melmo (regular)']['dist_W_prox'][-1]:.3e}")
    
    plt.loglog(results['melmo (epochs)']['dist_W_prox'])
    plt.scatter(x, results['melmo (epochs)']['dist_W_prox'][x], label = 'melmo (epochs)', marker="s")
    print(f"melmo (epochs) : {results['melmo (epochs)']['dist_W_prox'][-1]:.3e}")
    
    plt.ylabel(r'$\|TW_k - prox_{\beta_k}(TW_k)\|$')
    plt.xlabel('Iterations')
    plt.title('The proximal gap')

    plt.legend()
    plt.savefig(f"denoising_results/{results['dataset_name']}/proximalGap_rank.png")
    plt.show()
    
def plot_norms(results: dict):
    K = results['K']
    scatter_period = 2**K // 20
    norm_D = results['norm']

    x = np.arange(len(results['melmo (regular)']['dist_W_prox']))[::scatter_period]
    x = np.logspace(0, np.log10(len(results['melmo (regular)']['dist_W_prox'])-1), num=len(x), endpoint=True).astype(int)
    
    plt.loglog(results['melmo (regular)']['grad_norms'])
    plt.scatter(x, results['melmo (regular)']['grad_norms'][x], label = 'melmo (regular)', marker="^")
    print(f"melmo (regular) : {results['melmo (regular)']['grad_norms'][-1]:.3e}")
    
    plt.loglog(results['melmo (epochs)']['grad_norms'])
    plt.scatter(x, results['melmo (epochs)']['grad_norms'][x], label = 'melmo (epochs)', marker="s")
    print(f"melmo (epochs) : {results['melmo (epochs)']['grad_norms'][-1]:.3e}")
    
    plt.ylabel(r'$\|\nabla F_k\|_*$')
    plt.xlabel('Iterations')
    plt.title('The gradient norm')

    plt.legend()
    plt.savefig(f"denoising_results/{results['dataset_name']}/gradNorm_rank.png")
    plt.show()
    

    
def plot_loss(results: dict):
    K = results['K']
    scatter_period = 2**K // 20
    norm_D = results['norm']

    x = np.arange(len(results['melmo (regular)']['loss']))[::scatter_period]
    x = np.logspace(0, np.log10(len(results['melmo (regular)']['loss'])-1), num=len(x), endpoint=True).astype(int)
    
    plt.loglog(results['melmo (regular)']['loss']/norm_D)
    plt.scatter(x, results['melmo (regular)']['loss'][x]/norm_D, label = 'melmo (regular)', marker="o")
    print(f"melmo (regular) : {results['melmo (regular)']['loss'][-1]/norm_D:.3e}")
    
    plt.loglog(results['melmo (epochs)']['loss']/norm_D)
    plt.scatter(x, results['melmo (epochs)']['loss'][x]/norm_D, label = 'melmo (epochs)', marker="v")
    print(f"melmo (epochs) : {results['melmo (epochs)']['loss'][-1]/norm_D:.3e}")
    
    plt.ylabel(r'$\frac{\|Y - WH\|_F^2}{\|Y\|_F^2}$')
    plt.xlabel('Iterations')
    plt.title('The reconstruction loss')

    plt.legend()
    plt.savefig(f"denoising_results/{results['dataset_name']}/rLoss.png")
    plt.show()
    
def plot_ssims(results: dict):
    K = results['K']
    scatter_period = 2**K // 20
    norm_D = results['norm']

    x = np.arange(len(results['melmo (epochs)']['ssims']))[::scatter_period]
    x = np.logspace(0, np.log10(len(results['melmo (epochs)']['ssims'])-1), num=len(x), endpoint=True).astype(int)
    
    plt.loglog(results['melmo (regular)']['ssims'])
    plt.scatter(x, results['melmo (regular)']['ssims'][x], label = 'melmo (regular)', marker="o")
    print(f"melmo (regular) : {results['melmo (regular)']['ssims'][-1]:.3e}")
    
    plt.loglog(results['melmo (epochs)']['ssims'])
    plt.scatter(x, results['melmo (epochs)']['ssims'][x], label = 'melmo (epochs)', marker="v")
    print(f"melmo (epochs) : {results['melmo (epochs)']['ssims'][-1]:.3e}")

    
    plt.ylabel(r'$SSIM$')
    plt.xlabel('Iterations')
    plt.title('The SSIM')

    plt.legend()
    plt.savefig(f"denoising_results/{results['dataset_name']}/ssim.png")
    plt.show()
    
def plot_primal_gap_and_penalty(
    results: dict, 
    g:Callable = lambda W : np.sum(mcp(W)),
    ):
    
    K = results['K']
    scatter_period = 2**K // 20
    D = results['original']

    x = np.arange(K)[::scatter_period]
    x = np.logspace(0, np.log10(K-1), num=len(x), endpoint=True).astype(int)
    
    g_VS = results['melmo (regular)']['penalty']
    ls_VS = results['melmo (regular)']['loss'] + g_VS
    plt.loglog(ls_VS)
    plt.scatter(x, ls_VS[x], label = 'melmo (regular)', marker="o")
    print(f"melmo (regular) : {ls_VS[-1]:.3e}")
    
    g_NSD = results['melmo (epochs)']['penalty']
    ls_NSD = results['melmo (epochs)']['loss'] + g_NSD
    plt.loglog(ls_NSD)  
    plt.scatter(x, ls_NSD[x], label = 'melmo (epochs)', marker="v")
    print(f"melmo (epochs) : {ls_NSD[-1]:.3e}")
    
    
    plt.ylabel(r'$F(x_k)$')
    plt.xlabel(r'Iterations $k$')
    plt.title(r'Objectice function $F(x_k)$')

    plt.legend()
    plt.savefig(f"denoising_results/{results['dataset_name']}/Fk.png")
    plt.show()
    
    plt.loglog(g_VS)
    plt.scatter(x, g_VS[x], label = 'Variable Smoothing BW', marker="o")
    print(f"VS : {g_VS[-1]:.3e}")
    
    plt.loglog(g_NSD)
    plt.scatter(x, g_NSD[x], label = 'melmo (p = 2/3, q = 1/3)', marker="v")
    print(f"melmo (p = 2/3, q = 1/3) : {g_NSD[-1]:.3e}")

    plt.legend()
    plt.savefig(f"denoising_results/{results['dataset_name']}/Pk.png")
    plt.show()

