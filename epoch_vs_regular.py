import numpy as np
import torch
import os
import pickle

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
    T : callable = lambda x : x,
    T_torch : callable = lambda x : x,
    T_adj : callable = lambda x : x,
    K: int = 5_000,
    g: callable = lambda W : np.sum(mcp(W)),
    g_torch: callable = lambda W : torch.sum(mcp_torch(W)),
    prox: callable = prox_mcp,
    lmo: callable = lambda M : lmo_spectral(M, 1., 6),
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
    loss_NSD, penalty_NSD, ssims_NSD, WHs_NSD = run_melmo2(
        D, g = g, prox = prox, T = T, T_adj = T_adj, max_iter = 2**K, lmo = lambda M : lmo_spectral(M, 1., 6), 
        original = Y,
    )
    print(f'Gamons (regular) loss: {loss_NSD[-1]}')
    results['gamons (regular)'] = {
        'loss': loss_NSD,
        'penalty': penalty_NSD,
        'ssims': ssims_NSD,
        'WH': WHs_NSD,
    }
    
    loss_NSD, penalty_NSD, ssims_NSD, WHs_NSD = run_melmo2_epoch(
        D, g = g, prox = prox, T = T, T_adj = T_adj, max_K = K, lmo = lambda M : lmo_l2(M, 1.), 
        original = Y,
    )
    print(f'Gamons (epochs) loss: {loss_NSD[-1]}')
    results['gamons (epochs)'] = {
        'loss': loss_NSD,
        'penalty': penalty_NSD,
        'ssims': ssims_NSD,
        'WH': WHs_NSD,
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
        
    W = results['gamons (regular)']['WH']
    img = plt.imshow(W)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('Gamons (regular)')
    plt.show()
    
    W = results['gamons (epochs)']['WH']
    img = plt.imshow(W)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('Gamons (epochs)')
    plt.show()
    

    
def plot_loss(results: dict):
    K = results['K']
    scatter_period = 2**K // 20
    norm_D = results['norm']

    x = np.arange(len(results['gamons (regular)']['loss']))[::scatter_period]
    x = np.logspace(0, np.log10(len(results['gamons (regular)']['loss'])-1), num=len(x), endpoint=True).astype(int)
    
    plt.loglog(results['gamons (regular)']['loss']/norm_D)
    plt.scatter(x, results['gamons (regular)']['loss'][x]/norm_D, label = 'GAMONS (regular)', marker="o")
    print(f"GAMONS (regular) : {results['gamons (regular)']['loss'][-1]/norm_D:.3e}")
    
    plt.loglog(results['gamons (epochs)']['loss']/norm_D)
    plt.scatter(x, results['gamons (epochs)']['loss'][x]/norm_D, label = 'GAMONS (epochs)', marker="v")
    print(f"GAMONS (epochs) : {results['gamons (epochs)']['loss'][-1]/norm_D:.3e}")
    
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

    x = np.arange(len(results['gamons (epochs)']['ssims']))[::scatter_period]
    x = np.logspace(0, np.log10(len(results['gamons (epochs)']['ssims'])-1), num=len(x), endpoint=True).astype(int)
    
    plt.loglog(results['gamons (regular)']['ssims'])
    plt.scatter(x, results['gamons (regular)']['ssims'][x], label = 'GAMONS (regular)', marker="o")
    print(f"GAMONS (regular) : {results['gamons (regular)']['ssims'][-1]:.3e}")
    
    plt.loglog(results['gamons (epochs)']['ssims'])
    plt.scatter(x, results['gamons (epochs)']['ssims'][x], label = 'GAMONS (epochs)', marker="v")
    print(f"GAMONS (epochs) : {results['gamons (epochs)']['ssims'][-1]:.3e}")

    
    plt.ylabel(r'$SSIM$')
    plt.xlabel('Iterations')
    plt.title('The SSIM')

    plt.legend()
    plt.savefig(f"denoising_results/{results['dataset_name']}/ssim.png")
    plt.show()
    
def plot_primal_gap_and_penalty(
    results: dict, 
    g:callable = lambda W : np.sum(mcp(W)),
    ):
    
    K = results['K']
    scatter_period = 2**K // 20
    D = results['original']

    x = np.arange(K)[::scatter_period]
    x = np.logspace(0, np.log10(K-1), num=len(x), endpoint=True).astype(int)
    
    g_VS = results['gamons (regular)']['penalty']
    ls_VS = results['gamons (regular)']['loss'] + g_VS
    plt.loglog(ls_VS)
    plt.scatter(x, ls_VS[x], label = 'GAMONS (regular)', marker="o")
    print(f"GAMONS (regular) : {ls_VS[-1]:.3e}")
    
    g_NSD = results['gamons (epochs)']['penalty']
    ls_NSD = results['gamons (epochs)']['loss'] + g_NSD
    plt.loglog(ls_NSD)  
    plt.scatter(x, ls_NSD[x], label = 'GAMONS (epochs)', marker="v")
    print(f"GAMONS (epochs) : {ls_NSD[-1]:.3e}")
    
    
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
    plt.scatter(x, g_NSD[x], label = 'GAMONS (p = 2/3, q = 1/3)', marker="v")
    print(f"GAMONS (p = 2/3, q = 1/3) : {g_NSD[-1]:.3e}")

    plt.legend()
    plt.savefig(f"denoising_results/{results['dataset_name']}/Pk.png")
    plt.show()

