import numpy as np
import torch
import os
import pickle

from utils import  prox_mcp, mcp, mcp_torch, lmo_spectral, lmo_l2, lmo_nuclear, noisy_image
from BCD import load_dataset

from runs import run_melmo2, run_VS2, run_subgradient_descent2, run_melmo2_epoch

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
    save: bool = True,
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
    loss_sub, penalty_sub, ssims_sub, WHs_sub = run_subgradient_descent2(
        D, g = g_torch, shapes = (m, n), T = T_torch, max_iter = K, step_size_rule = 5e-4, 
        original = Y,
    ) 
    print(f'Subgradient descent loss: {loss_sub[-1]}')
    results['subgradient'] = {
        'loss': loss_sub,
        'penalty': penalty_sub,
        'ssims': ssims_sub,
        'WH': WHs_sub,
    }
    
    loss_NSD, penalty_NSD, ssims_NSD, _, WHs_NSD, _ = run_melmo2(
        D, g = g, prox = prox, T = T, T_adj = T_adj, max_iter = K, lmo = lambda M : lmo_spectral(M, 1., 6), 
        original = Y,
    )
    print(f'melmo (Spectral LMO) loss: {loss_NSD[-1]}')
    results['melmo (Spectral LMO)'] = {
        'loss': loss_NSD,
        'penalty': penalty_NSD,
        'ssims': ssims_NSD,
        'WH': WHs_NSD,
    }
    
    loss_NSD, penalty_NSD, ssims_NSD, _, WHs_NSD, _ = run_melmo2(
        D, g = g, prox = prox, T = T, T_adj = T_adj, max_iter = K, lmo = lambda M : lmo_l2(M, 1.), 
        original = Y,
    )
    print(f'melmo (L2 LMO) loss: {loss_NSD[-1]}')
    results['melmo (L2 LMO)'] = {
        'loss': loss_NSD,
        'penalty': penalty_NSD,
        'ssims': ssims_NSD,
        'WH': WHs_NSD,
    }
    
    loss_NSD, penalty_NSD, ssims_NSD, _, WHs_NSD, _ = run_melmo2(
        D, g = g, prox = prox, T = T, T_adj = T_adj, max_iter = K, lmo = lambda M : lmo_nuclear(M, 1.), 
        original = Y,
    )
    print(f'melmo (Nuclear LMO) loss: {loss_NSD[-1]}')
    results['melmo (Nuclear LMO)'] = {
        'loss': loss_NSD,
        'penalty': penalty_NSD,
        'ssims': ssims_NSD,
        'WH': WHs_NSD,
    }
    # loss_NSD4, penalty_NSD4, ssims_NSD4, WHs_NSD4 = run_melmo2(
    #     D, g = g, prox = prox, T = T, T_adj = T_adj, max_iter = K, p = 7/12, q = 1/3, lmo = lmo, original = Y,
    # )
    # print(f'melmo (p = 7/12, q = 1/3) loss: {loss_NSD4[-1]}')
    # results['melmo (p = 7/12, q = 1/3)'] = {
    #     'loss': loss_NSD4,
    #     'penalty': penalty_NSD4,
    #     'ssims': ssims_NSD4,
    #     'WH': WHs_NSD4,
    # }

    loss_VS, penalty_VS, ssims_VS, WHs_VS = run_VS2(
        D, g = g, prox = prox, T = T, T_adj = T_adj, max_iter = K, original = Y,
    )
    print(f'Variable Smoothing BW loss: {loss_VS[-1]}')
    results['variable smoothing'] = {
        'loss': loss_VS,
        'penalty': penalty_VS,
        'ssims': ssims_VS,
        'WH': WHs_VS,
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

    W = results['variable smoothing']['WH']
    img = plt.imshow(W)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('VS')
    plt.show()
        
    W = results['melmo (Spectral LMO)']['WH']
    img = plt.imshow(W)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('melmo (Spectral LMO)')
    plt.show()
    
    W = results['melmo (L2 LMO)']['WH']
    img = plt.imshow(W)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('melmo (L2 LMO)')
    plt.show()
    
    W = results['melmo (Nuclear LMO)']['WH']
    img = plt.imshow(W)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('melmo (Nuclear LMO)')
    plt.show()
    
    # W = results['melmo (p = 7/12, q = 1/3)']['WH']
    # img = plt.imshow(W)
    # img.set_cmap('gray')
    # plt.axis('off')
    # plt.title('melmo (p = 7/12, q = 1/3)')
    # plt.show()

    
    # W, H = WHs_cvxNSD[-1]
    # ax[3].imshow(norm_D*W)
    # ax[3].set_title('Ours (CVX MNSD)')
    # W = results['melmo (p = 7/12, q = 1/3)']['WH']
    # img = plt.imshow(W)
    # img.set_cmap('gray')
    # plt.axis('off')
    # plt.title('melmo (p = 7/12, q = 1/3)')
    # plt.show()

    W = results['subgradient']['WH']
    img = plt.imshow(W)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('Subgradient')
    plt.show()
    
def plot_loss(results: dict):
    K = results['K']
    scatter_period = K // 20
    norm_D = results['norm']

    x = np.arange(len(results['variable smoothing']['loss']))[::scatter_period]
    x = np.logspace(0, np.log10(len(results['variable smoothing']['loss'])-1), num=len(x), endpoint=True).astype(int)
    
    plt.loglog(results['variable smoothing']['loss']/norm_D)
    plt.scatter(x, results['variable smoothing']['loss'][x]/norm_D, label = 'Variable Smoothing BW', marker="o")
    print(f"VS : {results['variable smoothing']['loss'][-1]/norm_D:.3e}")
    
    plt.loglog(results['melmo (Spectral LMO)']['loss']/norm_D)
    plt.scatter(x, results['melmo (Spectral LMO)']['loss'][x]/norm_D, label = 'melmo (Spectral LMO)', marker="v")
    print(f"melmo (Spectral LMO) : {results['melmo (Spectral LMO)']['loss'][-1]/norm_D:.3e}")
    
    plt.loglog(results['melmo (L2 LMO)']['loss']/norm_D)
    plt.scatter(x, results['melmo (L2 LMO)']['loss'][x]/norm_D, label = 'melmo (L2 LMO)', marker="^")
    print(f"melmo (L2 LMO) : {results['melmo (L2 LMO)']['loss'][-1]/norm_D:.3e}")
    
    plt.loglog(results['melmo (Nuclear LMO)']['loss']/norm_D)
    plt.scatter(x, results['melmo (Nuclear LMO)']['loss'][x]/norm_D, label = 'melmo (Nuclear LMO)', marker="s")
    print(f"melmo (Nuclear LMO) : {results['melmo (Nuclear LMO)']['loss'][-1]/norm_D:.3e}")
    
    # plt.loglog(results['melmo (p = 2/3, q = 1/3)']['loss']/norm_D)
    # plt.scatter(x, results['melmo (p = 2/3, q = 1/3)']['loss'][x]/norm_D, label = 'melmo (p = 2/3. q = 1/3)', marker="^")
    # print(f"melmo (p = 2/3. q = 1/3) : {results['melmo (p = 2/3, q = 1/3)']['loss'][-1]/norm_D:.3e}")

    plt.loglog(results['subgradient']['loss']/norm_D)
    plt.scatter(x, results['subgradient']['loss'][x]/norm_D, label = 'Subgradient', marker="*")
    print(f"Subgradient : {results['subgradient']['loss'][-1]/norm_D:.3e}")

    # text = f"""
    # & ${results['subgradient']['loss'][-1]/norm_D:.3e}\\times10^{2}$
    # & ${results['variable smoothing']['loss'][-1]/norm_D:.3e}\\times10^{2}$
    # & ${results['melmo (p = 2/3, q = 1/3)']['loss'][-1]/norm_D:.3e}\\times10^{2}$
    # """
    # os.makedirs(f"denoising_results/{results['dataset_name']}", exist_ok=True)
    # with open(f"denoising_results/{results['dataset_name']}/rLoss.txt", 'w') as f:
    #     f.write(text)
    
    plt.ylabel(r'$\frac{\|Y - WH\|_F^2}{\|Y\|_F^2}$')
    plt.xlabel('Iterations')
    plt.title('The reconstruction loss')

    plt.legend()
    plt.savefig(f"denoising_results/{results['dataset_name']}/rLoss.png")
    plt.show()
    
def plot_ssims(results: dict):
    K = results['K']
    scatter_period = K // 20
    norm_D = results['norm']

    x = np.arange(len(results['variable smoothing']['ssims']))[::scatter_period]
    x = np.logspace(0, np.log10(len(results['variable smoothing']['ssims'])-1), num=len(x), endpoint=True).astype(int)
    
    plt.loglog(results['variable smoothing']['ssims'])
    plt.scatter(x, results['variable smoothing']['ssims'][x], label = 'Variable Smoothing BW', marker="o")
    print(f"VS : {results['variable smoothing']['ssims'][-1]:.3e}")
    
    plt.loglog(results['melmo (Spectral LMO)']['ssims'])
    plt.scatter(x, results['melmo (Spectral LMO)']['ssims'][x], label = 'melmo (Spectral LMO)', marker="v")
    print(f"melmo (Spectral LMO) : {results['melmo (Spectral LMO)']['ssims'][-1]:.3e}")
    
    plt.loglog(results['melmo (L2 LMO)']['ssims'])
    plt.scatter(x, results['melmo (L2 LMO)']['ssims'][x], label = 'melmo (L2 LMO)', marker="^")
    print(f"melmo (L2 LMO) : {results['melmo (L2 LMO)']['ssims'][-1]:.3e}")
    
    # plt.loglog(results['melmo (p = 7/12, q = 1/3)']['ssims'])
    # plt.scatter(x, results['melmo (p = 7/12, q = 1/3)']['ssims'][x], label = 'melmo (p = 7/12, q = 1/3)', marker="s")
    # print(f"melmo (p = 7/12, q = 1/3) : {results['melmo (p = 7/12, q = 1/3)']['ssims'][-1]:.3e}")
    plt.loglog(results['melmo (Nuclear LMO)']['ssims'])
    plt.scatter(x, results['melmo (Nuclear LMO)']['ssims'][x], label = 'melmo (Nuclear LMO)', marker="s")
    print(f"melmo (Nuclear LMO) : {results['melmo (Nuclear LMO)']['ssims'][-1]:.3e}")
    
    
    plt.loglog(results['subgradient']['ssims'])
    plt.scatter(x, results['subgradient']['ssims'][x], label = 'Subgradient', marker="*")
    print(f"Subgradient : {results['subgradient']['ssims'][-1]:.3e}")
    
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
    scatter_period = K // 20
    D = results['original']

    x = np.arange(K)[::scatter_period]
    x = np.logspace(0, np.log10(K-1), num=len(x), endpoint=True).astype(int)
    
    g_VS = results['variable smoothing']['penalty']
    ls_VS = results['variable smoothing']['loss'] + g_VS
    plt.loglog(ls_VS)
    plt.scatter(x, ls_VS[x], label = 'Variable Smoothing BW', marker="o")
    print(f"VS : {ls_VS[-1]:.3e}")
    
    g_NSD = results['melmo (Spectral LMO)']['penalty']
    ls_NSD = results['melmo (Spectral LMO)']['loss'] + g_NSD
    plt.loglog(ls_NSD)  
    plt.scatter(x, ls_NSD[x], label = 'melmo (Spectral LMO)', marker="v")
    print(f"melmo (Spectral LMO) : {ls_NSD[-1]:.3e}")
    
    g_NSD = results['melmo (L2 LMO)']['penalty']
    ls_NSD = results['melmo (L2 LMO)']['loss'] + g_NSD
    plt.loglog(ls_NSD)  
    plt.scatter(x, ls_NSD[x], label = 'melmo (L2 LMO)', marker="^")
    print(f"melmo (L2 LMO) : {ls_NSD[-1]:.3e}")
    
    g_NSD = results['melmo (Nuclear LMO)']['penalty']
    ls_NSD = results['melmo (Nuclear LMO)']['loss'] + g_NSD
    plt.loglog(ls_NSD)  
    plt.scatter(x, ls_NSD[x], label = 'melmo (Nuclear LMO)', marker="s")
    print(f"melmo (Nuclear LMO) : {ls_NSD[-1]:.3e}")
    
    g_sub = results['subgradient']['penalty']
    ls_sub = results['subgradient']['loss'] + g_sub
    plt.loglog(ls_sub)
    plt.scatter(x, ls_sub[x], label = 'Subgradient', marker="*")
    print(f"Subgradient : {ls_sub[-1]:.3e}")
    
    text = f"""
    & ${ls_sub[-1]:.3e}\\times10^{2}$ 
    & ${ls_VS[-1]:.3e}\\times10^{2}$ 
    & ${ls_NSD[-1]:.3e}\\times10^{2}$
    """
    
    os.makedirs(f"denoising_results/{results['dataset_name']}", exist_ok=True)
    with open(f"denoising_results/{results['dataset_name']}/Fk.txt", 'w') as f:
        f.write(text)
    
    
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
    
    # plt.loglog(g_NSD2)
    # plt.scatter(x, g_NSD2[x], label = 'melmo (p = 7/12, q = 1/3)', marker="s")
    # print(f"melmo (p = 7/12, q = 1/3) : {g_NSD2[-1]:.3e}")
    
    plt.loglog(g_sub)
    plt.scatter(x, g_sub[x], label = 'Subgradient', marker="*")
    plt.ylabel(r'$g(W_k)$')
    plt.xlabel('Iterations')
    plt.title('The penalty')

    plt.legend()
    plt.savefig(f"denoising_results/{results['dataset_name']}/Pk.png")
    plt.show()

# def plot_images_ranks(results: dict):
#     for rank in results.keys():
#         fig, ax = plt.subplots(1, 4, figsize=(16, 4))
#         ax[0].axis('off')
#         ax[1].axis('off')
#         ax[2].axis('off')
#         ax[3].axis('off')
#         ax[4].axis('off')
        
#         plt.suptitle(f'rank = {rank}', x=0.05, y=0.5, ha='left', va='center', fontsize=12)
        
#         ax[0].imshow(results[rank]['original'])
#         ax[0].set_title('Original')
        
#         W, H = results[rank]['variable smoothing']['WH']
#         ax[1].imshow(W)
#         ax[1].set_title('Variable Smoothing BW')
#         W, H = results[rank]['melmo (p = 2/3, q = 1/3)']['WH']
#         ax[2].imshow(W)
#         ax[2].set_title('melmo (p = 2/3, q = 1/3)')
#         W, H = results[rank]['melmo (p = 7/12, q = 1/3)']['WH']
#         ax[3].imshow(W)
#         ax[3].set_title('melmo (p = 7/12, q = 1/3)')
#         W, H = results[rank]['subgradient']['WH']
#         ax[4].imshow(W)
#         ax[4].set_title('Subgradient')
        
#         plt.show()
