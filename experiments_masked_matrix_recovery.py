import numpy as np
import torch
import os
import pickle
from typing import Callable, Any

from utils import  prox_mcp, mcp, mcp_torch, lmo_spectral, lmo_l2
from BCD import load_dataset

from runs import run_melmo_M, run_VS_M, run_subgradient_descent_M, run_melmo_M_epoch

import seaborn as sns
sns.set_theme('paper', 'whitegrid')
import matplotlib.pyplot as plt

def run_experiments(
    n_seeds: int = 10,
    ):
    """
    Run experiments on multiple datasets.

    Parameters
    ----------
    n_seeds : int, optional
        The number of seeds to use, by default 10
    """
    results: dict[Any, Any] = {}
    #todo
    
def run_experiment(
    dataset_name: str = 'football',
    K: int = 5_000,
    g: Callable = lambda W : np.sum(mcp(W)),
    g_torch: Callable = lambda W : torch.sum(mcp_torch(W)),
    prox: Callable = prox_mcp,
    lmo: Callable = lambda M : lmo_spectral(M, 1., 6),
    seed: int = 0,
    reg_coeff: float = 0.1,
    p: float = 1/2,
    comment: str = '',
    save: bool = False,
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
    n : int, optional
        The number of seeds, by default 10
    K : int, optional
        The number of iterations to run, by default 5_000
    """
    D = load_dataset(dataset_name)
    
    results = {}
    results['K'] = K
    # results['n_seeds'] = n_seeds
    results['dataset_name'] = dataset_name

    # Min-Max Scaling
    min_vals = D.min()
    max_vals = D.max()
    results['prob'] = p
    
    D = (D - min_vals) / (max_vals - min_vals)
    results['original'] = D
    m, n = D.shape
    rng = np.random.default_rng(seed=seed)
    num_observed = int(p * m * n)
    idx = rng.choice(m * n, size=num_observed, replace=False)
    M = np.zeros(m * n)
    M[idx] = 1.0
    M = M.reshape(m, n)
    norm_D = np.linalg.norm(D, 'fro')**2
    results['M'] = M
    results['norm'] = norm_D
    F_min = g(D)
    results['F_min'] = F_min


    # Running the experiments
    loss, held_out_loss, penalty, ssims, dist_W_prox, X, grad_norms = run_melmo_M(
        D, M, g, prox=prox, max_iter = K, p = 7/12, q = 1/3, lmo = lambda M : lmo_l2(M, 1), reg_coeff = reg_coeff,
    )
    print(f'melmo (l2) loss: {loss[-1]}')
    results['melmo (l2)'] = {
        'loss': loss,
        'held_out_loss': held_out_loss,
        'penalty': penalty,
        'ssims': ssims,
        'dist_W_prox': dist_W_prox,
        'grad_norms': grad_norms,
    }
    loss, held_out_loss, penalty, ssims, dist_W_prox, X, grad_norms = run_melmo_M(
        D, M, g, prox=prox, max_iter = K, p = 7/12, q = 1/3, lmo = lambda M : lmo_spectral(M, 1., 6), reg_coeff = reg_coeff,
    )
    print(f'melmo (spectral) loss: {loss[-1]}')
    results['melmo (spectral)'] = {
        'loss': loss,
        'held_out_loss': held_out_loss,
        'penalty': penalty,
        'ssims': ssims,
        'dist_W_prox': dist_W_prox,
        'grad_norms': grad_norms,
    }
    loss, held_out_loss, penalty, ssims, dist_W_prox, X, grad_norms = run_melmo_M_epoch(
        D, M, g, prox=prox, max_iter = K, p = 2/3, q = 1/3, lmo = lmo, reg_coeff = reg_coeff,
        )
    print(f'melmo epoch-wise loss: {loss[-1]}')
    results['melmo epoch-wise'] = {
        'loss': loss,
        'held_out_loss': held_out_loss,
        'penalty': penalty,
        'ssims': ssims,
        'dist_W_prox': dist_W_prox,
        'grad_norms': grad_norms,
    }
    
    loss, held_out_loss, penalty, ssims, dist_W_prox, X, grad_norms = run_subgradient_descent_M(
        D, M, g_torch,((m, n)), max_iter = K, step_size_rule = 5e-4, 
    ) 
    print(f'Subgradient descent loss: {loss[-1]}')
    results['subgradient'] = {
        'loss': loss,
        'held_out_loss': held_out_loss, 
        'penalty': penalty,
        'ssims': ssims,
        'dist_W_prox': dist_W_prox,
        'grad_norms': grad_norms,
    }



    # plt.scatter(np.arange(len(loss))[::50], loss[::50]/norm_D, label = 'melmo', marker="v")

    # loss = run_MoreauNSD(D, 10, lmo = lmo_fro)
    # plt.loglog(loss/norm_D, label = 'l2 lmo')

    loss, held_out_loss, penalty, ssims, dist_W_prox, X, grad_norms = run_VS_M(
        D, M, g, prox=prox, max_iter = K, reg_coeff = reg_coeff,
    )
    print(f'Variable Smoothing BW loss: {loss[-1]}')
    results['variable smoothing'] = {
        'loss': loss,
        'held_out_loss': held_out_loss, 
        'penalty': penalty,
        'ssims': ssims,
        'dist_W_prox': dist_W_prox,
        'grad_norms': grad_norms,
    }
    
    if save :
        with open(f"masked_matrix_recovery/{dataset_name}/{dataset_name}_prob{p}{"_" if comment else ""}{comment}.pkl", 'wb') as f:
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

    W, H = results['variable smoothing']['WH']
    img = plt.imshow(W@H)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('VS')
    plt.show()
    
    W, H = results['melmo (p = 2/3, q = 1/4)']['WH']
    img = plt.imshow(norm_D*W@H)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('melmo (p = 2/3, q = 1/4)')
    plt.show()

    W, H = results['melmo (p = 2/3, q = 1/3)']['WH']
    img = plt.imshow(norm_D*W@H)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('melmo (p = 2/3, q = 1/3)')
    plt.show()
    
    # W, H = WHs_cvxNSD[-1]
    # ax[3].imshow(norm_D*W@H)
    # ax[3].set_title('Ours (CVX MNSD)')
    W, H = results['melmo (p = 7/12, q = 1/3)']['WH']
    img = plt.imshow(norm_D*W@H)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('melmo (p = 7/12, q = 1/3)')
    plt.show()

    W, H = results['subgradient']['WH']
    img = plt.imshow(norm_D*W@H)
    img.set_cmap('gray')
    plt.axis('off')
    plt.title('Subgradient')
    plt.show()
    
def plot_loss(results: dict, save = False):
    K = results['K']
    scatter_period = K // 20
    # norm_D = results['norm']

    x = np.arange(len(results['melmo (l2)']['loss']))[::scatter_period]
    x = np.logspace(0, np.log10(len(results['melmo (l2)']['loss'])-1), num=len(x), endpoint=True).astype(int)
    
    plt.loglog(results['variable smoothing']['loss'])
    plt.scatter(x, results['variable smoothing']['loss'][x], label = 'Variable Smoothing BW', marker="o")
    print(f"VS : {results['variable smoothing']['loss'][-1]:.3e}")
    
    plt.loglog(results['melmo (l2)']['loss'])
    plt.scatter(x, results['melmo (l2)']['loss'][x], label = 'melmo (l2)', marker="v")
    print(f"melmo (l2) : {results['melmo (l2)']['loss'][-1]:.3e}")
    
    plt.loglog(results['melmo epoch-wise']['loss'])
    plt.scatter(x, results['melmo epoch-wise']['loss'][x], label = 'melmo epoch-wise', marker="^")
    print(f"melmo epoch-wise : {results['melmo epoch-wise']['loss'][-1]:.3e}")
    
    
    plt.loglog(results['melmo (spectral)']['loss'])
    plt.scatter(x, results['melmo (spectral)']['loss'][x], label = 'melmo (spectral)', marker="s")
    print(f"melmo (spectral) : {results['melmo (spectral)']['loss'][-1]:.3e}")
    
    

    plt.loglog(results['subgradient']['loss'])
    plt.scatter(x, results['subgradient']['loss'][x], label = 'Subgradient', marker="*")
    print(f"Subgradient : {results['subgradient']['loss'][-1]:.3e}")

    # if save:
    #     text = f"""
    #     & ${results['subgradient']['loss'][-1]/norm_D:.3e}\\times10^{2}$
    #     & ${results['variable smoothing']['loss'][-1]/norm_D:.3e}\\times10^{2}$
    #     & ${results['melmo (p = 7/12, q = 1/3)']['loss'][-1]/norm_D:.3e}\\times10^{2}$
    #     & ${results['melmo (p = 2/3, q = 1/4)']['loss'][-1]/norm_D:.3e}\\times10^{2}$
    #     """
    #     os.makedirs(f"results/{results['dataset_name']}", exist_ok=True)
    #     with open(f"results/{results['dataset_name']}/rLoss_rank_{results['rank']}.txt", 'w') as f:
    #         f.write(text)
    
    plt.ylabel(r'$\|M\odot(Y - X)\|_F^2$')
    plt.xlabel('Iterations')
    plt.title('Observed residual')

    plt.legend()
    plt.savefig(f"masked_matrix_recovery/{results['dataset_name']}/rLoss_prob_{results['prob']}.png")
    plt.show()
    
def plot_relative_held_out_loss(results: dict, save = False):
    K = results['K']
    scatter_period = K // 20
    norm_D = results['norm']
    M = results['M']
    Y = results['original']
    norm_held_out_Y = np.linalg.norm((1-M)*Y, 'fro')

    x = np.arange(len(results['melmo (l2)']['held_out_loss']))[::scatter_period]
    x = np.logspace(0, np.log10(len(results['melmo (l2)']['held_out_loss'])-1), num=len(x), endpoint=True).astype(int)
    
    plt.loglog(results['variable smoothing']['held_out_loss']/norm_held_out_Y)
    plt.scatter(x, results['variable smoothing']['held_out_loss'][x]/norm_held_out_Y, label = 'Variable Smoothing BW', marker="o")
    print(f"VS : {results['variable smoothing']['held_out_loss'][-1]/norm_held_out_Y:.3e}")
    
    plt.loglog(results['melmo (l2)']['held_out_loss']/norm_held_out_Y)
    plt.scatter(x, results['melmo (l2)']['held_out_loss'][x]/norm_held_out_Y, label = 'melmo (l2)', marker="v")
    print(f"melmo (l2) : {results['melmo (l2)']['held_out_loss'][-1]/norm_held_out_Y:.3e}")
    
    plt.loglog(results['melmo epoch-wise']['held_out_loss']/norm_held_out_Y)
    plt.scatter(x, results['melmo epoch-wise']['held_out_loss'][x]/norm_held_out_Y, label = 'melmo epoch-wise', marker="^")
    print(f"melmo epoch-wise : {results['melmo epoch-wise']['held_out_loss'][-1]/norm_held_out_Y:.3e}")
    
    
    plt.loglog(results['melmo (spectral)']['held_out_loss']/norm_held_out_Y)
    plt.scatter(x, results['melmo (spectral)']['held_out_loss'][x]/norm_held_out_Y, label = 'melmo (spectral)', marker="s")
    print(f"melmo (spectral) : {results['melmo (spectral)']['held_out_loss'][-1]/norm_held_out_Y:.3e}")
    
    

    plt.loglog(results['subgradient']['held_out_loss']/norm_held_out_Y)
    plt.scatter(x, results['subgradient']['held_out_loss'][x]/norm_held_out_Y, label = 'Subgradient', marker="*")
    print(f"Subgradient : {results['subgradient']['held_out_loss'][-1]/norm_held_out_Y:.3e}")

    # if save:
    #     text = f"""
    #     & ${results['subgradient']['held_out_loss'][-1]:.3e}\\times10^{2}$
    #     & ${results['variable smoothing']['held_out_loss'][-1]:.3e}\\times10^{2}$
    #     & ${results['melmo (p = 7/12, q = 1/3)']['held_out_loss'][-1]:.3e}\\times10^{2}$
    #     & ${results['melmo (p = 2/3, q = 1/4)']['held_out_loss'][-1]:.3e}\\times10^{2}$    
    #     """
    #     os.makedirs(f"results/{results['dataset_name']}", exist_ok=True)
    #     with open(f"results/{results['dataset_name']}/heldOutLoss_rank_{results['rank']}.txt", 'w') as f:
    #         f.write(text)
    
    plt.ylabel(r'$\frac{\|P_{\Omega^c}(X_k-Y)\|_F}{\|P_{\Omega^c}(Y)\|_F}$')
    plt.xlabel('Iterations')
    plt.title('Relative held-out prediction error')

    plt.legend()
    plt.savefig(f"masked_matrix_recovery/{results['dataset_name']}/heldOutLoss_prob_{results['prob']}.png")
    plt.show()
    


def plot_smoothing_loss(results: dict):
    K = results['K']
    scatter_period = K // 20
    norm_D = results['norm']

    x = np.arange(len(results['variable smoothing']['dist_W_prox']))[::scatter_period]
    x = np.logspace(0, np.log10(len(results['variable smoothing']['dist_W_prox'])-1), num=len(x), endpoint=True).astype(int)
    
    plt.loglog(results['variable smoothing']['dist_W_prox'])
    plt.scatter(x, results['variable smoothing']['dist_W_prox'][x], label = 'Variable Smoothing BW', marker="o")
    print(f"VS : {results['variable smoothing']['dist_W_prox'][-1]:.3e}")
    
    plt.loglog(results['melmo (l2)']['dist_W_prox'])
    plt.scatter(x, results['melmo (l2)']['dist_W_prox'][x], label = 'melmo (l2)', marker="v")
    print(f"melmo (l2) : {results['melmo (l2)']['dist_W_prox'][-1]:.3e}")
    
    plt.loglog(results['melmo epoch-wise']['dist_W_prox'])
    plt.scatter(x, results['melmo epoch-wise']['dist_W_prox'][x], label = 'melmo epoch-wise', marker="^")
    print(f"melmo epoch-wise : {results['melmo epoch-wise']['dist_W_prox'][-1]:.3e}")
    
    plt.loglog(results['melmo (spectral)']['dist_W_prox'])
    plt.scatter(x, results['melmo (spectral)']['dist_W_prox'][x], label = 'melmo (spectral)', marker="s")
    print(f"melmo (spectral) : {results['melmo (spectral)']['dist_W_prox'][-1]:.3e}")
    
    plt.ylabel(r'$\|X - prox_{\beta_k}(X)\|$')
    plt.xlabel('Iterations')
    plt.title('Proximal gap')

    plt.legend()
    plt.savefig(f"masked_matrix_recovery/{results['dataset_name']}/proximalGap_prob_{results['prob']}.png")
    plt.show()
    
def plot_primal_gap_and_penalty(
    results: dict, 
    save : bool = False,
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
    
    g_NSD = results['melmo (l2)']['penalty']
    ls_NSD = results['melmo (l2)']['loss'] + g_NSD
    plt.loglog(ls_NSD)  
    plt.scatter(x, ls_NSD[x], label = 'melmo (l2)', marker="v")
    print(f"melmo (l2) : {ls_NSD[-1]:.3e}")
    
    g_NSD3 = results['melmo epoch-wise']['penalty']
    ls_NSD3 = results['melmo epoch-wise']['loss'] + g_NSD3
    plt.loglog(ls_NSD3)  
    plt.scatter(x, ls_NSD3[x], label = 'melmo epoch-wise', marker="^")
    print(f"melmo epoch-wise : {ls_NSD3[-1]:.3e}")
    
    g_NSD2 = results['melmo (spectral)']['penalty']
    ls_NSD2 = results['melmo (spectral)']['loss'] + g_NSD2
    plt.loglog(ls_NSD2)  
    plt.scatter(x, ls_NSD2[x], label = 'melmo (spectral)', marker="s")
    print(f"melmo (spectral) : {ls_NSD2[-1]:.3e}")
    
    
    
    g_sub = results['subgradient']['penalty']
    ls_sub = results['subgradient']['loss'] + g_sub
    plt.loglog(ls_sub)
    plt.scatter(x, ls_sub[x], label = 'Subgradient', marker="*")
    print(f"Subgradient : {ls_sub[-1]:.3e}")
    
    if save:
        text = f"""
        & prob = {results['prob']} 
        & ${ls_sub[-1]:.3e}\\times10^{2}$ 
        & ${ls_VS[-1]:.3e}\\times10^{2}$ 
        & ${ls_NSD2[-1]:.3e}\\times10^{2}$
        & ${ls_NSD3[-1]:.3e}\\times10^{2}$
        & ${ls_NSD[-1]:.3e}\\times10^{2}$
        """
        
        os.makedirs(f"masked_matrix_recovery/{results['dataset_name']}", exist_ok=True)
        with open(f"masked_matrix_recovery/{results['dataset_name']}/Fk_prob_{results['prob']}.txt", 'w') as f:
            f.write(text)
    
    
    plt.ylabel(r'$F(X_k)$')
    plt.xlabel(r'Iterations $k$')
    plt.title(r'Objectice function $F(X_k)$')

    plt.legend()
    plt.savefig(f"masked_matrix_recovery/{results['dataset_name']}/Fk_prob_{results['prob']}.png")
    plt.show()
    
    plt.loglog(g_VS)
    plt.scatter(x, g_VS[x], label = 'Variable Smoothing BW', marker="o")
    print(f"VS : {g_VS[-1]:.3e}")
    
    plt.loglog(g_NSD)
    plt.scatter(x, g_NSD[x], label = 'melmo (l2)', marker="v")
    print(f"melmo (l2) : {g_NSD[-1]:.3e}")
    
    plt.loglog(g_NSD3)
    plt.scatter(x, g_NSD3[x], label = 'melmo epoch-wise', marker="^")
    print(f"melmo epoch-wise : {g_NSD3[-1]:.3e}")
    
    plt.loglog(g_NSD2)
    plt.scatter(x, g_NSD2[x], label = 'melmo (spectral)', marker="s")
    print(f"melmo (spectral) : {g_NSD2[-1]:.3e}")
    
    plt.loglog(g_sub)
    plt.scatter(x, g_sub[x], label = 'Subgradient', marker="*")
    plt.ylabel(r'$g(X_k)$')
    plt.xlabel('Iterations')
    plt.title('The penalty')

    plt.legend()
    plt.savefig(f"masked_matrix_recovery/{results['dataset_name']}/Pk_prob_{results['prob']}.png")
    plt.show()

def plot_dual_norms(results: dict):
    K = results['K']
    scatter_period = K // 20

    x = np.arange(len(results['variable smoothing']['dist_W_prox']))[::scatter_period]
    x = np.logspace(0, np.log10(len(results['variable smoothing']['dist_W_prox'])-1), num=len(x), endpoint=True).astype(int)
    
    plt.loglog(results['variable smoothing']['grad_norms'])
    plt.scatter(x, results['variable smoothing']['grad_norms'][x], label = 'Variable Smoothing BW', marker="o")
    print(f"VS : {results['variable smoothing']['grad_norms'][-1]:.3e}")
    
    plt.loglog(results['melmo (l2)']['grad_norms'])
    plt.scatter(x, results['melmo (l2)']['grad_norms'][x], label = 'melmo (l2)', marker="v")
    print(f"melmo (l2) : {results['melmo (l2)']['grad_norms'][-1]:.3e}")
    
    plt.loglog(results['melmo epoch-wise']['grad_norms'])
    plt.scatter(x, results['melmo epoch-wise']['grad_norms'][x], label = 'melmo epoch-wise', marker="^")
    print(f"melmo epoch-wise : {results['melmo epoch-wise']['grad_norms'][-1]:.3e}")
    
    plt.loglog(results['melmo (spectral)']['grad_norms'])
    plt.scatter(x, results['melmo (spectral)']['grad_norms'][x], label = 'melmo (spectral)', marker="s")
    print(f"melmo (spectral) : {results['melmo (spectral)']['grad_norms'][-1]:.3e}")
    
    plt.ylabel(r'$\|\nabla F_k(X_k)\|_*$')
    plt.xlabel('Iterations')
    plt.title('Dual norm of the gradients')

    plt.legend()
    plt.savefig(f"masked_matrix_recovery/{results['dataset_name']}/dualNorms_prob_{results['prob']}.png")
    plt.show()

# def plot_images_ranks(results: dict):
#     for rank in results.keys():
#         fig, ax = plt.subplots(1, 6, figsize=(16, 4))
#         ax[0].axis('off')
#         ax[1].axis('off')
#         ax[2].axis('off')
#         ax[3].axis('off')
#         ax[4].axis('off')
        
#         plt.suptitle(f'rank = {rank}', x=0.05, y=0.5, ha='left', va='center', fontsize=12)
        
#         ax[0].imshow(results[rank]['original'])
#         ax[0].set_title('Original')
        
#         W, H = results[rank]['variable smoothing']['WH']
#         ax[1].imshow(W@H)
#         ax[1].set_title('Variable Smoothing BW')
#         W, H = results[rank]['melmo (p = 2/3, q = 1/4)']['WH']
#         ax[2].imshow(W@H)
#         ax[2].set_title('melmo (p = 2/3. q = 1/4)')
#         W, H = results[rank]['melmo (p = 2/3, q = 1/3)']['WH']
#         ax[3].imshow(W@H)
#         ax[3].set_title('melmo (p = 2/3. q = 1/3)')
#         W, H = results[rank]['melmo (p = 7/12, q = 1/3)']['WH']
#         ax[4].imshow(W@H)
#         ax[4].set_title('melmo (p = 7/12. q = 1/3)')
#         W, H = results[rank]['subgradient']['WH']
#         ax[5].imshow(W@H)
#         ax[5].set_title('Subgradient')  
        
#         plt.show()
