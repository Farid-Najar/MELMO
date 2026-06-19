# MELMO: Moreau Envelope with Linear Minimization Oracle

A Python implementation of the MELMO algorithm for non-smooth composite optimization problems using a geometry aware descent direction.

## Features

- **MELMO Algorithm**: Moreau Envelope with Linear Minimization Oracle for non-convex problems
- **Multiple Optimization Methods**:
  - Subgradient descent
  - Variable Smoothing from Bohm-Wright (VS)
  - MELMO variants with different step size rules
- **Supported Regularizers**:
  - Minimax Concave Penalty (MCP)
  - L1 and L2 norms
  - Spectral and nuclear norms
  - Any custom regulizers can be used
- **Linear Minimization Oracles (LMOs)**:
  - Frobenius norm ball
  - Spectral norm ball (Newton-Schulz method)
  - Nuclear norm ball
  - L1 and L-infinity balls
  - Entry-wise and block L1 norms
- **Applications**:
  - Low-rank matrix factorization
  - Image denoising
- **Benchmark datasets included**:
  - Camera, Olivetti faces, Spectrometer
  - Synthetic low-rank data
  - Les Miserables and Football graphs

## Installation

This project uses [uv](https://docs.astral.sh/uv) for dependency management.

```bash
# Clone the repository
git clone <repository-url>
cd MELMO

# Install dependencies
uv sync
```

### Requirements

- Python ≥ 3.12
- numpy
- numba
- torch
- matplotlib
- seaborn
- scikit-learn
- scikit-image

## Usage

### Quick Start

```python
import numpy as np
from melmo import melmo
from utils import mcp, prox_mcp, lmo_spectral

# Define your objective function
def f(x):
    return 0.5 * np.linalg.norm(x - target)**2

def grad_f(x):
    return x - target

# Run MELMO
result = melmo(
    x0=x0,
    f=f,
    grad_f=grad_f,
    g=mcp,
    prox=prox_mcp,
    lmo=lambda M: lmo_spectral(M, 1., 6),
    max_iter=1000
)
```

### Running Experiments

You can either run the results.ipynb notebook or run the following code

```python
from experiments import run_experiment, plot_loss, plot_primal_gap_and_penalty

# Run experiment on camera dataset
results = run_experiment(
    dataset_name='camera',
    rank=10,
    K=5000
)

# Plot results
plot_loss(results)
plot_primal_gap_and_penalty(results)
```

## Project Structure

```
.
├── melmo.py                  # Core MELMO algorithm implementation
├── utils.py                  # Utility functions (LMOs, regularizers, prox operators)
├── runs.py                   # Experiment runners for different methods
├── experiments.py            # High-level experiment orchestration and plotting
├── denoising_experiments.py  # Image denoising experiments
├── BCD/                      # Block Coordinate Descent implementation
│   ├── Hadamard_BCD.py
│   ├── main.py
│   └── datasets/
├── results/                  # Experimental results and plots
└── denoising_results/        # Image denoising results
```

## Algorithm Overview

MELMO solves problems of the form:

```
min_x f(x) + g(T(x))
```

where:

- `f` is smooth
- `g` is non-smooth and weakly-convex
- `T` is a linear operator

The algorithm uses:

- Moreau envelope smoothing of the non-smooth term
- Frank-Wolfe style linear minimization steps
- Diminishing step sizes for convergence guarantees

## Citations

If you use this code in your research, please cite the corresponding paper.
