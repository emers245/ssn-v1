# ssn-v1

A Python implementation of the **Stabilized Supralinear Network (SSN)** model of primary visual cortex (V1), with tools for simulation, stimulus design, and parameter optimization via Bayesian methods.

## Overview

The SSN is a recurrent neural network model that captures key features of cortical dynamics, including orientation selectivity, surround suppression, and inhibition-stabilized network (ISN) regimes. This package provides:

- A flexible `SSN` class for constructing and simulating E-I networks of arbitrary size
- Bayesian optimization and random search for fitting model parameters to target data
- Stimulus generation utilities (gratings, plaids, center-surround)
- CLI tools for large-scale parameter optimization and sweeps
- Tutorials covering basic simulation, model recovery, and stochastic dynamics

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/ssn-v1.git
cd ssn-v1

# Create and activate the conda environment
conda env create -f environment.yml
conda activate ssn-v1

# Install the package in editable mode
pip install -e .
```

## Quick Start

```python
from ssn_v1 import SSN, bayesopt

# Build a network from a JSON config file
net = SSN()
net.load_config("path/to/config.json")
net.add_nodes()
net.add_edges()
net.load_inputs("path/to/inputs.h5")
net.connect_inputs()

# Run simulation
net.run()

# Save outputs
net.save_outputs("path/to/outputs.h5")
```

See the [tutorial notebook](tutorial/tutorial.ipynb) for a complete walkthrough.

## Package Structure

```
ssn-v1/
├── src/ssn_v1/
│   ├── SSN.py                  # Core SSN class and NumericalInstabilityError
│   ├── SSN_utils.py            # Math utilities, cost functions, file I/O
│   ├── designStim.py           # Stimulus generation (gratings, plaids, masks)
│   ├── bayesopt.py             # Bayesian optimization (GP + Expected Improvement)
│   ├── randomopt.py            # Random search baseline
│   ├── run_optimization.py     # CLI: ssn-optimize
│   ├── run_random_search.py    # CLI: ssn-random-search
│   └── run_parameter_sweep.py  # CLI: ssn-sweep
├── tutorial/                   # Basic SSN workflow tutorial
├── bayesopt_tutorial/          # Model recovery on a 512-neuron E-I network
├── sde_tutorial/               # Stochastic (SDE) network tutorial
└── tests/                      # Pytest test suite
```

## Core Components

### `SSN` — Network Simulation

The `SSN` class supports:

- Constructing networks from JSON config files or programmatically via `add_nodes()` / `add_edges()`
- Spatial organization with orientation preference maps (Kaschube method)
- ODE integration via `scipy.integrate.solve_ivp`
- SDE integration via `torchsde` (additive Itô noise)
- Remote file I/O via `paramiko`
- HDF5 output for firing rate traces and network state

```python
from ssn_v1 import SSN, NumericalInstabilityError

net = SSN()
net.load_config("config.json")
net.run()
```

### `bayesopt` — Bayesian Optimization

Fits model parameters to target firing rate distributions using a Gaussian Process surrogate model and Expected Improvement acquisition function.

```python
from ssn_v1 import bayesopt

opt = bayesopt(param_bounds, cost_fn, n_init=50, n_iter=100)
opt.run()
best_params = opt.history_params[opt.history_costs.argmin()]
```

### `SSN_utils` — Utilities

Cost metrics (KL divergence, MSE, spectral analysis), mathematical functions (von Mises, Gaussian, DoG), and file I/O helpers.

### `designStim` — Stimulus Generation

Generates grating, plaid, and center-surround stimuli with configurable spatial frequency, orientation, contrast, and temporal envelopes.

## CLI Tools

After installation, three command-line tools are available:

| Command | Description |
|---|---|
| `ssn-optimize` | Bayesian optimization over network parameters |
| `ssn-random-search` | Random search baseline |
| `ssn-sweep` | Grid/parameter sweep |

```bash
ssn-optimize --config config.json --n-cores 4
```

## Tutorials

### 1. Basic SSN Workflow — [tutorial/tutorial.ipynb](tutorial/tutorial.ipynb)

Covers network construction, spatial organization, stimulus loading, simulation, and output analysis.

### 2. Bayesian Optimization — [bayesopt_tutorial/tutorial_bayesopt.ipynb](bayesopt_tutorial/tutorial_bayesopt.ipynb)

Model recovery on a 512-neuron E-I ring network (256 E + 256 I cells on a 16×16 grid). Recovers 4 synaptic weights (jEE, jEI, jIE, jII) from synthetic plaid-grating responses using KL divergence as the cost function.

**Before running:** generate inputs and targets first:
```bash
cd bayesopt_tutorial
python make_inputs.py
python make_targets.py
```

## Testing

```bash
pytest tests/
```

See [tests/README.md](tests/README.md) for details on individual test files and CI/CD integration.

## Dependencies

- **Core**: NumPy, SciPy, pandas, h5py, joblib, tqdm
- **Optimization**: scikit-learn, PyTorch, torchsde
- **Visualization**: matplotlib
- **Remote I/O**: paramiko, cloudpickle
