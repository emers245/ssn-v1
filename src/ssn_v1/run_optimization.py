#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI for running SSN Bayesian optimization across multiple target subsets.

Invocation example (from the repo root):

    python -m ssn_v1.run_optimization \
        --config config.json \
        --inputs-dir inputs/training_sample \
        --targets-dir target_data/training_sample \
        --scale 1.0 0.01 1.0 0.01 \
        --fractions 1.0 0.8 0.5 \
        --results-dir optimization_runs \
        --n-init 20 \
        --n-inst 5 \
        --n-iter 50 \
        --seed 42 \
        --fix-seed \
        --fix-nodes

Alternatively:
    install ssn-v1 as a package (e.g., pip install -e .) and run:
        ssn-optimize [args...]

Notes:
    - ``ssn_v1`` is the package name. Using ``python -m ssn_v1.run_optimization``
      avoids path hacks and works from anywhere as long as the package is
      installed (e.g., via ``pip install -e .``).

Flags:
    --config
        Path to the SSN JSON config used by ``SSN.load_config``.
    --inputs-dir / --input-files
        Directory containing feed-forward input HDF5s, optionally narrowed to an
        explicit list of filenames. Each file must expose a ``stimulus`` dataset.
    --targets-dir / --target-files
        Directory (and optional file list) of target rate recordings. Each HDF5
        must expose the ``y`` dataset; the final time bin becomes the cost target.
    --scale
        Gain multipliers per input condition. Provide one value to broadcast or a
        value per stimulus file in load order.
    --fractions
        Fractions of the available ground truth cells to retain for successive 
        optimization runs (e.g., 1.0, 0.8, 0.5). Each fraction yields a separate 
        BO result. This subsamples the ground truth data cells. Use --mask-config 
        to separately filter model output cells based on spatial position.
    --results-dir
        Output directory for individual fraction summaries and the combined
        ``summary.json``.
    --n-init, --n-inst, --n-iter
        Bayesian optimizer hyperparameters: initial random samples, number of
        stochastic instantiations per evaluation, and main iteration count.
    --seed
        Master random seed reused by Bayesian optimization and subset sampling.
    --fix-seed
        When set, reuse ``--seed`` for every evaluation instead of drawing new
        subseeds per instantiation.
    --fix-nodes / --node-seed
        Keep spatial node layouts fixed between evaluations by reusing the given
        ``node_seed`` instead of the per-run seed.
    --feas-threshold
        Upper bound on firing rates; values exceeding this mark a simulation as
        infeasible and skip cost computation.
    --bin-method
        Histogram rule (int or NumPy estimator string) passed to
        ``SSN_utils.compute_cost``.
    --subset-strategy
        Strategy for choosing cell subsets when applying ``--fractions``. Only
        random sampling is currently implemented.
    --track-acquisition
        Enable acquisition tracking in ``bayesopt`` for debugging or analysis.
    --param-bounds
        Optional JSON file overriding :data:`DEFAULT_PARAM_BOUNDS`.
    --param-map
        Optional JSON file overriding :data:`DEFAULT_PARAM_MAP`.
    --mask-config
        Optional JSON file specifying spatial mask for filtering model cells
        used in cost computation. Feasibility checks always use all model cells.
        JSON format: {"mask_type": "circular" or "rectangular",
                      "pos": [x_center, y_center],
                      "rad": radius (for circular),
                      "width": w, "height": h (for rectangular)}
        Allows cost evaluation over a limited spatial window (e.g., field of view)
        while model includes latent variables outside the observed region.
    --debug-eval
        Verbose logging from ``evaluate_parameters`` showing seeds, feasibility,
        and cost diagnostics.
    --save-bo
        Persist each fraction's ``bayesopt`` object as a ``*.joblib`` file in the
        results directory for later inspection or reuse.
    --n-cores
        Number of CPU cores to use during parameter evaluation. Enables a
        shared-memory multiprocessing pool to parallelize stochastic
        instantiations within each evaluation.
    --verbose
        Print detailed output from SSN simulations and parameter evaluations.
        By default (without this flag), only a progress bar is shown during
        Bayesian Optimization iterations.
    --debug-eval
        Print debug info during parameter evaluations.
    --no-events       
        Disable in-simulation feasibility checks. Not recommended except for debugging.
    --eval-subtypes
        When set, compute cost separately for each cell subtype and sum. Requires
        that target data files include a 'model_index' dataset with subtype indices
        for each cell, and that the model config assigns 'model_index' node values.
    --use-feas
        When set, apply a feasibility threshold.
    --feas-thresh
        If --use-feas is set, this specifies the minimum feasibility required to
        consider a parameter set viable (e.g., 0.8 means at least 80% of
        instances must be feasible).
    --cost-funcs
        Cost function to use (e.g., "kl", "mse", "kl_rect"). Defaults to KL divergence.
    --set-k
        Set the k parameter for rectified KL divergence (used with --cost-func kl_rect).
    --use-log-cost
        When set, apply a log transform to the cost value before returning it to
        the optimizer. This can help with optimization stability when costs vary
        widely in scale.
    --feas-acq-threshold
        This argument is followed by a floating point number which sets a threshold on
        the feasibility, below which, the feasibility penalizes the acquisition value.
        This defaults to None, which means that no threshold is applied.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import h5py
import numpy as np
import joblib
import os
import multiprocessing as mp
import ctypes
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
import faulthandler
from tqdm import tqdm

try:
    from .SSN import SSN, NumericalInstabilityError
    from . import SSN_utils
    from .bayesopt import bayesopt
except ImportError:
    from ssn_v1.SSN import SSN, NumericalInstabilityError
    from ssn_v1 import SSN_utils
    from ssn_v1.bayesopt import bayesopt

# --- Module-level multiprocessing helpers (moved out of evaluate_parameters to be pickleable) ---
faulthandler.enable()
_sched_getcpu = ctypes.CDLL(None).sched_getcpu
_sched_getcpu.restype = ctypes.c_int

# shared memory views populated in worker init
_shared_inp = {}
_shared_shm_objects = {}

def _create_shared_blocks(input_data):
    smm = SharedMemoryManager()
    smm.start()
    meta = {}
    for key, arr in input_data.items():
        shm = smm.SharedMemory(size=arr.nbytes)
        buf = np.ndarray(arr.shape, arr.dtype, buffer=shm.buf)
        buf[:] = arr[:]
        meta[key] = (shm.name, arr.shape, arr.dtype.str)
    return meta, smm


def _worker_init(meta, verbose=True, pin_core=True):
    # Pin worker to a single core (Linux only)
    if pin_core and hasattr(os, "sched_setaffinity"):
        try:
            wid = mp.current_process()._identity[0] - 1
            cores = list(os.sched_getaffinity(0))
            os.sched_setaffinity(0, {cores[wid % len(cores)]})
        except Exception:
            pass

    for key, (name, shape, dtype_str) in meta.items():
        shm = SharedMemory(name=name)
        arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        _shared_inp[key] = arr
        _shared_shm_objects[key] = shm

    if verbose:
        try:
            wid
        except NameError:
            wid = 0
        try:
            affinity = sorted(os.sched_getaffinity(0))
        except Exception:
            affinity = []
        print(f"[worker_init] wid={wid:02d}  pid={os.getpid()}  affinity={affinity}  currently_on_cpu={_sched_getcpu()}")
    
    try:
        wid
    except NameError:
        wid = 0
    np.random.seed(10000 + wid)


def _run_instance_condition(args):
    try:
        (theta, param_map, config_file, scale_one, feas_events, fix_nodes, node_seed,
         bin_method, cost_func, subseed, cond_key, ground_truth_data, mask_config,
         return_model_indices, instance_id, noise_spec, target_noise_eigvals,
         max_components) = args
        inp = _shared_inp[cond_key]

        ssn_model = build_network(config_file, theta, param_map, seed=subseed, node_seed=node_seed, verbose=False)

        feas = 1
        noise_eig_cost = None
        try:
            ssn_model = run_simulation(ssn_model, inp, scale=scale_one, events=feas_events)
        except NumericalInstabilityError:
            # Mark as infeasible if overflow occurs
            feas = 0
        
        if feas == 1:
            feas = feasibility_check(ssn_model, feas_events=feas_events)

        if feas == 1:
            final_rates = ssn_model.outputs.y[:, -1]

            # Compute per-condition noise eigenspectrum cost inside worker
            if noise_spec is not None and target_noise_eigvals is not None:
                gt_eig = target_noise_eigvals.get(cond_key)
                if gt_eig is not None:
                    W = ssn_model.construct_W()
                    g = resolve_noise_matrix(noise_spec, len(ssn_model.nodes))
                    h_vec = np.asarray(ssn_model.h[:, -1]).reshape(-1)
                    noise_eig_cost = SSN_utils.noise_eigenspectrum_mse(
                        ssn_model,
                        r_star=final_rates,
                        g=g,
                        gt_eigvals=gt_eig,
                        W=W,
                        max_components=max_components,
                        model_input=h_vec,
                    )

            # Apply spatial mask to model outputs if provided
            if mask_config is not None:
                mask_indices = compute_spatial_mask(ssn_model.nodes['x'], ssn_model.nodes['y'], mask_config)
                final_rates = final_rates[mask_indices]
                # Also subset model indices if needed for subtype cost computation
                model_indices = None
                if return_model_indices:
                    model_indices = ssn_model.nodes['model_index'][mask_indices]
            else:
                model_indices = None
                if return_model_indices:
                    model_indices = ssn_model.nodes['model_index']
        else:
            final_rates = None
            model_indices = None

        return dict(feasible=feas, rates=final_rates, cond_key=cond_key, instance_seed=subseed, 
                   model_indices=model_indices, instance_id=instance_id,
                   noise_eig_cost=noise_eig_cost)
    except Exception as e:
        print(f"[EXC] pid={os.getpid()} -> {e}", flush=True)
        raise

# -----------------------------------------------------------------------------------------------


DEFAULT_PARAM_BOUNDS = {
    "jEE": [0.0, 5.0],
    "jEI": [-5.0, 0.0],
    "jIE": [0.0, 5.0],
    "jII": [-5.0, 0.0],
    "kappaEE": [0.0, 1.0],
    "kappaEI": [0.0, 1.0],
    "kappaIE": [0.0, 1.0],
    "kappaII": [0.0, 1.0],
    "sigmaEE": [0.0, 20.0],
    "sigmaEI": [0.0, 20.0],
    "sigmaIE": [0.0, 20.0],
    "sigmaII": [0.0, 20.0],
    "sigmajEE": [0.0, 1.0],
    "sigmajEI": [0.0, 1.0],
    "sigmajIE": [0.0, 1.0],
    "sigmajII": [0.0, 1.0],
    "sigmaOriEE": [0.0, np.pi/2],
    "sigmaOriEI": [0.0, np.pi/2],
    "sigmaOriIE": [0.0, np.pi/2],
    "sigmaOriII": [0.0, np.pi/2],
}

DEFAULT_PARAM_MAP = {
    "jEE": ("edges", "spatial_tuning", "E<-E", "j"),
    "jEI": ("edges", "spatial_tuning", "E<-I", "j"),
    "jIE": ("edges", "spatial_tuning", "I<-E", "j"),
    "jII": ("edges", "spatial_tuning", "I<-I", "j"),
    "kappaEE": ("edges", "spatial_tuning", "E<-E", "kappa"),
    "kappaEI": ("edges", "spatial_tuning", "E<-I", "kappa"),
    "kappaIE": ("edges", "spatial_tuning", "I<-E", "kappa"),
    "kappaII": ("edges", "spatial_tuning", "I<-I", "kappa"),
    "sigmaEE": ("edges", "spatial_tuning", "E<-E", "sigma"),
    "sigmaEI": ("edges", "spatial_tuning", "E<-I", "sigma"),
    "sigmaIE": ("edges", "spatial_tuning", "I<-E", "sigma"),
    "sigmaII": ("edges", "spatial_tuning", "I<-I", "sigma"),
    "sigmajEE": ("edges", "spatial_tuning", "E<-E", "sigma_j"),
    "sigmajEI": ("edges", "spatial_tuning", "E<-I", "sigma_j"),
    "sigmajIE": ("edges", "spatial_tuning", "I<-E", "sigma_j"),
    "sigmajII": ("edges", "spatial_tuning", "I<-I", "sigma_j"),
    "sigmaOriEE": ("edges", "func_tuning", "E<-E", "sigma_ori"),
    "sigmaOriEI": ("edges", "func_tuning", "E<-I", "sigma_ori"),
    "sigmaOriIE": ("edges", "func_tuning", "I<-E", "sigma_ori"),
    "sigmaOriII": ("edges", "func_tuning", "I<-I", "sigma_ori"),
}


def build_network(config_file: str, theta: np.ndarray, param_keys: Mapping[str, Sequence[str]], seed: int = 0,
                    node_seed: int = 0, verbose: bool = False) -> SSN:
    ssn = SSN("ei_model", verbose=verbose)
    ssn.load_config(config_file)
    ssn.set_rand_seed(seed=seed, map_seed=node_seed)

    for i, path in enumerate(param_keys.values()):
        d = ssn.parameters
        for key in path[:-1]:
            d = d[key]
        d[path[-1]] = float(theta[i])

    ssn.add_nodes()
    ssn.add_edges()
    return ssn


def run_simulation(ssn: SSN, input_matrix: np.ndarray, scale: float = 1.0, events=None) -> SSN:
    ssn.inputs = input_matrix
    ssn.connect_inputs()
    ssn.node_types["c"] = scale
    ssn.nodes["c"] = scale
    ssn.run(events=events)
    return ssn


class ThresholdEvent:
    """Event function for solve_ivp that triggers when values exceed a threshold.
    
    This class is picklable for use with multiprocessing.
    
    Parameters
    ----------
    threshold : float
        Maximum allowed value. Integration stops if any element exceeds this.
    """
    terminal = True
    direction = -1
    
    def __init__(self, threshold: float = 1e4):
        self.threshold = threshold
    
    def __call__(self, t, y, *args):
        """Returns negative when any element of y exceeds threshold."""
        return self.threshold - np.max(y)


def create_threshold_event(threshold: float = 1e4):
    """Create a threshold event function for solve_ivp.
    
    Parameters
    ----------
    threshold : float
        Maximum allowed value. Integration stops if any element exceeds this.
    
    Returns
    -------
    ThresholdEvent
        Event function compatible with solve_ivp that returns negative when
        threshold is exceeded.
    """
    return ThresholdEvent(threshold)


def parse_mask_config(path: Optional[str]) -> Optional[Dict]:
    """Load and validate spatial mask configuration from JSON file.
    
    Parameters
    ----------
    path : str or None
        Path to JSON file with mask specification.
    
    Returns
    -------
    dict or None
        Mask configuration dictionary, or None if path is None.
    
    Expected JSON format:
        {
            "mask_type": "circular" or "rectangular",
            "pos": [x_center, y_center],
            "rad": radius  // for circular
            // OR
            "width": w, "height": h  // for rectangular
        }
    """
    if path is None:
        return None
    
    with open(path, 'r') as f:
        config = json.load(f)
    
    # Validate required fields
    if 'mask_type' not in config:
        raise ValueError("Mask config must specify 'mask_type'")
    if 'pos' not in config:
        raise ValueError("Mask config must specify 'pos' as [x, y]")
    
    mask_type = config['mask_type']
    if mask_type not in ['circular', 'rectangular']:
        raise ValueError(f"mask_type must be 'circular' or 'rectangular', got '{mask_type}'")
    
    if mask_type == 'circular':
        if 'rad' not in config:
            raise ValueError("Circular mask requires 'rad' parameter")
        if config['rad'] <= 0:
            raise ValueError("Circular mask radius must be positive")
    elif mask_type == 'rectangular':
        if 'width' not in config or 'height' not in config:
            raise ValueError("Rectangular mask requires 'width' and 'height' parameters")
        if config['width'] <= 0 or config['height'] <= 0:
            raise ValueError("Rectangular mask width and height must be positive")
    
    if len(config['pos']) != 2:
        raise ValueError("'pos' must be a 2-element array [x, y]")
    
    return config


def parse_noise_spec(raw: Optional[str]):
    """Parse the --noise-matrix argument into a scalar or ndarray.

    Parameters
    ----------
    raw : str or None
        Either a positive float literal (e.g. ``"0.1"``) which will be
        returned as a Python float, or a path to a ``.npy`` / ``.npz`` /
        HDF5 file containing the full noise matrix, returned as an ndarray.
        If *None*, returns *None*.

    Returns
    -------
    float, np.ndarray, or None
    """
    if raw is None:
        return None
    # Try interpreting as a scalar first
    try:
        val = float(raw)
    except ValueError:
        val = None
    if val is not None:
        if val <= 0:
            raise ValueError(f"--noise-matrix scalar must be positive, got {val}")
        return val
    # Otherwise treat as a file path
    p = Path(raw)
    if not p.exists():
        raise FileNotFoundError(f"--noise-matrix file not found: {raw}")
    ext = p.suffix.lower()
    if ext == ".npy":
        arr = np.load(p)
    elif ext == ".npz":
        data = np.load(p)
        keys = list(data.keys())
        arr = data[keys[0]]
    elif ext in (".h5", ".hdf5"):
        with h5py.File(p, "r") as f:
            keys = list(f.keys())
            arr = f[keys[0]][:]
    else:
        raise ValueError(f"Unsupported noise-matrix file format: {ext}")
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Noise matrix must be square, got shape {arr.shape}")
    return arr


def resolve_noise_matrix(noise_spec, n_cells: int) -> np.ndarray:
    """Convert a noise spec (scalar or matrix) to a full (N, N) noise matrix.

    Parameters
    ----------
    noise_spec : float or np.ndarray
        Scalar σ (builds σ·I_N) or pre-loaded (N, N) matrix.
    n_cells : int
        Number of network nodes.

    Returns
    -------
    np.ndarray, shape (n_cells, n_cells)
    """
    if isinstance(noise_spec, (int, float)):
        return float(noise_spec) * np.eye(n_cells)
    noise_spec = np.asarray(noise_spec, dtype=float)
    if noise_spec.shape != (n_cells, n_cells):
        raise ValueError(
            f"Noise matrix shape {noise_spec.shape} does not match "
            f"network size ({n_cells}, {n_cells})"
        )
    return noise_spec


def load_noise_eigvals(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    """Load ground-truth noise eigenvalues keyed per stimulus condition.

    Parameters
    ----------
    path : str or None
        Path to a **directory** containing per-condition HDF5 files named
        ``noise_eigs.{model_name}.{condition_name}.h5``.  Each file stores
        a 1-D ``eigenvalues`` dataset.

    Returns
    -------
    dict mapping condition key → 1-D ndarray, or None.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--noise-eigvals path not found: {path}")
    if not p.is_dir():
        raise ValueError(
            f"--noise-eigvals must be a directory containing per-condition "
            f"noise_eigs.*.h5 files, got file: {path}"
        )
    files = sorted(p.glob("noise_eigs.*.h5"))
    if not files:
        raise FileNotFoundError(
            f"No noise_eigs.*.h5 files found in --noise-eigvals directory: {path}"
        )
    result: Dict[str, np.ndarray] = {}
    for fpath in files:
        # Filename: noise_eigs.{model_name}.{condition_name}.h5
        # condition_name is the last dot-separated component of the stem.
        stem = fpath.stem                    # e.g. "noise_eigs.EI512_model.dg_0deg_4Hz"
        condition_name = stem.split(".")[-1] # e.g. "dg_0deg_4Hz"
        with h5py.File(fpath, "r") as f:
            result[condition_name] = np.asarray(f["eigenvalues"][:], dtype=float).ravel()
    return result


def compute_spatial_mask(nodes_x: np.ndarray, nodes_y: np.ndarray, mask_config: Dict) -> np.ndarray:
    """Compute spatial mask indices based on node positions.
    
    Parameters
    ----------
    nodes_x : np.ndarray
        X coordinates of nodes.
    nodes_y : np.ndarray
        Y coordinates of nodes.
    mask_config : dict
        Mask configuration with 'mask_type', 'pos', and type-specific parameters.
    
    Returns
    -------
    np.ndarray
        Indices of nodes that fall within the spatial mask.
    
    Raises
    ------
    ValueError
        If the mask results in zero cells.
    """
    mask_type = mask_config['mask_type']
    pos = mask_config['pos']
    
    if mask_type == 'circular':
        rad = mask_config['rad']
        distances = np.sqrt((nodes_x - pos[0])**2 + (nodes_y - pos[1])**2)
        mask = distances < rad
    elif mask_type == 'rectangular':
        width = mask_config['width']
        height = mask_config['height']
        half_w = width / 2.0
        half_h = height / 2.0
        mask = ((np.abs(nodes_x - pos[0]) < half_w) & 
                (np.abs(nodes_y - pos[1]) < half_h))
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")
    
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        raise ValueError(
            f"Spatial mask of type '{mask_type}' with config {mask_config} "
            f"results in zero cells. Check mask parameters."
        )
    
    return indices


def feasibility_check(ssn_model: SSN, feas_events: list = None) -> int:
    """Check if simulation results are feasible based on event conditions.
    
    Parameters
    ----------
    ssn_model : SSN
        SSN model with completed simulation (outputs populated).
    feas_events : list of callables, optional
        List of event functions (compatible with solve_ivp format) to check.
        Each function should take (t, y, *args) and return a value that is
        negative when the feasibility condition is violated.
        If None, uses default threshold check at 1e4.
    
    Returns
    -------
    int
        1 if feasible, 0 if infeasible.
    """
    # Check if the simulation was terminated early by an event
    if ssn_model.outputs.status == 1:
        return 0
    
    # If no events specified, use default threshold
    if feas_events is None:
        feas_events = [create_threshold_event(threshold=1e4)]
    
    # Check each feasibility condition against final state
    final_state = ssn_model.outputs.y[:, -1]
    for event_func in feas_events:
        # Call event function with final time and state
        # Events return negative when condition is violated
        if event_func(ssn_model.outputs.t[-1], final_state) <= 0:
            return 0
    
    return 1


def evaluate_parameters(theta, param_map, ground_truth_data, input_data, n_inst=5, seed=0,
                        fix_seed=False, fix_nodes=False, node_seed=0, **kwargs):
    """Evaluate SSN parameters across stochastic instantiations.
    
    Note: ground_truth_data arrives pre-filtered by --fractions in main().
    Model outputs are filtered here by mask_config (--mask-config).
    These are independent: fractions subsamples ground truth cells randomly,
    while mask_config selects model cells spatially.
    """
    config_file = kwargs.get("config_file", "config.json")
    scale = kwargs.get("scale", 1.0)
    threshold = kwargs.get("feas_threshold", 1e5)
    bin_method = kwargs.get("bin_method", "fd")
    debug = kwargs.get("debug", False)
    cost_func = kwargs.get("cost_func", SSN_utils.kl_divergence)
    cost_terms = kwargs.get("cost_terms")
    corr_method = kwargs.get("corr_method", "pearson")
    max_components = kwargs.get("max_components", None)
    target_eigvals = kwargs.get("target_eigvals", None)
    target_eigvals_by_subtype = kwargs.get("target_eigvals_by_subtype", None)
    mask_config = kwargs.get("mask_config")
    n_cores = kwargs.get("n_cores", 1)
    verbose = kwargs.get("verbose", False)
    eval_subtypes = kwargs.get("eval_subtypes", False)
    target_model_indices = kwargs.get("target_model_indices", None)
    noise_spec = kwargs.get("noise_spec", None)
    target_noise_eigvals = kwargs.get("target_noise_eigvals", None)

    if cost_terms is None:
        cost_terms = [{"name": "legacy", "weight": 1.0, "callable": cost_func}]
    
    needs_noise_cost = any(t["name"] == "noise_eig_mse" for t in cost_terms)

    # Create feasibility event list from threshold or use provided events
    feas_events = kwargs.get("feas_events", [create_threshold_event(threshold)])

    if isinstance(scale, (int, float)):
        scale = [scale] * len(input_data)
    elif isinstance(scale, (list, tuple)):
        if len(scale) == 1:
            scale = list(scale) * len(input_data)
        elif len(scale) != len(input_data):
            raise ValueError("Scale vector length must match number of input conditions")
    else:
        raise TypeError("scale must be numeric or a list/tuple")

    rng = np.random.default_rng(seed)
    stable_count = 0
    accum_cost = 0.0
    last_all_sims = None

    # If single-core or multiprocessing not requested, fall back to simple loop
    if n_cores is None or int(n_cores) <= 1:
        input_keys = list(input_data.keys())
        for _ in range(n_inst):
            subseed = seed if fix_seed else rng.integers(0, 9_999_999)
            node_seed = node_seed if fix_nodes else rng.integers(0, 9_999_999)
            
            # Pass verbose flag directly to SSN
            ssn_model = build_network(config_file, theta, param_map, seed=subseed,
                                      node_seed=node_seed, verbose=verbose)

            if debug:
                print(f"\n--- DEBUG: Model Parameters ---\nSeed = {subseed}\n{ssn_model.parameters}")

            # Build adjacency matrix once per network instance
            W = ssn_model.construct_W()

            # Resolve noise matrix for this instance (needs N from ssn_model)
            noise_matrix = None
            if needs_noise_cost and noise_spec is not None:
                noise_matrix = resolve_noise_matrix(noise_spec, len(ssn_model.nodes))

            # Determine expected output size (with spatial mask if provided)
            n_cond = len(input_data)
            if mask_config is not None:
                # Compute mask from this model instance to get expected size
                mask_indices = compute_spatial_mask(ssn_model.nodes['x'], ssn_model.nodes['y'], mask_config)
                model_cell_count = len(mask_indices)
            else:
                # Use ground truth size if no mask (assumes model and data have same number of cells)
                mask_indices = None
                model_cell_count = len(ssn_model.nodes)
            
            all_sims = np.zeros((model_cell_count, n_cond))
            per_condition_meta = []
            feas = 1

            for cidx, (cond_key, inp) in enumerate(input_data.items()):
                try:
                    ssn_model = run_simulation(ssn_model, inp, scale=scale[cidx], events=feas_events)
                except NumericalInstabilityError:
                    # Mark as infeasible if overflow occurs
                    feas = 0
                    break
                feas = feasibility_check(ssn_model, feas_events=feas_events)
                if feas == 0:
                    break
                final_rates = ssn_model.outputs.y[:, -1]
                # Collect per-condition metadata for model-based cost terms
                if needs_noise_cost:
                    per_condition_meta.append({
                        "r_star": final_rates.copy(),
                        "h": np.asarray(ssn_model.h[:, -1]).reshape(-1).copy(),
                    })
                # Apply spatial mask to model outputs if provided (not for feasibility, only for cost)
                if mask_indices is not None:
                    final_rates = final_rates[mask_indices]
                all_sims[:, cidx] = final_rates

            if feas == 1:
                stable_count += 1
                # Debug / validation: ensure model outputs and ground-truth have matching columns
                if debug:
                    print(f"DEBUG: all_sims.shape={all_sims.shape}, ground_truth_data.shape={ground_truth_data.shape}, n_cond={n_cond}")
                if all_sims.shape[1] != ground_truth_data.shape[1]:
                    raise ValueError(
                        f"evaluate_parameters: mismatch in number of conditions — model has {all_sims.shape[1]} columns, "
                        f"but ground_truth_data has {ground_truth_data.shape[1]} columns. Check inputs-dir vs targets-dir and their ordering."
                    )

                # Compute cost, handling per-subtype case
                model_indices = None
                if eval_subtypes:
                    if target_model_indices is None:
                        raise ValueError("eval_subtypes=True but target_model_indices not provided")
                    model_indices = ssn_model.nodes['model_index']
                    if mask_indices is not None:
                        # If spatial mask was applied, subset the model indices accordingly
                        model_indices = model_indices[mask_indices]
                
                run_cost = _compute_weighted_cost(
                    all_sims,
                    ground_truth_data,
                    cost_terms=cost_terms,
                    bin_method=bin_method,
                    corr_method=corr_method,
                    target_eigvals=target_eigvals,
                    max_components=max_components,
                    model_indices=model_indices,
                    target_indices=target_model_indices,
                    target_eigvals_by_subtype=target_eigvals_by_subtype,
                    ssn_model=ssn_model if needs_noise_cost else None,
                    W=W if needs_noise_cost else None,
                    per_condition_meta=per_condition_meta if needs_noise_cost else None,
                    noise_matrix=noise_matrix,
                    target_noise_eigvals=target_noise_eigvals,
                    condition_keys=input_keys if needs_noise_cost else None,
                )
                accum_cost += run_cost
                last_all_sims = all_sims.copy()

        feas_value = stable_count / n_inst
        avg_cost = accum_cost / stable_count if stable_count > 0 else 1e9

        if debug and last_all_sims is not None:
            flattened = last_all_sims.ravel()
            hist, bin_edges = np.histogram(flattened, bins=20)
            print(f"\n--- DEBUG: Evaluate Params for theta={theta} ---")
            print("Histogram of final outputs (flattened):")
            print("  bins:", bin_edges)
            print("  hist:", hist)
            print(f"Cost Value (sum of losses): {avg_cost:.4f}")

        return feas_value, avg_cost

    # ----------------- Multiprocessing path (shared-memory + pin cores) -----------------
    # Use the module-level helpers (_create_shared_blocks, _worker_init, _run_instance_condition)
    # which were defined at module scope so they are pickleable by multiprocessing.

    # Prepare shared memory metadata and pool
    if verbose:
        print("Setting up shared memory manager for multiprocessing...")
    meta, smm = _create_shared_blocks(input_data)
    input_keys = list(input_data.keys())
    n_tasks = n_inst * len(input_keys)
    n_workers = min(int(n_cores), n_tasks, len(os.sched_getaffinity(0)))
    if verbose:
        print(f"Using {n_workers} Workers (requested {n_cores})")

    try:
        with mp.Pool(processes=n_workers, initializer=_worker_init, initargs=(meta, verbose)) as pool:
            tasks = []
            instance_seeds = ([seed] * n_inst if fix_seed else rng.integers(0, 9_999_999, size=n_inst))
            for inst_id, subseed in enumerate(instance_seeds):
                for cidx, cond_key in enumerate(input_keys):
                    tasks.append((theta, param_map, config_file, scale[cidx], feas_events,
                                  fix_nodes, node_seed, bin_method, cost_func, int(subseed),
                                  cond_key, ground_truth_data, mask_config, eval_subtypes,
                                  inst_id, noise_spec if needs_noise_cost else None,
                                  target_noise_eigvals if needs_noise_cost else None,
                                  max_components))

            raw_results = pool.map(_run_instance_condition, tasks)
    finally:
        if verbose:
            print("Shutting down shared memory manager...")
        smm.shutdown()

    # regroup results by instance and compute cost
    from collections import defaultdict
    per_instance = defaultdict(list)
    for r in raw_results:
        per_instance[r["instance_seed"]].append(r)

    # Find the noise_eig_mse weight (if any) for the MP aggregation path
    noise_eig_weight = 0.0
    for t in cost_terms:
        if t["name"] == "noise_eig_mse":
            noise_eig_weight = float(t["weight"])
            break

    stable_count = 0
    accum_cost = 0.0
    for seed_key, cond_list in per_instance.items():
        if not all(c["feasible"] for c in cond_list):
            continue
        stable_count += 1
        all_sims = np.column_stack([c["rates"] for c in cond_list])
        if all_sims.shape[1] != ground_truth_data.shape[1]:
            raise ValueError(f"evaluate_parameters (mp): mismatch in cols — model {all_sims.shape[1]} vs gt {ground_truth_data.shape[1]}")
        
        # Compute cost, handling per-subtype case
        model_indices = None
        if eval_subtypes:
            if target_model_indices is None:
                raise ValueError("eval_subtypes=True but target_model_indices not provided")
            # Extract model indices from the first result (same for all conditions in this instance)
            model_indices = cond_list[0]["model_indices"]
            if model_indices is None:
                raise ValueError("eval_subtypes=True but model_indices not returned from worker")
        
        # Rate-based cost terms (Phase 1 only — noise_eig_mse handled below from workers)
        rate_cost_terms = [t for t in cost_terms if t["name"] != "noise_eig_mse"]
        inst_cost = _compute_weighted_cost(
            all_sims,
            ground_truth_data,
            cost_terms=rate_cost_terms,
            bin_method=bin_method,
            corr_method=corr_method,
            target_eigvals=target_eigvals,
            max_components=max_components,
            model_indices=model_indices,
            target_indices=target_model_indices,
            target_eigvals_by_subtype=target_eigvals_by_subtype,
        )

        # Aggregate per-condition noise_eig_cost from workers (Phase 2)
        if needs_noise_cost and noise_eig_weight > 0:
            noise_cost_sum = 0.0
            for c in cond_list:
                if c.get("noise_eig_cost") is not None:
                    noise_cost_sum += c["noise_eig_cost"]
            inst_cost += noise_eig_weight * noise_cost_sum

        accum_cost += inst_cost

    feas_value = stable_count / n_inst
    avg_cost = accum_cost / stable_count if stable_count > 0 else 1e9

    # Apply feasibility thresholding if specified
    feas_thresh = kwargs.get("feas_thresh", None)
    if feas_thresh is not None:
        feas_value = 1 if feas_value >= feas_thresh else 0

    if debug:
        print(f"[DEBUG] feasibility = {feas_value:.2f}, avg cost = {avg_cost:.4g}")

    return feas_value, avg_cost


def resolve_file_list(directory: str, file_list: Optional[Sequence[str]]) -> List[Path]:
    base = Path(directory)
    if file_list:
        return [base / Path(name) for name in file_list]
    return sorted(p for p in base.glob("*.h5") if not p.name.startswith("noise_eigs."))


def load_target_data(file_paths: Sequence[Path], dataset: str = "y", load_model_index: bool = False) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
    """Load target data and optionally model indices from HDF5 files.
    
    Parameters
    ----------
    file_paths : Sequence[Path]
        Paths to HDF5 files containing target data.
    dataset : str
        Name of the dataset to load (default: "y").
    load_model_index : bool
        If True, also load the 'model_index' dataset from files.
    
    Returns
    -------
    tuple
        (matrix, labels, model_indices) where:
        - matrix: stacked target data (n_cells, n_conditions)
        - labels: file names
        - model_indices: model indices for each cell, or None if not loaded
    """
    columns = []
    indices = []
    labels = []
    for path in file_paths:
        with h5py.File(path, "r") as h5file:
            data = h5file[dataset][:]
            columns.append(data[:, -1]) # assuming we take the last time point in the series as the target
            if load_model_index:
                if "model_index" in h5file:
                    indices.append(h5file["model_index"][:])
                else:
                    raise KeyError(f"File {path.name} does not contain 'model_index' dataset")
            labels.append(path.name)
    matrix = np.stack(columns, axis=1)
    model_indices = np.concatenate(indices) if load_model_index else None
    return matrix, labels, model_indices


def load_input_data(file_paths: Sequence[Path], dataset: str = "stimulus") -> Tuple[OrderedDict, List[str]]:
    inputs = OrderedDict()
    labels = []
    for path in file_paths:
        with h5py.File(path, "r") as h5file:
            inputs[path.stem] = h5file[dataset][:]
        labels.append(path.name)
    return inputs, labels


def broadcast_scale(scale_values: Sequence[float], n_conditions: int) -> List[float]:
    if len(scale_values) == 0:
        raise ValueError("Scale vector cannot be empty")
    if len(scale_values) == 1:
        return [scale_values[0]] * n_conditions
    if len(scale_values) != n_conditions:
        raise ValueError("Scale vector length must match number of inputs")
    return list(scale_values)


def select_cell_subset(data: np.ndarray, fraction: float, seed: int, strategy: str = "random", 
                       indices_data: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    n_cells = data.shape[0]
    n_keep = max(1, int(round(fraction * n_cells)))
    indices = np.arange(n_cells)
    if strategy == "random" and n_keep < n_cells:
        rng = np.random.default_rng(seed)
        subset_idx = rng.choice(indices, size=n_keep, replace=False)
        subset_idx.sort()
    else:
        subset_idx = indices
    subset = data[subset_idx]
    subset_indices_data = None
    if indices_data is not None:
        subset_indices_data = indices_data[subset_idx]
    return subset_idx, subset, subset_indices_data


def _compute_weighted_cost(model_outputs: np.ndarray,
                           target_data: np.ndarray,
                           cost_terms: Sequence[Mapping[str, object]],
                           bin_method: str = "fd",
                           corr_method: str = "pearson",
                           target_eigvals: Optional[np.ndarray] = None,
                           max_components: Optional[int] = None,
                           model_indices: Optional[np.ndarray] = None,
                           target_indices: Optional[np.ndarray] = None,
                           target_eigvals_by_subtype: Optional[Mapping[int, np.ndarray]] = None,
                           ssn_model: Optional[object] = None,
                           W: Optional[np.ndarray] = None,
                           per_condition_meta: Optional[Sequence[Mapping[str, np.ndarray]]] = None,
                           noise_matrix: Optional[np.ndarray] = None,
                           target_noise_eigvals: Optional[Mapping[str, np.ndarray]] = None,
                           condition_keys: Optional[Sequence[str]] = None) -> float:
    """Compute weighted sum of requested cost terms in two phases.

    Phase 1 (rate-based): kl, mse, kl_rect, eig_mse — operate on the stacked
    model_outputs vs target_data matrices.

    Phase 2 (model-based): noise_eig_mse — requires the SSN object, adjacency
    matrix W, per-condition metadata (r_star, h), and noise matrix.  Loops over
    conditions and sums.

    Each term can optionally have ``use_subtypes`` flag to compute per-subtype
    (only supported for rate-based terms).
    """
    total_cost = 0.0

    # ---- Phase 1: rate-based terms ----
    for term in cost_terms:
        name = str(term["name"])
        weight = float(term["weight"])

        # Model-based terms are handled in Phase 2
        if name == "noise_eig_mse":
            continue

        use_subtypes = term.get("use_subtypes", False)

        if use_subtypes and model_indices is not None and target_indices is not None:
            term_cost = _compute_term_by_subtype(
                model_outputs=model_outputs,
                target_data=target_data,
                model_indices=model_indices,
                target_indices=target_indices,
                term=term,
                bin_method=bin_method,
                corr_method=corr_method,
                target_eigvals_by_subtype=target_eigvals_by_subtype,
                max_components=max_components,
            )
        else:
            if name == "eig_mse":
                term_cost = SSN_utils.eigenspectrum_mse(
                    model_outputs,
                    gt_eigvals=target_eigvals,
                    data=target_data if target_eigvals is None else None,
                    method=corr_method,
                    rowvar=False,
                    max_components=max_components,
                )
            else:
                cost_callable = term["callable"]
                term_cost = SSN_utils.compute_cost(
                    model_outputs,
                    target_data,
                    method=bin_method,
                    cost_function=cost_callable,
                )

        total_cost += weight * term_cost

    # ---- Phase 2: model-based terms (noise_eig_mse) ----
    for term in cost_terms:
        name = str(term["name"])
        if name != "noise_eig_mse":
            continue
        weight = float(term["weight"])

        if ssn_model is None or per_condition_meta is None or noise_matrix is None or target_noise_eigvals is None:
            raise ValueError(
                "noise_eig_mse requires ssn_model, per_condition_meta, noise_matrix, "
                "and target_noise_eigvals to be provided."
            )

        cond_cost = 0.0
        keys = condition_keys if condition_keys is not None else [str(i) for i in range(len(per_condition_meta))]
        for cidx, cond_key in enumerate(keys):
            meta = per_condition_meta[cidx]
            gt_eig = target_noise_eigvals.get(cond_key)
            if gt_eig is None:
                raise ValueError(f"noise_eig_mse: no ground-truth eigenvalues for condition '{cond_key}'")
            cond_cost += SSN_utils.noise_eigenspectrum_mse(
                ssn_model,
                r_star=meta["r_star"],
                g=noise_matrix,
                gt_eigvals=gt_eig,
                W=W,
                max_components=max_components,
                model_input=meta["h"],
            )
        total_cost += weight * cond_cost

    return float(total_cost)


def _compute_term_by_subtype(model_outputs: np.ndarray,
                             target_data: np.ndarray,
                             model_indices: np.ndarray,
                             target_indices: np.ndarray,
                             term: Mapping[str, object],
                             bin_method: str,
                             corr_method: str,
                             target_eigvals_by_subtype: Optional[Mapping[int, np.ndarray]],
                             max_components: Optional[int]) -> float:
    """Compute a single cost term separately per subtype and sum."""
    name = str(term["name"])
    model_subtypes = np.unique(model_indices)
    target_subtypes = np.unique(target_indices)
    
    total_cost = 0.0
    
    for subtype in model_subtypes:
        if subtype not in target_subtypes:
            continue
        
        model_mask = model_indices == subtype
        target_mask = target_indices == subtype
        
        model_subset = model_outputs[model_mask]
        target_subset = target_data[target_mask]
        
        if model_subset.shape[0] == 0 or target_subset.shape[0] == 0:
            continue
        
        # Compute cost for this subtype
        if name == "eig_mse":
            subtype_eigvals = None
            if target_eigvals_by_subtype is not None and int(subtype) in target_eigvals_by_subtype:
                subtype_eigvals = target_eigvals_by_subtype[int(subtype)]
            
            term_cost = SSN_utils.eigenspectrum_mse(
                model_subset,
                gt_eigvals=subtype_eigvals,
                data=target_subset if subtype_eigvals is None else None,
                method=corr_method,
                rowvar=False,
                max_components=max_components,
            )
        else:
            cost_callable = term["callable"]
            term_cost = SSN_utils.compute_cost(
                model_subset,
                target_subset,
                method=bin_method,
                cost_function=cost_callable,
            )
        
        total_cost += term_cost
    
    return float(total_cost)


def _compute_target_eigvals_by_subtype(target_data: np.ndarray,
                                       target_indices: np.ndarray,
                                       corr_method: str = "pearson",
                                       max_components: Optional[int] = None) -> Dict[int, np.ndarray]:
    """Precompute target eigenspectra once per subtype."""
    eig_by_subtype: Dict[int, np.ndarray] = {}
    for subtype in np.unique(target_indices):
        subtype_mask = target_indices == subtype
        subtype_target = target_data[subtype_mask]
        if subtype_target.shape[0] == 0:
            continue
        eig_by_subtype[int(subtype)] = SSN_utils.correlation_eigenspectrum(
            subtype_target,
            method=corr_method,
            rowvar=False,
            sort_desc=True,
            max_components=max_components,
        )
    return eig_by_subtype


def compute_subtype_cost(model_outputs: np.ndarray, target_data: np.ndarray, model_indices: np.ndarray,
                         target_indices: np.ndarray, bin_method: str = "fd",
                         cost_function=None, cost_terms: Optional[Sequence[Mapping[str, object]]] = None,
                         corr_method: str = "pearson",
                         target_eigvals_by_subtype: Optional[Mapping[int, np.ndarray]] = None,
                         max_components: Optional[int] = None) -> float:
    """Compute cost separately for each subtype and sum.
    
    Parameters
    ----------
    model_outputs : np.ndarray
        Model outputs with shape (n_cells, n_conditions).
    target_data : np.ndarray
        Target data with shape (n_cells, n_conditions).
    model_indices : np.ndarray
        Subtype indices for model cells.
    target_indices : np.ndarray
        Subtype indices for target cells.
    bin_method : str
        Histogram binning method (passed to compute_cost).
    cost_function : callable
        Cost function to use (e.g., SSN_utils.kl_divergence).
    
    Returns
    -------
    float
        Sum of costs across all subtypes present in both model and target.
    """
    if cost_terms is None:
        if cost_function is None:
            cost_function = SSN_utils.kl_divergence
        cost_terms = [{"name": "legacy", "weight": 1.0, "callable": cost_function}]
    
    model_subtypes = np.unique(model_indices)
    target_subtypes = np.unique(target_indices)
    
    total_cost = 0.0
    computed_subtypes = []
    
    for subtype in model_subtypes:
        if subtype not in target_subtypes:
            print(f"Warning: subtype {subtype} present in model but not in target data. Skipping.")
            continue
        
        model_mask = model_indices == subtype
        target_mask = target_indices == subtype
        
        model_subset = model_outputs[model_mask]
        target_subset = target_data[target_mask]
        
        if model_subset.shape[0] == 0 or target_subset.shape[0] == 0:
            print(f"Warning: subtype {subtype} has zero cells in model or target. Skipping.")
            continue
        
        subtype_target_eigvals = None
        if target_eigvals_by_subtype is not None and int(subtype) in target_eigvals_by_subtype:
            subtype_target_eigvals = target_eigvals_by_subtype[int(subtype)]

        subtype_cost = _compute_weighted_cost(
            model_subset,
            target_subset,
            cost_terms=cost_terms,
            bin_method=bin_method,
            corr_method=corr_method,
            target_eigvals=subtype_target_eigvals,
            max_components=max_components,
        )
        total_cost += subtype_cost
        computed_subtypes.append(subtype)
    
    for subtype in target_subtypes:
        if subtype not in model_subtypes:
            print(f"Warning: subtype {subtype} present in target data but not in model. Skipping.")
    
    return total_cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SSN Bayesian optimization across subsets of target data.")
    parser.add_argument("--config", required=True, help="Path to SSN config file.")
    parser.add_argument("--inputs-dir", required=True, help="Directory containing input HDF5 files.")
    parser.add_argument("--targets-dir", required=True, help="Directory containing target HDF5 files.")
    parser.add_argument("--input-files", nargs="*", help="Optional explicit list of input file names relative to inputs-dir.")
    parser.add_argument("--target-files", nargs="*", help="Optional explicit list of target file names relative to targets-dir.")
    parser.add_argument("--scale", nargs="*", type=float, default=[1.0], help="Scale applied per input condition (one value or one per input).")
    parser.add_argument("--fractions", nargs="*", type=float, default=[1.0], help="Fractions of cells to use for successive runs (e.g. 1.0 0.8 0.5).")
    parser.add_argument("--results-dir", default="optimization_runs", help="Directory to store optimization summaries.")
    parser.add_argument("--n-init", type=int, default=20, help="Number of initial random samples for BO.")
    parser.add_argument("--n-inst", type=int, default=1, help="Number of stochastic instantiations per evaluation.")
    parser.add_argument("--n-iter", type=int, default=50, help="Number of BO iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed for BO and subset selection.")
    parser.add_argument("--fix-seed", action="store_true", help="Use the same seed for every evaluation run.")
    parser.add_argument("--fix-nodes", action="store_true", help="Keep node layout fixed across evaluations.")
    parser.add_argument("--node-seed", type=int, default=0, help="Seed used when fix-nodes is enabled.")
    parser.add_argument("--feas-threshold", type=float, default=1e5, help="Feasibility threshold for firing rates.")
    parser.add_argument("--feas-thresh", type=float, default=None, help="Threshold for feasibility binarization (0-1). If provided, feasibility >= threshold becomes 1, else 0.")
    parser.add_argument("--bin-method", default="fd", help="Bin selection rule passed to compute_cost().")
    parser.add_argument("--subset-strategy", choices=["random"], default="random", help="How to choose cell subsets.")
    parser.add_argument("--track-acquisition", action="store_true", help="Enable acquisition tracking in bayesopt.")
    parser.add_argument("--n-candidates", type=int, default=1000,
                        help="Number of candidate parameterizations sampled each BO iteration (passed to bayesopt as n_candidates).")
    parser.add_argument("--track-gp-params", action="store_true", help="Track GP kernel hyperparameters on each iteration of bayesopt.")
    parser.add_argument("--kernel-fit-schedule", default=None,
                        help=("Path to a JSON or NPY file containing an array of 0-based iteration indices at which to fit GP kernel hyperparameters, "
                              "or a comma-separated list of integers. If omitted, kernels are fit on every iteration."))
    parser.add_argument("--param-bounds", help="Optional JSON file overriding parameter bounds.")
    parser.add_argument("--param-map", help="Optional JSON file overriding parameter map.")
    parser.add_argument("--mask-config", help="Optional JSON file specifying spatial mask for model cells (e.g., circular or rectangular window).")
    parser.add_argument("--debug-eval", action="store_true", help="Print debug info during evaluations.")
    parser.add_argument("--save-bo", action="store_true", help="Save bayesopt objects as joblib per fraction.")
    parser.add_argument("--n-cores", type=int, default=1, help="Number of CPU cores to use during parameter evaluation (shared-memory pool).")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output from SSN and parameter evaluations. By default, only a progress bar is shown.")
    parser.add_argument("--no-events", action="store_false", help="Disable event tracking in SSN.run(). Event tracking identifies feasibility violations within simulations. This flag may cause slowdowns.")
    parser.add_argument("--eval-subtypes", action="store_true", help="Compute cost separately for each cell subtype and sum. Requires 'model_index' field in target HDF5 files.")
    parser.add_argument("--use-feas", type=int, choices=[0, 1], default=1, help="Use feasibility model (1=True, 0=False). Default is 1 (True) for Bayesian optimization.")
    parser.add_argument("--cost-func", type=str, choices=["kl", "mse", "kl_rect", "eig_mse", "noise_eig_mse"], default="kl", help="Single cost function (backward compatible). Choices: 'kl', 'mse', 'kl_rect', 'eig_mse', 'noise_eig_mse'.")
    parser.add_argument("--cost-funcs", nargs="+", choices=["kl", "mse", "kl_rect", "eig_mse", "noise_eig_mse"], default=None, help="Optional list of cost functions for a composite objective (weighted sum). If provided, overrides --cost-func.")
    parser.add_argument("--cost-weights", nargs="*", type=str, default=None, help="Weights for --cost-funcs. Either inline floats (one per cost func) or a single path to a JSON file with a 'weights' dict mapping cost names to floats. Defaults to all 1.0.")
    parser.add_argument("--cost-subtypes", nargs="*", type=int, choices=[0, 1], default=None, help="Whether each cost function should be computed per-subtype (1) or across all cells (0). Must match length of --cost-funcs. Defaults to 1 if --eval-subtypes, else 0.")
    parser.add_argument("--corr-method", type=str, default="pearson", help="Correlation method for eigenspectrum cost (currently pearson).")
    parser.add_argument("--max-components", type=int, default=None, help="Number of top eigenvalues to use in eigenspectrum MSE. Defaults to all available.")
    parser.add_argument("--set-k", type=int, default=None, help="Set the k parameter for rectified KL divergence (used with --cost-func kl_rect).")
    parser.add_argument("--use-log-cost", action="store_true", help="Apply log transformation to cost values before passing to the GP surrogate model.")
    parser.add_argument("--feas-acq-threshold", type=float, default=None, help="Feasibility threshold for acquisition weighting. When set, candidates with feasibility above this value are treated as fully feasible (weight=1); below it the raw feasibility estimate scales the acquisition. Default: None (always multiply by feasibility).")
    parser.add_argument("--noise-matrix", type=str, default=None, help="Noise matrix specification for noise_eig_mse cost. A positive float builds sigma*I_N; a file path (.npy/.npz/.h5) loads a full matrix.")
    parser.add_argument("--noise-eigvals", type=str, default=None, help="Directory containing per-condition noise eigenspectra files named noise_eigs.{model_name}.{condition_name}.h5. Each file stores an 'eigenvalues' dataset.")
    return parser.parse_args()


def load_param_bounds(path: Optional[str]) -> Dict[str, List[float]]:
    if path is None:
        return DEFAULT_PARAM_BOUNDS.copy()
    with open(path, "r") as fp:
        return json.load(fp)


def load_param_map(path: Optional[str]) -> Dict[str, List[str]]:
    if path is None:
        return DEFAULT_PARAM_MAP.copy()
    with open(path, "r") as fp:
        return json.load(fp)


def parse_kernel_fit_schedule(raw: Optional[str]):
    """Parse kernel fit schedule specification.

    Accepts None, a comma-separated list of ints, or a path to a .json/.npy file
    that contains an array of integers. Returns a list of ints or None.
    """
    if raw is None:
        return None
    # If looks like a file path, try loading
    p = Path(raw)
    if p.exists():
        if p.suffix.lower() == '.npy':
            arr = np.load(p)
            return [int(x) for x in np.atleast_1d(arr).tolist()]
        elif p.suffix.lower() == '.json':
            with open(p, 'r') as f:
                data = json.load(f)
            if isinstance(data, list):
                return [int(x) for x in data]
            else:
                raise ValueError(f"kernel-fit-schedule JSON must be a list, got {type(data)}")
        else:
            raise ValueError(f"Unsupported file extension for --kernel-fit-schedule: {p.suffix}")
    # Otherwise interpret as comma-separated list
    try:
        parts = [s.strip() for s in raw.split(',') if s.strip()!='']
        return [int(x) for x in parts]
    except Exception:
        raise ValueError("--kernel-fit-schedule must be a path to .json/.npy or a comma-separated list of integers")


def run_single_fraction(fraction: float, subset_indices: np.ndarray, target_subset: np.ndarray,
                        input_data: OrderedDict, eval_kwargs: MutableMapping[str, object], args: argparse.Namespace,
                        param_bounds: Mapping[str, Sequence[float]], param_map: Mapping[str, Sequence[str]],
                        results_dir: Path, subset_indices_data: Optional[np.ndarray] = None,
                        kernel_fit_schedule: Optional[Sequence[int]] = None) -> Dict[str, object]:
    """Run Bayesian optimization for a single cell fraction.
    
    Parameters
    ----------
    fraction : float
        Fraction of ground truth cells used (from --fractions flag).
    subset_indices : np.ndarray
        Indices of cells selected from original ground truth (for bookkeeping).
    target_subset : np.ndarray
        Already-filtered ground truth data with shape (n_cells * fraction, n_conditions).
        This is passed to evaluate_parameters as ground_truth_data.
    subset_indices_data : np.ndarray, optional
        Subtype indices corresponding to target_subset (for per-subtype cost computation).
    """
    eval_kwargs = dict(eval_kwargs)
    # Note: target_subset is already filtered by fraction; subset_indices kept for bookkeeping
    # Model cell filtering (via mask_config) is independent and handled in evaluate_parameters
    
    # Override target_model_indices with the fraction-filtered indices if available
    if subset_indices_data is not None:
        eval_kwargs["target_model_indices"] = subset_indices_data

    cost_terms = eval_kwargs.get("cost_terms", [])
    
    # Determine if we need eigenspectrum precomputation
    # - Need full-data eigvals if any eig_mse term has use_subtypes=False
    # - Need by-subtype eigvals if any eig_mse term has use_subtypes=True
    needs_full_eigvals = any(t.get("name") == "eig_mse" and not t.get("use_subtypes", False) for t in cost_terms)
    needs_subtype_eigvals = any(t.get("name") == "eig_mse" and t.get("use_subtypes", False) for t in cost_terms)
    
    if needs_full_eigvals or needs_subtype_eigvals:
        corr_method = str(eval_kwargs.get("corr_method", "pearson"))
        max_components = eval_kwargs.get("max_components", None)
        
        if needs_subtype_eigvals and subset_indices_data is not None:
            eval_kwargs["target_eigvals_by_subtype"] = _compute_target_eigvals_by_subtype(
                target_subset,
                subset_indices_data,
                corr_method=corr_method,
                max_components=max_components,
            )
        
        if needs_full_eigvals:
            eval_kwargs["target_eigvals"] = SSN_utils.correlation_eigenspectrum(
                target_subset,
                method=corr_method,
                rowvar=False,
                sort_desc=True,
                max_components=max_components,
            )

    bo = bayesopt(param_bounds, param_map, user_evaluate=evaluate_parameters, evaluation_kwargs=eval_kwargs)
    start_time = time.time()
    best_params = bo.bayesopt(
        n_init=args.n_init,
        n_inst=args.n_inst,
        n_iter=args.n_iter,
        random_state=args.seed,
        acquisition_func=SSN_utils.expected_improvement,
        ground_truth_data=target_subset,
        input_data=input_data,
        fix_seed=args.fix_seed,
        fix_nodes=args.fix_nodes,
        node_seed=args.node_seed,
        use_feas=bool(args.use_feas),
        track_acquisition=args.track_acquisition,
        track_gp_params=args.track_gp_params,
        kernel_fit_schedule=kernel_fit_schedule,
        n_candidates=args.n_candidates,
        use_log_cost=args.use_log_cost,
        feas_acq_threshold=args.feas_acq_threshold,
        verbose=True, # Always show BO progress bars regardless of CLI --verbose flag
    )
    elapsed = time.time() - start_time

    result = {
        "fraction": fraction,
        "subset_size": int(len(subset_indices)),
        "best_params": best_params,
        "elapsed_seconds": elapsed,
        "args": vars(args),
    }

    out_path = results_dir / f"fraction_{int(round(fraction * 100))}.json"
    with open(out_path, "w") as fp:
        json.dump(result, fp, indent=2)

    if args.save_bo:
        bo_file = results_dir / f"fraction_{int(round(fraction * 100))}_bo.joblib"
        joblib.dump(bo, bo_file)

    return result


def main():
    args = parse_args()
    input_files = resolve_file_list(args.inputs_dir, args.input_files)
    target_files = resolve_file_list(args.targets_dir, args.target_files)
    if not input_files:
        raise FileNotFoundError("No input HDF5 files found.")
    if not target_files:
        raise FileNotFoundError("No target HDF5 files found.")

    input_data, _ = load_input_data(input_files)
    target_data_mat, _, target_model_indices = load_target_data(target_files, load_model_index=args.eval_subtypes)

    scale_vector = broadcast_scale(args.scale, len(input_data))
    param_bounds = load_param_bounds(args.param_bounds)
    param_map = load_param_map(args.param_map)
    mask_config = parse_mask_config(args.mask_config)

    # Map histogram-based cost function strings to actual callables
    hist_cost_func_map = {
        "kl": SSN_utils.kl_divergence,
        "mse": SSN_utils.compute_MSE,
        "kl_rect": SSN_utils.rectified_kl_divergence,
    }

    requested_cost_names = args.cost_funcs if args.cost_funcs is not None else [args.cost_func]
    if args.cost_weights is None:
        requested_cost_weights = [1.0] * len(requested_cost_names)
    elif len(args.cost_weights) == 1 and Path(args.cost_weights[0]).suffix == ".json":
        # Load weights from JSON file
        weights_path = Path(args.cost_weights[0])
        if not weights_path.exists():
            raise FileNotFoundError(f"--cost-weights file not found: {weights_path}")
        with open(weights_path, "r") as fp:
            weights_data = json.load(fp)
        if "weights" not in weights_data or not isinstance(weights_data["weights"], dict):
            raise ValueError(
                f"--cost-weights JSON must contain a 'weights' key with a dict "
                f"mapping cost names to floats. Got keys: {list(weights_data.keys())}"
            )
        weights_dict = weights_data["weights"]
        requested_cost_weights = []
        for name in requested_cost_names:
            if name not in weights_dict:
                raise ValueError(
                    f"--cost-weights JSON missing weight for cost '{name}'. "
                    f"Available: {sorted(weights_dict.keys())}"
                )
            requested_cost_weights.append(float(weights_dict[name]))
    else:
        try:
            requested_cost_weights = [float(w) for w in args.cost_weights]
        except ValueError:
            raise ValueError(
                "--cost-weights values must be floats or a single JSON file path"
            )
        if len(requested_cost_weights) != len(requested_cost_names):
            raise ValueError("--cost-weights length must match --cost-funcs length")
    
    # Parse cost_subtypes: which cost functions should be computed per-subtype
    if args.cost_subtypes is None:
        # Default: use subtypes for all if --eval-subtypes is set, else none
        requested_cost_subtypes = [1 if args.eval_subtypes else 0] * len(requested_cost_names)
    else:
        requested_cost_subtypes = list(args.cost_subtypes)
        if len(requested_cost_subtypes) != len(requested_cost_names):
            raise ValueError("--cost-subtypes length must match --cost-funcs length")

    cost_terms = []
    for name, weight, use_sub in zip(requested_cost_names, requested_cost_weights, requested_cost_subtypes):
        term = {"name": name, "weight": float(weight), "use_subtypes": bool(use_sub)}
        if name == "noise_eig_mse" and use_sub:
            raise ValueError("noise_eig_mse does not support per-subtype evaluation (--cost-subtypes must be 0 for this term).")
        if name in hist_cost_func_map:
            cost_callable = hist_cost_func_map[name]
            if name == "kl_rect" and args.set_k is not None:
                cost_callable = partial(cost_callable, k=args.set_k)
            term["callable"] = cost_callable
        cost_terms.append(term)

    # Validate noise_eig_mse requirements
    has_noise_eig_mse = any(t["name"] == "noise_eig_mse" for t in cost_terms)
    if has_noise_eig_mse:
        if args.noise_matrix is None:
            raise ValueError("--noise-matrix is required when using noise_eig_mse cost.")
        if args.noise_eigvals is None:
            raise ValueError("--noise-eigvals is required when using noise_eig_mse cost.")

    # Parse noise specification
    noise_spec = parse_noise_spec(args.noise_matrix)
    condition_keys = list(input_data.keys())
    target_noise_eigvals = load_noise_eigvals(args.noise_eigvals)

    # Validate that noise eigvals cover all conditions
    if target_noise_eigvals is not None:
        missing = set(condition_keys) - set(target_noise_eigvals.keys())
        if missing:
            raise ValueError(
                f"--noise-eigvals is missing entries for conditions: {sorted(missing)}. "
                f"Available keys: {sorted(target_noise_eigvals.keys())}"
            )

    # Backward-compatible single callable for older code paths
    cost_func = cost_terms[0].get("callable", SSN_utils.kl_divergence)

    base_eval_kwargs = {
        "config_file": args.config,
        "scale": scale_vector,
        "feas_threshold": args.feas_threshold,
        "bin_method": args.bin_method,
        "debug": args.debug_eval,
        "n_cores": args.n_cores,
        "cost_func": cost_func,
        "cost_terms": cost_terms,
        "corr_method": args.corr_method,
        "max_components": args.max_components,
        "mask_config": mask_config,
        "verbose": args.verbose,
        "eval_subtypes": args.eval_subtypes,
        "target_model_indices": target_model_indices,
        "feas_thresh": args.feas_thresh,
        "noise_spec": noise_spec,
        "target_noise_eigvals": target_noise_eigvals,
    }

    # Parse kernel fit schedule (passed to bayesopt)
    kernel_fit_schedule = parse_kernel_fit_schedule(args.kernel_fit_schedule)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for idx, fraction in enumerate(args.fractions):
        subset_seed = args.seed + idx
        # Subsample ground truth cells by fraction (--fractions flag)
        subset_indices, subset_matrix, subset_indices_data = select_cell_subset(
            target_data_mat, fraction, subset_seed, strategy=args.subset_strategy, 
            indices_data=target_model_indices)
        # subset_matrix is already filtered; it flows as target_subset -> ground_truth_data
        result = run_single_fraction(fraction, subset_indices, subset_matrix, input_data,
                         base_eval_kwargs, args, param_bounds, param_map, results_dir,
                         subset_indices_data=subset_indices_data,
                         kernel_fit_schedule=kernel_fit_schedule)
        summary.append(result)
        print(f"Completed fraction {fraction:.2f}: best params = {result['best_params']}")

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
