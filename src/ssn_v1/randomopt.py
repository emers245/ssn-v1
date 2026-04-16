#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 2026

randomopt.py

Implements a Random Search Optimization class for fitting a set of model
parameters to data. Provides a baseline comparison to Bayesian Optimization
by randomly sampling parameters within specified bounds.

This class mirrors the interface of bayesopt.py to enable direct comparisons
between optimization strategies.

References:
  Bergstra & Bengio (2012) - Random Search for Hyper-Parameter Optimization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
try:
    from . import SSN_utils
except Exception:
    import SSN_utils
import cloudpickle


class randomopt:
    """
    A Random Search Optimization class for parameter optimization.
    It randomly samples parameters from specified bounds and evaluates
    them using a user-supplied cost function.
    """

    def __init__(self,
                 bounds: Dict[str, Tuple[float, float]],
                 param_map: Dict[str, Tuple],
                 user_evaluate: Callable,
                 prior: Dict[str, Callable] = None,
                 evaluation_kwargs: dict = None,
                 ground_truth_data: np.ndarray = None,
                 **kwargs):
        """
        Parameters
        ----------
        bounds : dict
            Parameter bounds
            param_name --> (lower_bound, upper_bound)
        param_map : dict
            Location of free parameters within a parameters dictionary
            param_name --> (key1, key2, ..., keyN)
        user_evaluate : callable
            A function that evaluates a parameter set and returns a cost (or feasibility
            and cost). Required signature:
            
                user_evaluate(params, param_map, ground_truth_data, input_data,
                             n_inst=..., seed=..., fix_seed=..., fix_nodes=...,
                             node_seed=..., **kwargs)
            
            Parameters:
                params : np.ndarray, shape (n_params,)
                    Flat array of parameter values to evaluate
                param_map : dict
                    Specifies where each parameter goes in the configuration
                ground_truth_data : np.ndarray
                    Reference data to fit against
                input_data : np.ndarray
                    Input stimuli/conditions for the ground truth data
                n_inst : int
                    Number of instantiations to simulate
                seed : int
                    Random seed for reproducibility
                fix_seed : bool
                    Whether to use fixed seed on every evaluation
                fix_nodes : bool
                    Whether to fix node instantiation seed
                node_seed : int
                    Random seed for node instantiation
                **kwargs : dict
                    Additional arguments from evaluation_kwargs
            
            Returns:
                If use_feas=False: scalar cost_value (float)
                If use_feas=True: tuple (feasibility_value, cost_value)
                    where feasibility_value is typically 1 (feasible) or 0 (infeasible)
        prior : dict, optional
            Specifies the prior distributions to sample for each parameter.
            This may be specified for each parameter independently in a 
            dictionary, or universally for all parameters with a single 
            function. If None, assume to be a bounded uniform distribution.
        evaluation_kwargs : dict, optional
            If the user-defined evaluation has keyword arguments, pass them 
            here as a dictionary
        ground_truth_data : np.ndarray, optional
            If needed for reference, but typically you'd pass this to your
            evaluation callback instead.
        kwargs : dict
            Additional config (currently unused but maintained for compatibility).
        """
        self.bounds = bounds
        self.prior = prior
        self.param_map = param_map
        self.ground_truth_data = ground_truth_data
        self.user_evaluate = user_evaluate
        self.evaluation_kwargs = evaluation_kwargs if evaluation_kwargs else {}
        self.config = kwargs

    def optimize(self,
                 n_iter: int = 100,
                 n_inst: int = 5,
                 random_state: int = 0,
                 ground_truth_data: np.ndarray = None,
                 input_data: np.ndarray = None,
                 verbose: bool = False,
                 **kwargs):
        """
        Main loop for Random Search Optimization.
        - Randomly sample parameters from bounds or prior
        - Evaluate them via user_evaluate
        - Track all sampled parameters and costs
        - Return the best discovered parameters
        
        Parameters
        ----------
        n_iter : int, optional
            Total number of random parameter samples to evaluate
        n_inst : int, optional
            Number of random instantiations of each parameterization
        random_state : int, optional
            Random seed for reproducibility
        ground_truth_data : np.ndarray, optional
            Data array to fit. Must have size N x Ncond where Ncond is the
            number of unique data samples to fit simultaneously. This may also
            be set when initializing the randomopt object. Setting the data here
            will override any data set during initialization of the randomopt 
            object.
        input_data : np.ndarray, optional
            Inputs to the model. Should have the same second dimension size as 
            ground_truth_data.
        verbose : bool, optional
            If True, print detailed output from parameter evaluations and iterations.
            If False (default), show a progress bar instead.
        kwargs : dict
            Advanced options for randomopt method.
            
            fix_seed : bool
                If True, the random seed will be fixed for all iterations.
            fix_nodes : bool
                If True, the random seed for node instantiation will be fixed.
            node_seed : int
                Random seed for node instantiation. Only used if fix_nodes is True.
                If not provided, defaults to random_state.
            use_feas : bool
                If True, expect feasibility and cost from user_evaluate.
            evaluate_feasibility : callable
                Function to determine if a feasibility value is acceptable.
                Default: lambda x: x == 1
        
        Returns
        -------
        dict
            Dictionary of best parameters found during optimization.
        """
        if self.user_evaluate is None:
            raise ValueError("No user_evaluate function provided. "
                             "Set randomopt.user_evaluate = your_callback.")
        if ground_truth_data is None:
            ground_truth_data = self.ground_truth_data

        # Store optimization parameters
        self.n_inst = n_inst
        self.n_iter = n_iter
            
        # Get number of params
        n_params = len(self.bounds.keys())
            
        # Collect kwargs
        fix_seed = kwargs.get('fix_seed', False)  # fix the random state on every iteration
        fix_nodes = kwargs.get('fix_nodes', False)  # fix the random state for node instantiation
        node_seed = kwargs.get('node_seed', random_state)  # random state for node instantiation
        use_feas = kwargs.get('use_feas', True)  # use feasibility model
        evaluate_feasibility = kwargs.get('evaluate_feasibility', lambda x: x == 1)  # Default: feasibility value must equal 1 to be accepted
        
        # Import tqdm if not verbose (needed for progress bar)
        if not verbose:
            try:
                from tqdm import tqdm as tqdm_bar
            except ImportError:
                # Fallback if tqdm is not available
                tqdm_bar = None
        else:
            tqdm_bar = None
            
        # Set random state
        rng = np.random.default_rng(random_state)
        self.rng = rng
        
        # Initialize histories
        self.history_params = []
        self.history_costParams = []
        self.history_costs = []
        self.history_minCosts = []
        self.history_feas = []
    
        # Random search loop
        if verbose:
            print("Starting Random Search...")
        
        search_iterator = tqdm_bar(range(n_iter), desc="Random Search", disable=False) if tqdm_bar else range(n_iter)
        
        for iteration in search_iterator:
            
            # Choose random seed
            if not fix_seed:
                subseed = self.rng.integers(0, 9999999)
            else:
                subseed = random_state
            
            # Choose random node seed if fix_nodes is False
            if not fix_nodes:
                node_seed = self.rng.integers(0, 9999999)
            
            # Sample random parameters from prior or uniform distribution
            params = np.zeros((n_params,))    

            if not self.prior:
                # Each element of 'bounds' is (low, high)
                low = np.array([b[0] for b in self.bounds.values()])
                high = np.array([b[1] for b in self.bounds.values()])
            
                # Draw all parameters at once
                params = self.rng.uniform(low=low, high=high)
            
            else: 
                for p_i, param_name in enumerate(self.bounds.keys()):
                    if isinstance(self.prior, dict):
                        params[p_i] = self.prior[param_name]()
                    # If only a single prior distribution is specified, apply across all params
                    elif isinstance(self.prior, Callable):
                        params[p_i] = self.prior()
                    # Otherwise raise error
                    else:
                        raise TypeError(f"randomopt: {type(self.prior[param_name])} is not a valid dtype for prior")
                
            # Evaluate Parameters
            # user_evaluate always returns (feasibility, cost)
            feas_val, cost_val = self.user_evaluate(
                params, 
                self.param_map, 
                ground_truth_data,
                input_data,
                n_inst=n_inst,
                seed=subseed,
                fix_seed=fix_seed,
                fix_nodes=fix_nodes,
                node_seed=node_seed,
                **self.evaluation_kwargs
            )

            # If using log cost, apply log transform to cost value (with small epsilon to avoid log(0))
            if kwargs.get("log_cost", False):
                cost_val = np.log(cost_val + 1e-10)
            
            # If not using feasibility, treat all as feasible
            if not use_feas:
                feas_val = 1
                
            # Save parameters and costs
            self.params = self._get_params_dict(params)
            if use_feas:
                self.history_params.append(params)
                self.history_feas.append(feas_val)
                if evaluate_feasibility(feas_val):
                    self.history_costParams.append(params)
                    self.history_costs.append(cost_val)
            else:
                self.history_params.append(params)
                self.history_costParams.append(params)
                self.history_costs.append(cost_val)
            
            # Track best cost so far
            if len(self.history_costs) > 0:
                best_cost = np.min(self.history_costs)
                self.history_minCosts.append(best_cost)
                
                if verbose:
                    print(f"Iteration {iteration+1}/{n_iter}: loss={cost_val:.4f}, best so far={best_cost:.4f}")
                elif tqdm_bar:
                    search_iterator.set_postfix({"loss": f"{cost_val:.4f}", "best": f"{best_cost:.4f}"})
            else:
                # No feasible solutions yet
                if verbose:
                    print(f"Iteration {iteration+1}/{n_iter}: loss={cost_val:.4f} (infeasible)")
                elif tqdm_bar:
                    search_iterator.set_postfix({"loss": f"{cost_val:.4f}"})
        
        # Check if we found any feasible solutions
        if len(self.history_costs) == 0:
            raise ValueError("Error: No Feasible Solutions Found during random search. "
                           "Ensure the parameter bounds are appropriately set.")
    
        # Get best parameters
        overall_best_idx = np.argmin(self.history_costs)
        self.params = self._get_params_dict(self.history_costParams[overall_best_idx])
        best_loss_value = self.history_costs[overall_best_idx]
    
        if verbose:
            print("\nFinal Random Search Results:")
            print(f"  best_params={self.params}")
            print(f"  best_loss={best_loss_value:.6f}")
            print(f"  total_evaluations={len(self.history_params)}")
            print(f"  feasible_evaluations={len(self.history_costs)}")

        return self.params
    
    def _get_params_dict(self, params: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Convert a flat parameter array into a dictionary with names.
    
        Parameters
        ----------
        params : np.ndarray or None
            Flat array of parameters. If None, defaults to self.params.
    
        Returns
        -------
        Dict[str, float]
            Parameter dictionary with named keys.
        """
        if params is None:
            params = self.params
        return dict(zip(self.bounds.keys(), params))

    def plot_convergence(self, ax=None, label='Random Search', **plot_kwargs):
        """
        Plot the convergence of the best cost over iterations.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, creates new figure.
        label : str, optional
            Label for the plot line.
        **plot_kwargs : dict
            Additional keyword arguments passed to ax.plot()
            
        Returns
        -------
        fig, ax
            Figure and axis objects
        """
        if not hasattr(self, 'history_minCosts'):
            raise ValueError("No optimization history found. Run optimize() first.")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()
        
        iterations = np.arange(1, len(self.history_minCosts) + 1)
        ax.plot(iterations, self.history_minCosts, label=label, **plot_kwargs)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Cost')
        ax.set_title('Random Search Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def plot_parameter_samples(self, param_names=None, figsize=(12, 8)):
        """
        Plot histograms of sampled parameter values.
        
        Parameters
        ----------
        param_names : list, optional
            List of parameter names to plot. If None, plots all parameters.
        figsize : tuple, optional
            Figure size (width, height).
            
        Returns
        -------
        fig, axes
            Figure and axes objects
        """
        if not hasattr(self, 'history_params'):
            raise ValueError("No sampling history found. Run optimize() first.")
        
        if param_names is None:
            param_names = list(self.bounds.keys())
        
        n_params = len(param_names)
        n_cols = min(3, n_params)
        n_rows = int(np.ceil(n_params / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        history_array = np.array(self.history_params)
        
        for i, param_name in enumerate(param_names):
            param_idx = list(self.bounds.keys()).index(param_name)
            ax = axes[i]
            
            ax.hist(history_array[:, param_idx], bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(self.params[param_name], color='red', linestyle='--', 
                      linewidth=2, label='Best')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Frequency')
            ax.set_title(f'{param_name} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig, axes
    
    def plot_cost_vs_parameters(self, param_names=None, figsize=(12, 8)):
        """
        Scatter plot of cost vs each parameter.
        
        Parameters
        ----------
        param_names : list, optional
            List of parameter names to plot. If None, plots all parameters.
        figsize : tuple, optional
            Figure size (width, height).
            
        Returns
        -------
        fig, axes
            Figure and axes objects
        """
        if not hasattr(self, 'history_costParams'):
            raise ValueError("No evaluation history found. Run optimize() first.")
        
        if param_names is None:
            param_names = list(self.bounds.keys())
        
        n_params = len(param_names)
        n_cols = min(3, n_params)
        n_rows = int(np.ceil(n_params / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        history_array = np.array(self.history_costParams)
        costs_array = np.array(self.history_costs)
        
        for i, param_name in enumerate(param_names):
            param_idx = list(self.bounds.keys()).index(param_name)
            ax = axes[i]
            
            scatter = ax.scatter(history_array[:, param_idx], costs_array, 
                               alpha=0.5, c=costs_array, cmap='viridis')
            
            # Mark best point
            best_idx = np.argmin(costs_array)
            ax.scatter(history_array[best_idx, param_idx], costs_array[best_idx],
                      color='red', s=100, marker='*', edgecolor='black',
                      label='Best', zorder=5)
            
            ax.set_xlabel(param_name)
            ax.set_ylabel('Cost')
            ax.set_title(f'Cost vs {param_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax, label='Cost')
        
        # Hide extra subplots
        for i in range(n_params, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig, axes

    def __getstate__(self):
        """Custom pickling to handle user_evaluate function with cloudpickle."""
        state = self.__dict__.copy()
        user_eval = state.get("user_evaluate")
        if user_eval is not None:
            state["_user_evaluate_blob"] = cloudpickle.dumps(user_eval)
            state["user_evaluate"] = None
        return state

    def __setstate__(self, state):
        """Custom unpickling to restore user_evaluate function."""
        blob = state.pop("_user_evaluate_blob", None)
        self.__dict__.update(state)
        if blob is not None:
            self.user_evaluate = cloudpickle.loads(blob)
