#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 26 15:19:04 2025

@author: Joe

bayesopt.py

Implements a Bayesian Optimization class, `bayesopt` for fitting a set of model
parameters to data.

References:
  Wu et al. (2024)
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
try:
    from . import SSN_utils
except Exception:
    import SSN_utils
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
import scipy.stats as stats
import cloudpickle
import importlib

class bayesopt:
    """
    A minimal Bayesian Optimization class. It expects a user-supplied
    function that evaluates the cost given a parameter set.
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
            function. If None, assume to be a bounded uniform distrbution.
        evaluation_kwargs : dict, optional
            If the user-defined evaluation has keyword arguments, pass them 
            here as a dictionary
        ground_truth_data : np.ndarray, optional
            If needed for reference, but typically you'd pass this to your
            evaluation callback instead. 
        kwargs : dict
            Additional config for the GP model or acquisition function.
            Currently unused.
        """
        self.bounds = bounds
        self.prior = prior
        self.param_map = param_map
        self.ground_truth_data = ground_truth_data
        self.user_evaluate = user_evaluate
        self.evaluation_kwargs = evaluation_kwargs
        self.config = kwargs

        # Placeholder for GP or other surrogate model
        self.gp_cost = None
        self.gp_feas = None

    def bayesopt(self,
                 n_init: int = 20,
                 n_inst: int = 5,
                 n_iter: int = 20,
                 random_state: int = 0,
                 acquisition_func: Callable = SSN_utils.expected_improvement,
                 ground_truth_data: np.ndarray = None,
                 input_data: np.ndarray = None,
                 verbose: bool = False,
                 **kwargs):
        """
        Main loop for Bayesian Optimization.
        - Propose parameters using an acquisition function
        - Evaluate them via user_evaluate
        - Update GP with the new data
        - Return the best discovered parameters
        
        Parameters
        ----------
        n_init : int, optional
            Number of initial parameterization to sample
        n_inst : int, optional
            Number of random instatiations of each parameterization
        n_iter : int, optional
            Number of iterations
        ground_truth_data : np.ndarray, optional
            Data array to fit. Must have size N x Ncond where Ncond is the
            number of unique data samples to fit simultaneously. This may also
            be set when initializing the bayesopt object. Setting the data here
            will override any data set during initialization of the bayesopt 
            object.
        input_data : np.ndarray, optional
            Inputs to the model. Should have the same second dimension size as 
            ground_truth_data.
        verbose : bool, optional
            If True, print detailed output from parameter evaluations and iterations.
            If False (default), show a progress bar instead.
        kwargs : dict
            Advanced options for bayesopt method.
            
            fix_seed : bool
                If True, the random seed will be fixed for all iterations.
            fix_nodes : bool
                If True, the random seed for node instantiation will be fixed.
            node_seed : int
                Random seed for node instantiation. Only used if fix_nodes is True.
                If not provided, defaults to random_state.
        """
        if self.user_evaluate is None:
            raise ValueError("No user_evaluate function provided. "
                             "Set bayesopt.user_evaluate = your_callback.")
        if ground_truth_data is None:
            ground_truth_data = self.ground_truth_data

        # Store n_inst, n_iter, and n_init
        self.n_inst = n_inst
        self.n_iter = n_iter
        self.n_init = n_init
            
        # Assign acquitision func to object
        self.acquisition_func = acquisition_func
            
        # Get number of params
        n_params = len(self.bounds.keys())
            
        # Collect kwargs
        fix_seed = kwargs.get('fix_seed', False) #fix the random state on every iteration
        fix_nodes = kwargs.get('fix_nodes', False) #fix the random state for node instantiation
        node_seed = kwargs.get('node_seed', random_state) #random state for node instantiation
        use_feas = kwargs.get('use_feas', True) #use feasibility model
        evaluate_feasibility = kwargs.get('evaluate_feasibility', lambda x: x == 1) #Default: feasibility value must equal 1 to be accepted
        kernel_cost = kwargs.get('kernel_cost', Matern(nu=2.5, length_scale=np.ones((n_params,)), length_scale_bounds=(0.001, 100.0)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-8, 1e2))) #kernel for cost GP
        kernel_feas = kwargs.get('kernel_feas', Matern(nu=2.5, length_scale=np.ones((n_params,)),length_scale_bounds=(0.001, 100.0)) + WhiteKernel()) #kernel for feasibility GP
        n_candidates = kwargs.get('n_candidates', 1000) #number of parameterization to consider on each iteration
        debug = kwargs.get('debug', False) #debug mode
        track_acquisition = kwargs.get('track_acquisition', False) #this optionally tracks the acquisition function values for all candidate parameters. Ordinarily, you will not enable this as it will consume a lot of memory.
        track_gp_params = kwargs.get('track_gp_params', False) #track GP kernel hyperparameters on each iteration
        plot_params = kwargs.get('plot_params', None) #plots produced in debug mode
        kernel_fit_schedule = kwargs.get('kernel_fit_schedule', None)
        use_log_cost = kwargs.get('use_log_cost', False) #apply log transformation to cost values before GP fitting
        feas_acq_threshold = kwargs.get('feas_acq_threshold', None) #threshold below which feasibility scales acquisition; None means always multiply (default behavior)
        suppress_warnings = kwargs.get('suppress_warnings', False) #suppress sklearn GP convergence warnings
        
        # Import tqdm for progress bar (only when not verbose)
        if not verbose:
            try:
                from tqdm import tqdm as tqdm_bar
            except ImportError:
                tqdm_bar = None
        else:
            tqdm_bar = None
            
        # Set random state
        rng = np.random.default_rng(random_state)
        self.rng = rng
        
        # Set histories
        self.history_params = []
        self.history_costParams = []
        self.history_costs = []
        self.history_minCosts = []
        self.history_feas = []
    
        # Initialize Gaussian Process
        if verbose:
            print("Initializing...")
        initialization_iterator = tqdm_bar(range(n_init), desc="Initialization", disable=False) if tqdm_bar else range(n_init)

        for _ in initialization_iterator:
            
            # Choose random seed
            if not fix_seed:
                subseed = self.rng.integers(0, 9999999)
            else:
                subseed = random_state
            # Choose random node seed if fix_nodes is False
            if not fix_nodes:
                node_seed = self.rng.integers(0, 9999999)
            
            # Choose parameters from prior
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
                        params[p_i] = self.piror()
                    # Otherwise raise error
                    else:
                        raise TypeError(f"bayesopt: {type(self.prior[param_name])} is not a valid dtype for prior")
                
            # Evaluate Parameters
            # user_evaluate always returns (feasibility, cost)
            feas_val, cost_val = self.user_evaluate(params, 
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
            
            # If not using feasibility, treat all as feasible
            if not use_feas:
                feas_val = 1
                
            # Save Costs
            # If using feasibility
            self.params = self._get_params_dict(params)
            if use_feas:
                self.history_params.append(params)
                self.history_feas.append(feas_val)
                if evaluate_feasibility(feas_val):
                    self.history_costParams.append(params)
                    self.history_costs.append(cost_val)
            # If not using feasibility
            else:
                self.history_params.append(params)
                self.history_costParams.append(params)
                self.history_costs.append(cost_val)

            if tqdm_bar:
                initialization_iterator.set_postfix({"loss": f"{cost_val:.4f}"})
                
        # Test if any feasible solutions are found
        if len(self.history_costParams) == 0:
            raise ValueError("Error: No Feasible Initial Solutions Found. Ensure the parameter bounds are appropriately set and increase intial samples as needed.")
        elif len(self.history_costParams) < 5:
            print("Warning: Less than 5 initial feasible solutions found. Low cost samples can lead to poor initial estimates of the cost over parameter space. Consider increasing initial samples.")

        # Initialize Gaussian processes
        # gp_feas for feasibility (logistic-likelihood or simple regression).
        # gp_cost for cost regression.
        if use_feas:
            # For simplicity, we do normal GP on feasibility [0..1] ignoring that it's not truly normal.
            self.gp_feas = GaussianProcessRegressor(
                kernel=kernel_feas,
                alpha=0.0,
                normalize_y=False,  # we have [0,1], no need to normalize
                n_restarts_optimizer=3,
                random_state=random_state
            )
            #length_scale_feas = gp_feas.kernel_.length_scale

        self.gp_cost = GaussianProcessRegressor(
            kernel=kernel_cost,
            alpha=0.0,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=random_state
        )
        #length_scale_cost = gp_cost.kernel_.length_scale

        # Set up tracking variable if tracking acquisition
        if track_acquisition:
            self.history_acq = []
            self.history_candidates = []
            self.history_costMeans = []
            self.history_costStds = []
            if use_feas:
                self.history_acqfeas = []
                self.history_feasMeans = []
                self.history_feasStds = []

        # Set up tracking for GP kernel hyperparameters
        if track_gp_params:
            self.history_gp_cost_params = []
            if use_feas:
                self.history_gp_feas_params = []

        # Iterating
        if verbose:
            print("Iterating...")
        
        # Create progress bar with disable=False to override any global tqdm disabling
        # (we want the BO progress bar to always show when not verbose)
        iteration_iterator = tqdm_bar(range(n_iter), desc="Bayesian Optimization", disable=False) if tqdm_bar else range(n_iter)
        for iteration in iteration_iterator:
            
            # Choose random seed
            if not fix_seed:
                subseed = self.rng.integers(0, 9999999)
            else:
                subseed = random_state
            
            # Decide whether to fit kernel hyperparameters on this iteration
            if kernel_fit_schedule is None:
                fit_kernel = True
            else:
                # normalize schedule to set of ints if needed (expect 0-based iteration indices)
                try:
                    # turn into a set to get unique values if not already passed as a set
                    kernel_fit_set = set(int(i) for i in kernel_fit_schedule) if not isinstance(kernel_fit_schedule, set) else kernel_fit_schedule
                except Exception:
                    # if turning into a set fails, raise error
                    raise ValueError("kernel_fit_schedule must be a list, tuple, or set of integers indicating the iterations on which to fit kernel hyperparameters.")
                fit_kernel = (iteration in kernel_fit_set)

            # fit/update GP on existing data (optionally skip kernel hyperparameter optimization)
            if use_feas:
                self.gp_feas = self._update_surrogate(self.gp_feas, self.history_params, self.history_feas, debug=debug, plot_params=plot_params, fit_kernel=fit_kernel, suppress_warnings=suppress_warnings)
            self.gp_cost = self._update_surrogate(self.gp_cost, self.history_costParams, self.history_costs, debug=debug, plot_params=plot_params, use_log_cost=use_log_cost, fit_kernel=fit_kernel, suppress_warnings=suppress_warnings)

            # Track GP kernel hyperparameters
            if track_gp_params:
                self.history_gp_cost_params.append(self.gp_cost.kernel_.get_params(deep=True))
                if use_feas:
                    self.history_gp_feas_params.append(self.gp_feas.kernel_.get_params(deep=True))
            
            # store best cost
            best_cost = np.min(self.history_costs)
            self.history_minCosts.append(best_cost)
            
            # Propose next param set
            if track_acquisition:
                if use_feas:
                    params, candidates, acq, acq_feas, cost_mean, cost_std, feas_mean, feas_std = self._acquire_next_parameters(n_candidates, best_cost, use_feas, return_all=track_acquisition, feas_acq_threshold=feas_acq_threshold)
                    self.history_acqfeas.append(acq_feas)
                    self.history_feasMeans.append(feas_mean)
                    self.history_feasStds.append(feas_std)
                else:
                    params, candidates, acq, cost_mean, cost_std = self._acquire_next_parameters(n_candidates, best_cost, use_feas, return_all=track_acquisition, feas_acq_threshold=feas_acq_threshold)
                self.history_acq.append(acq)
                self.history_candidates.append(candidates)
                self.history_costMeans.append(cost_mean)
                self.history_costStds.append(cost_std)
            else:
                params = self._acquire_next_parameters(n_candidates, best_cost, use_feas, feas_acq_threshold=feas_acq_threshold)

            # Evaluate cost
            # user_evaluate always returns (feasibility, cost)
            feas_val, cost_val = self.user_evaluate(params, 
                                               self.param_map, 
                                               ground_truth_data,
                                               input_data,
                                               n_inst=n_inst,
                                               seed=subseed,
                                               fix_seed=fix_seed,
                                               **self.evaluation_kwargs
                                               )
            
            # If not using feasibility, treat all as feasible
            if not use_feas:
                feas_val = 1

            # Save Costs
            # If using feasibility
            self.params = self._get_params_dict(params)
            if use_feas:
                self.history_params.append(params)
                self.history_feas.append(feas_val)
                if evaluate_feasibility(feas_val):
                    self.history_costParams.append(params)
                    self.history_costs.append(cost_val)
            # If not using feasibility
            else:
                self.history_params.append(params)
                self.history_costParams.append(params)
                self.history_costs.append(cost_val)   
    
            if tqdm_bar:
                iteration_iterator.set_postfix({"loss": f"{cost_val:.4f}", "best": f"{best_cost:.4f}"})
                if verbose:
                    iteration_iterator.write(f"Iteration {iteration+1}/{n_iter}: new loss={cost_val:.4f}, best so far={best_cost:.4f}")
            elif verbose:
                print(f"Iteration {iteration+1}/{n_iter}: new loss={cost_val:.4f}, best so far={best_cost:.4f}")
    
        overall_best_idx = np.argmin(self.history_costs)
        self.params = self._get_params_dict(self.history_costParams[overall_best_idx])
        best_loss_value = self.history_costs[overall_best_idx]
    
        if verbose:
            print("\nFinal:")
            print(f"  best_params={self.params}")
            print(f"  best_loss={best_loss_value:.6f}")

        return self.params

    def _acquire_next_parameters(self, n_candidates, best_cost, use_feas, return_all=False, feas_acq_threshold=None):
        """
        Example: just picks random parameters from the bounds.
        You'd replace with a real acquisition function: e.g. EI on the GP.
        """
        
        # Get list of parameter names (assumes consistent ordering)
        param_names = list(self.bounds.keys())
        n_params = len(param_names)
        param_bounds = np.array([self.bounds[p] for p in param_names])
        
        # Sample all candidate values at once
        candidates = self.rng.uniform(param_bounds[:, 0], param_bounds[:, 1], size=(n_candidates, n_params))

        # Predict feasibility
        if use_feas:
            feas_mean, feas_std = self.gp_feas.predict(candidates, return_std=True)
            # clamp feasible predictions to [0,1]
            #feas_est = np.clip(feas_mean, 0, 1)
            Z_feas = (feas_mean - 0.5) / (feas_std + 1e-10) # standardize around 0.5 (the decision boundary) instead of 0
            feas_est = stats.norm.cdf(Z_feas)

        # Predict cost
        cost_mean, cost_std = self.gp_cost.predict(candidates, return_std=True)
        
        # compute acquisition
        acq = self.acquisition_func(cost_mean, cost_std, best_cost)

        # modify acquisition values if using feasibility
        if use_feas:
            if feas_acq_threshold is not None:
                # Only penalize candidates below the threshold; above it treat as fully feasible
                feas_weight = np.where(feas_est < feas_acq_threshold, feas_est, 1.0)
                acq_values = feas_weight * acq
            else:
                # Default: a(theta) = feas_est * acq
                acq_values = feas_est * acq
        else:
            acq_values = acq
        
        best_idx = np.argmax(acq_values)
        new_theta = candidates[best_idx]
        
        if return_all:
            if use_feas:
                return new_theta, candidates, acq, acq_values, cost_mean, cost_std, feas_mean, feas_std
            else:
                return new_theta, candidates, acq, cost_mean, cost_std
        else:
            return new_theta

    def _update_surrogate(self, gp: Callable, theta: list, cost: list, debug=False, plot_params = {'param_names' : []}, use_log_cost=False, fit_kernel: bool = True, suppress_warnings: bool = False):
        """
        Update the surrogate model:
            - Fit the GP to the observed data (theta, cost)

        Parameters
        ----------
        gp : GaussianProcessRegressor
            The Gaussian Process regressor to fit.
        theta : list
            List of parameter sets (each should be an array-like of shape (n_params,)).
        cost : list
            List of observed costs corresponding to each parameter set in theta.
        debug : bool, optional
            If True, will produce diagnostic plots of the GP fit for the specified parameters in plot_params.
        plot_params : dict, optional
            Dictionary specifying which parameters to plot in debug mode. Should contain:
                'param_names': list of parameter names to plot (must be keys in self.bounds)
        use_log_cost : bool, optional
            If True, applies a log transformation to the cost values before fitting the GP. This can help stabilize 
            variance when cost values span several orders of magnitude. Default is False.
        fit_kernel : bool, optional
            If True, the GP will optimize its kernel hyperparameters during fitting. If False, the GP will fit 
            without optimizing kernel hyperparameters (i.e., using the current kernel_ parameters). Default is True. 
            Note that if kernel hyperparameters have not been previously fitted, setting fit

        Returns
        -------
        GaussianProcessRegressor
            The updated GP regressor fitted to the provided data.
        """
        # Apply log transformation to cost if requested
        cost_array = np.array(cost)
        if use_log_cost:
            # Add a small epsilon to avoid log(0)
            cost_array = np.log(cost_array + 1e-10)
        
        # If we're not fitting kernel hyperparameters on this call, temporarily
        # disable the GP optimizer so `fit()` will not change kernel_.
        def _fit(gp, X, y):
            if suppress_warnings:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gp.fit(X, y)
            else:
                gp.fit(X, y)

        if not fit_kernel:
            # If kernel_ is not already present, optimization will be called anyway
            if not hasattr(gp, 'kernel_'):
                # Send a warning
                print("Warning: fit_kernel=False but GP has not been previously fitted. Fitting will proceed with kernel optimization.")
                _fit(gp, np.array(theta), cost_array)
            else:
                # Store the original optimizer, n_restarts_optimizer, and kernel to restore after fitting
                orig_optimizer = getattr(gp, 'optimizer', None)
                orig_restarts = getattr(gp, 'n_restarts_optimizer', None)
                orig_kernel = getattr(gp, 'kernel', None)
                try:
                    # Use the last fitted kernel (with tuned hyperparameters) as the starting kernel
                    gp.kernel = gp.kernel_
                    gp.optimizer = None
                    if orig_restarts is not None:
                        gp.n_restarts_optimizer = 0
                    _fit(gp, np.array(theta), cost_array)
                finally:
                    # restore the original optimizer settings and kernel
                    gp.optimizer = orig_optimizer
                    if orig_restarts is not None:
                        gp.n_restarts_optimizer = orig_restarts
                    if orig_kernel is not None:
                        gp.kernel = orig_kernel
        else:
            # Fit gp normally (optimize kernel hyperparameters)
            _fit(gp, np.array(theta), cost_array)
        
        if debug:
            bounds = np.array([self.bounds[key] for key in plot_params['param_names']])
            self.plot_gp_for_parameter(
                np.array(theta),
                np.array(cost),
                gp,
                bounds,
                param_names=plot_params['param_names']
            )
        
        return gp
    
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


    def plot_gp_heatmaps_2d_with_samples(
        self,
        param_names,
        gp = None,
        param_bounds = None,
        X_samples = None,
        y_samples = None,
        n_points=50,
        cax_name='Predicted Loss',
        samples_c='white',
        samples_cmap='gray',
        plot_min=False,
        true_min=None,
        vlim_mean=[None,None],
        vlim_std=[None,None]
        ):
        """
        Plot side-by-side heatmaps of the GP-predicted mean and std for a 2D (alpha,beta) model,
        and overlay the sampled points (X_samples).
    
        Parameters
        ----------
        gp : GaussianProcessRegressor
            The fitted GP model.
        param_bounds : list of (float, float)
            Bounds for the two parameters, e.g. [(alpha_min, alpha_max), (beta_min, beta_max)].
        X_samples : np.ndarray
            Shape [n_samples, 2] with [alpha, beta] for each sample.
        y_samples : np.ndarray
            Shape [n_samples,] with the observed losses for each sample (used only to color or label).
            If you just want to see the points, you can ignore y_samples or pass None and adapt.
        n_points : int, optional
            Number of points per dimension in the grid for plotting.
        """
        
        # Default to self attributes
        if gp is None:
            gp = self.gp_cost
        if param_bounds is None:
            param_bounds = self.bounds
        if X_samples is None:
            X_samples = np.array(self.history_costParams)
        if y_samples is None:
            y_samples = np.array(self.history_costs)
    
        # Ensure we have exactly 2D param space
        if len(param_bounds) != 2:
            raise ValueError("plot_gp_heatmaps_2d_with_samples only supports 2 parameters (2D).")
        if X_samples.shape[1] != 2:
            raise ValueError("X_samples must have shape [n_samples, 2].")
    
        (alpha_min, alpha_max) = param_bounds[param_names[0]]
        (beta_min, beta_max) = param_bounds[param_names[1]]
    
        # Build a meshgrid of alpha, beta
        alpha_space = np.linspace(alpha_min, alpha_max, n_points)
        beta_space  = np.linspace(beta_min, beta_max, n_points)
        A, B = np.meshgrid(alpha_space, beta_space)  # shape: (n_points, n_points)
    
        # Flatten the grid and predict
        test_points = np.column_stack((A.ravel(), B.ravel()))  # shape (n_points^2, 2)
        mu_pred, sigma_pred = gp.predict(test_points, return_std=True)
    
        # Reshape into 2D for plotting
        M = mu_pred.reshape(n_points, n_points)
        S = sigma_pred.reshape(n_points, n_points)
    
        # Create side-by-side subplots
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4), sharey=True)
    
        # Left: predicted mean
        mean_ax = axes[0]
        c0 = mean_ax.imshow(
            M, 
            origin='lower', 
            extent=[alpha_min, alpha_max, beta_min, beta_max],
            aspect='auto', 
            cmap='viridis',
            vmin=vlim_mean[0],
            vmax=vlim_mean[1]
        )
        mean_ax.set_title(f"GP Mean ({cax_name})")
        mean_ax.set_xlabel(param_names[0])
        mean_ax.set_ylabel(param_names[1])
        fig.colorbar(c0, ax=mean_ax, label=f"{cax_name} Mean")
    
        # Overlay the sampled points on mean subplot
        # alpha => X_samples[:,0], beta => X_samples[:,1]
        mean_ax.scatter(X_samples[:,0], X_samples[:,1],
                            c=samples_c, edgecolor='black', s=40,
                            label='Sampled points', cmap=samples_cmap)
        
        # Plot min
        if plot_min:
            theta_minI = np.argmin(y_samples)
            X_samples_min = X_samples[theta_minI]
            mean_ax.scatter(X_samples_min[0], X_samples_min[1], c='b', edgecolor="white", label="Optimum")
        if true_min:
            mean_ax.scatter(true_min[0], true_min[1], c='r', edgecolor="white", label="True Min")
    
        mean_ax.legend()    
        
        # Right: predicted std
        std_ax = axes[1]
        c1 = std_ax.imshow(
            S,
            origin='lower',
            extent=[alpha_min, alpha_max, beta_min, beta_max],
            aspect='auto',
            cmap='plasma',
            vmin=vlim_std[0],
            vmax=vlim_std[1]
        )
        std_ax.set_title("GP Std Dev")
        std_ax.set_xlabel(param_names[0])
        fig.colorbar(c1, ax=std_ax, label=f"{cax_name} Std")
    
        # Also overlay sampled points on std subplot (optional)
        std_ax.scatter(X_samples[:,0], X_samples[:,1],
                       c=samples_c, edgecolor='black', s=40, 
                       cmap=samples_cmap)
        
        # Plot min
        if plot_min:
            theta_minI = np.argmin(y_samples)
            X_samples_min = X_samples[theta_minI]
            std_ax.scatter(X_samples_min[0], X_samples_min[1], c='b', edgecolor="white")
        if true_min:
            std_ax.scatter(true_min[0], true_min[1], c='r', edgecolor="white")
    
        plt.tight_layout()
        plt.show()
        
        return fig, axes

    def plot_gp_slice_2d(self, param_keys, gp=None, resolution=50, bounds=None, plot_style='heatmap', vlim_mean = [None, None], vlim_std = [None, None]):
        """
        Plot a 2D slice of the mean and standard deviation of a Gaussian Process.
    
        Parameters:
            param_keys (tuple): Names of the two parameters to vary (e.g., ('x1', 'x2')).
            gp (object, optional): The GP model to use. Defaults to self.gp_cost.
            resolution (int): Number of points per axis for the grid.
            bounds (dict, optional): Optional bounds for the plot. If None, use GP or known parameter bounds.
        """
        if gp is None:
            gp = self.gp_cost
    
        if len(param_keys) != 2:
            raise ValueError("param_keys must contain exactly two parameters to vary.")
    
        p1, p2 = param_keys
        all_keys = list(self.params.keys())
        fixed_params = self.params.copy()
    
        if bounds is None:
            # Default to full parameter bounds if available
            bounds = self.bounds if hasattr(self, 'bounds') else {
                k: (0, 1) for k in all_keys  # fallback dummy bounds
            }
    
        p1_range = np.linspace(*bounds[p1], resolution)
        p2_range = np.linspace(*bounds[p2], resolution)
        P1, P2 = np.meshgrid(p1_range, p2_range)
    
        # Create input points for prediction
        X = []
        param_grid = []
        for i in range(resolution):
            for j in range(resolution):
                params = fixed_params.copy()
                params[p1] = P1[i, j]
                params[p2] = P2[i, j]
                X.append([params[k] for k in all_keys])
                param_grid.append((P1[i, j], P2[i, j]))
        X = np.array(X)
    
        mu, sigma = gp.predict(X, return_std=True)
    
        MU = mu.reshape(resolution, resolution)
        SIGMA = sigma.reshape(resolution, resolution)
    
        # Find the index of the minimum mean value
        min_index = np.argmin(mu)
        min_p1, min_p2 = param_grid[min_index]
    
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        if plot_style == 'contour':
            cs0 = axs[0].contourf(P1, P2, MU, cmap='viridis', vmin=vlim_mean[0], vmax = vlim_mean[1])
        elif plot_style == 'heatmap':
            cs0 = axs[0].imshow(MU, origin='lower', extent=(p1_range[0], p1_range[-1], p2_range[0], p2_range[-1]), aspect='auto', cmap='viridis', vmin=vlim_mean[0], vmax = vlim_mean[1])
        else:
            raise TypeError("bayesopt.plot_gp_slice_2D: Invalid 'plot_style'.")
        axs[0].scatter(min_p1, min_p2, color='blue', s=50, label='Minimum', edgecolor='white')
        axs[0].set_title('GP Mean')
        axs[0].set_xlabel(p1)
        axs[0].set_ylabel(p2)
        axs[0].legend()
        fig.colorbar(cs0, ax=axs[0])
        
        if plot_style == 'contour':
            cs1 = axs[1].contourf(P1, P2, SIGMA, cmap='plasma', vmin=vlim_std[0], vmax = vlim_std[1])
        elif plot_style == 'heatmap':
            cs1 = axs[1].imshow(SIGMA, origin='lower', extent=(p1_range[0], p1_range[-1], p2_range[0], p2_range[-1]), aspect='auto', cmap='plasma', vmin=vlim_std[0], vmax = vlim_std[1])
        else:
            raise TypeError("bayesopt.plot_gp_slice_2D: Invalid 'plot_style'.")
        axs[1].set_title('GP Std. Dev.')
        axs[1].set_xlabel(p1)
        axs[1].set_ylabel(p2)
        fig.colorbar(cs1, ax=axs[1])
    
        plt.tight_layout()
        #plt.show()
        
        return fig, axs
        
        
    def plot_gp_for_parameter(
        self,
        X_samples, 
        y_samples, 
        gp, 
        param_bounds,
        fixed_params=None, 
        n_points=200,
        param_names=None
        ):
        """
        Plot one or more 1D slices of the Gaussian Process for selected parameter dimensions.
        
        Parameters
        ----------
        X_samples : np.ndarray
            Shape [n_samples, n_params]. The parameter sets that have been sampled.
        y_samples : np.ndarray
            Shape [n_samples,]. The associated losses for each sampled parameter set.
        gp : GaussianProcessRegressor
            The fitted GP model.
        param_bounds : list of (float, float)
            The lower and upper bounds for each free parameter, as used in the optimization.
        param_index : int or list/tuple of int, optional
            The index (or indices) of the parameter dimension(s) to visualize.
            If multiple, each dimension gets its own subplot.
        fixed_params : np.ndarray or None, optional
            If None, we default to the best parameters from the optimization. Otherwise,
            you can pass in an array of shape [n_params] specifying how to fix
            the other parameter dimensions.
        n_points : int, optional
            Number of points in the grid for plotting.
        """
        
        X_samples = np.array(X_samples)
        y_samples = np.array(y_samples)
        n_samples, n_params = X_samples.shape
        
        param_index = np.arange(0,len(param_names))
    
        # Validate each dimension
        for dim in param_index:
            if not (0 <= dim < n_params):
                raise ValueError(f"Invalid param_index {dim}, must be in [0, {n_params - 1}]")
    
        # If no fixed_params specified, pick the best from the samples
        if fixed_params is None:
            best_idx = np.argmin(y_samples)
            fixed_params = X_samples[best_idx].copy()
        else:
            fixed_params = np.array(fixed_params).copy()
            if fixed_params.shape[0] != n_params:
                raise ValueError(f"fixed_params must have length {n_params}")
                
        # Create param names if none created
        if not param_names:
            param_names = param_index
    
        # Create subplots: one per requested dimension
        fig, axes = plt.subplots(
            1, len(param_index), 
            figsize=(5 * len(param_index), 4), 
            squeeze=False
        )
        axes = axes.flatten()  # so we can index easily
        
        for ax_i, dim in enumerate(param_index):
            # Build a grid for the chosen dimension
            lb, ub = param_bounds[dim]
            X_grid_1d = np.linspace(lb, ub, n_points)
    
            # Build test array
            X_test = []
            for val in X_grid_1d:
                test_params = fixed_params.copy()
                test_params[dim] = val
                X_test.append(test_params)
            X_test = np.array(X_test)
    
            # Predict with the GP
            mu_pred, sigma_pred = gp.predict(X_test, return_std=True)
    
            ax = axes[ax_i]
            ax.plot(X_grid_1d, mu_pred, label='GP mean prediction')
            ax.fill_between(
                X_grid_1d,
                mu_pred - 2.0 * sigma_pred,
                mu_pred + 2.0 * sigma_pred,
                alpha=0.2,
                label='GP ± 2 std'
            )
            # Overlay the actual sampled points along this dimension
            ax.scatter(X_samples[:, dim], y_samples,
                       marker='x', color='black', label='Sampled points')
    
            ax.set_xlabel(f'{param_names[dim]}')
            ax.set_ylabel('Loss')
            
            # Show which slice we are looking at
            # i.e. the fixed values of other parameters
            # (Skip the parameter currently being plotted)
            other_dims = [d for d in range(n_params) if d != dim]
            slice_info = [f"{param_names[dim]}={fixed_params[d]:.3g}" for d in other_dims]
            slice_text = ", ".join(slice_info)
    
            ax.set_xlabel(f'{param_names[dim]}')
            ax.set_ylabel('Loss')
            # We'll embed that slice info in the subplot title
            ax.set_title(f'1D Slice (Param {dim})\nFixed: {slice_text}', fontsize=10)
            ax.legend()
    
        plt.tight_layout()
        plt.show()

    def __getstate__(self):
        state = self.__dict__.copy()
        user_eval = state.get("user_evaluate")
        if user_eval is not None:
            state["_user_evaluate_blob"] = cloudpickle.dumps(user_eval)
            state["user_evaluate"] = None
        return state

    def __setstate__(self, state):
        blob = state.pop("_user_evaluate_blob", None)
        self.__dict__.update(state)
        if blob is not None:
            self.user_evaluate = cloudpickle.loads(blob)
        
