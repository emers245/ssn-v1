#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 16:59:19 2022

@author: joe

Define various utilities for the SSN.py class.
"""

import numpy as np
import os
import h5py
from scipy.special import i0, i1
from scipy.optimize import brentq
import scipy.stats as stats
from scipy.linalg import solve_continuous_lyapunov
from scipy.special import rel_entr
from sklearn.neighbors import NearestNeighbors


#%% File handling

# Find files in multiple directories
def find_file(target_file, directories):
    found_paths = []
    for directory in directories:
        for root, dirs, files in os.walk(directory):
            if target_file in files:
                found_paths.append(os.path.join(root, target_file))
    return found_paths

# Resolve paths in config files
def resolve_path(path, manifest):
    # Continue to resolve path until no $-variables are left
    while any(key in path for key in manifest):
        for key, value in manifest.items():
            path = path.replace(key, value)
    return path

# Scan an HDF5 file to reveal structure
def scan_hdf5(path, recursive=True, tab_step=2):
    def scan_node(g, tabs=0):
        print(' ' * tabs, g.name)
        for k, v in g.items():
            if isinstance(v, h5py.Dataset):
                print(' ' * tabs + ' ' * tab_step + ' -', v.name)
            elif isinstance(v, h5py.Group) and recursive:
                scan_node(v, tabs=tabs + tab_step)
    with h5py.File(path, 'r') as f:
        scan_node(f)

# Make a dictionary json serilizable by converting numpy types to native python types
def make_json_serializable(d):
    if isinstance(d, dict):
        return {k: make_json_serializable(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [make_json_serializable(v) for v in d]
    elif isinstance(d, np.integer):
        return int(d)
    elif isinstance(d, np.floating):
        return float(d)
    elif isinstance(d, np.ndarray):
        return d.tolist()
    else:
        return d

#%% Common functions

def circD(theta1,theta2,mod):
  """
   min distance on a circle between two points
   Inputs:
     theta1, theta2 (numpy arrays): angles
     mod (scalar): circumference of circle
   Outputs
     Dtheta (numpy array): min difference in angles on a circle
  """

  Dtheta = np.minimum(np.mod(theta1-theta2,mod),np.mod(theta2-theta1,mod))

  return(Dtheta)

def circGauss(x,mu,sigma,cyc=2*np.pi,normalize=False):
  """
   circular Gaussian function
   Inputs:
     mu (scalar): mean
     sigma (scalar): standard deviation
     cyc (scalar): cycle size (rads)
     x (numpy array): independent variable (circular)
     normalize (bool): if True, normalize so that approximate integral is 1 (in the limit sigma goes to zero)
   Outputs:
     vm (numpy array): circular Gaussian as a function of x
  """

  if normalize:
    norm = 1/(np.sqrt(2*np.pi)*sigma)
  else:
    norm = 1.0

  cg = norm * np.exp(-(circD(x,mu,cyc)**2)/(2*sigma**2))

  return(cg)

def Gaussian(x,mu,sigma,normalize=False):
  """
   Gaussian function
   Inputs:
     mu (numpy array): mean (Ngx1)
     sigma (numpy array): standard deviation (Ngx1 or alternatively scalar for all same sigma)
     x (numpy array): independent variable (NgxN)
     normalize (bool): if True, normalize so that integral is 1
   Outputs:
     vm (numpy array): Gaussian as a function of x
  """

  if normalize:
    norm = 1/(np.sqrt(2*np.pi)*sigma)
  else:
    norm = 1.0

  cg = norm * np.exp(-((x-mu)**2)/(2*sigma**2))

  return(cg)

def DoG(x, m1, m2, s1, s2, A1, A2, C, x_shift=0):
    """
    Difference of Gaussians (DoG) function

    Parameters:
    x (float or array-like): The input variable, representing spatial or temporal points.
    m1 (float): Mean (center) of the first Gaussian.
    m2 (float): Mean (center) of the second Gaussian.
    s1 (float): Standard deviation (width) of the first Gaussian.
    s2 (float): Standard deviation (width) of the second Gaussian.
    A1 (float): Amplitude of the first Gaussian.
    A2 (float): Amplitude of the second Gaussian.
    C (float): Constant offset added to the result.
    x_shift (float, optional): A shift applied to both Gaussians along the x-axis. Default is 0.

    Returns:
    float or array-like: The value of the Difference of Gaussians at the given x values.

    Description:
    The Difference of Gaussians function is often used in image processing and
    computational neuroscience to approximate the receptive field profiles of neurons.
    It is computed by subtracting one Gaussian from another, resulting in a function 
    with a band-pass characteristic, highlighting spatial or temporal differences.
    """
    return A1*Gaussian(x - x_shift, m1, s1) - A2*Gaussian(x - x_shift, m2, s2) + C

def von_mises(x, mu, sigma=None, kappa=None, cyc=2*np.pi, normalize=False):
    """
    Computes the von Mises function for a given vector x and given parameters mu and kappa (or sigma).

    Inputs:
        x (array-like): Input vector of angles (in radians).
        mu (float): Mean direction of the distribution.
        kappa (float, optional): Concentration parameter.
        sigma (float, optional): Approximate circular standard deviation (rad); kappa = 1 / sigma².
        cyc (float): Normalization cycle length (default 2pi).
        normalize (bool): If True, normalize the output so the integral is 1 over one cycle.

    Outputs:
        array-like: Values of the von Mises function at each angle in x.
    """
    if (kappa is None) == (sigma is None):
        raise ValueError("Specify exactly one of kappa or sigma.")
    if sigma is not None:
      if sigma <= 0:
        raise ValueError("sigma must be positive.")
      # Interpret `sigma` in the same units as `x` and `cyc`.
      # Convert sigma -> radians before using the circular std relation.
      sigma_rad = sigma * ((2.0 * np.pi) / cyc)
      # Convert circular standard deviation (sigma_rad) to concentration kappa.
      # Exact relation uses R = I1(kappa)/I0(kappa) and sigma_rad = sqrt(-2 ln R).
      # Compute R from sigma_rad and invert using a robust approximation R -> kappa.
      R_target = np.exp(-(sigma_rad ** 2) / 2.0)
      # Handle extreme values
      if R_target <= 0.0:
        kappa = 0.0
      elif R_target < 1e-12:
        kappa = 0.0
      else:
        R = R_target
        # Use the standard piecewise approximation (best performance & stability)
        if R < 0.53:
          kappa = 2*R + R**3 + (5.0/6.0)*R**5
        elif R < 0.85:
          kappa = -0.4 + 1.39*R + 0.43/(1.0 - R)
        else:
          denom = (R**3 - 4*R**2 + 3*R)
          if denom == 0:
            kappa = 1e6
          else:
            kappa = 1.0 / denom
        # ensure non-negative
        if kappa < 0:
          kappa = 0.0

    if kappa is None:
      raise ValueError("Specify exactly one of kappa or sigma.")
    if kappa < 0:
      raise ValueError("kappa must be non-negative.")

    vm = np.exp(kappa * np.cos(((2*np.pi)/cyc)*(x - mu)))

    if normalize:
        norm = cyc * i0(kappa)
    else:
        norm = np.max(vm) if np.max(vm) != 0 else 1.0

    return vm / (norm)

#%% Input functions

def lineInput(l,sigmaRF,x):
  """
   Shape of line inputs
   Inputs:
     l (numpy array): length of input (degrees visual angle)
     sigmaRF (scalar): st. dev. of Gaussian RF (degrees visual angle)
     x (numpy array): independent variable (degrees visual angle)
   Outputs:
     sl: a function of the line stimulus input
  """

  sl = (1/(1+np.exp(-(x+l/2)/sigmaRF)))*(1-1/(1+np.exp(-(x-l/2)/sigmaRF)))

  return(sl)

def Input2D(d,sigmaRF,xy):
  """
   Shape of 2D circular grating input with hard edge
   Inputs:
     d (numpy array): diameter of input (Nrx1) (degrees visual angle)
     sigmaRF (scalar): st. dev. of Gaussian RF (degrees visual angle)
     xy (numpy array): independent variable (2xNyxNx) (degrees visual angle)
   Outputs:
     2Dstim: a function of the 2D stimulus
  
   Note, the stimulus will be centered on the origin of whatever xy 
   coordinates are given to the function.
  """
  
  dist = np.sqrt(xy[0,:,:]**2+xy[1,:,:]**2)
  stim2D = (1/(1+np.exp(-(dist+d/2)/sigmaRF)))*(1-1/(1+np.exp(-(dist-d/2)/sigmaRF)))
  
  return(stim2D)

#%% Generate orientation columns

def makeOriMap(kc,n,RFxy,seed=None):
  """
   Use equation 20 from Kaschube et al. (2010) supplementary materials to 
   generate an orientation tuning map on a 2D surface.
   Inputs:
     kc (float) - radians per unit length
     n (int) - number of orientation preferences
     RFxy (numpy array) - positions of 2D grid along x and y dimensions (2 x Nx x Ny)
   Outputs:
     OriMap (numpy array) - a map of orientation preferences on a 2D surface (Nx x Ny)
  """
  rng = np.random.default_rng(seed)
  
  orientations = np.linspace(0, np.pi, n, endpoint=False)  # j*pi/n
  phase = 2*np.pi * rng.random(n)
  l = rng.choice([-1, 1], n)
  k = kc*np.array([np.cos(orientations),np.sin(orientations)])
  xy = np.array([np.ndarray.flatten(RFxy[0,:]),np.ndarray.flatten(RFxy[1,:])])
  planewave = np.exp(1j*(l[:,None]*(k.T @ xy)+phase[:,None]))
  # planewave = 1j*np.ones([len(orientations),np.shape(xy)[1]])
  # for oi in range(len(orientations)):
  #     planewave[oi,:] = np.exp(1j*(l[oi]*np.dot(k[:,oi],xy)+phase[oi]))
  planform = np.reshape(np.sum(planewave,axis=0),[np.shape(RFxy)[1],np.shape(RFxy)[2]]) #the planform give be Eq. 20
  OriMap = (np.angle(planform)+np.pi)/2 #the orientation preference function wrt space is the phase of the planform
  
          
  return(OriMap)

#%% Cost functions

def rectified_kl_divergence(data1, data2, method='fd', eps=1e-12, k=5):
    """ Compute a rectified KL divergence D_KL(P || Q) between samples data1 ~ P and data2 ~ Q. 
    This should be used when the distributions sampled are rectified (e.g., firing rates). This 
    loss is based on the loss function used by Palmigiano et al. (2022). Rectification is assumed 
    to happen at zero, with only positive values surviving. 
    
    Supported methods ----------------- 
    - Histogram-based (original behavior): method in {'fd','doane','auto',...} or int 
    - Wang-Kulkarni-Verdú kNN estimator: method == 'wkv_knn' 
    
    Parameters ---------- 
    data1 : np.ndarray Samples for dataset 1 (e.g. ground truth), shape (n,) or (n, d). 
    data2 : np.ndarray Samples for dataset 2 (e.g. model), shape (m,) or (m, d). 
    method : str or int Histogram bin rule (np.histogram_bin_edges) OR 'wkv_knn'. 
    eps : float Small constant to avoid log(0) / handle degeneracies. 
    k : int Number of neighbors for kNN KL estimator (used when method='wkv_knn') 
    """

    data1 = np.asarray(data1).ravel()
    data2 = np.asarray(data2).ravel()

    n1 = len(data1)
    n2 = len(data2)
    if n1 == 0 or n2 == 0:
        raise ValueError("data1 and data2 must be nonempty.")

    # Empirical mass at zero
    p0 = (np.sum(data1 == 0) + eps) / (n1 + eps)
    q0 = (np.sum(data2 == 0) + eps) / (n2 + eps)

    # Bernoulli KL for event {r==0} vs {r>0}
    kl_bern = float(rel_entr(p0, q0) + rel_entr(1.0 - p0, 1.0 - q0))

    # Positive parts (conditional)
    x_pos = data1[data1 > 0]
    y_pos = data2[data2 > 0]

    # If P has no positive mass, then P is (almost) all-zero ? KL reduces to Bernoulli term.
    if x_pos.size == 0:
        return kl_bern

    # If Q has no positive samples but P does, true KL is infinite.
    # With eps smoothing you could return a large number, but it's important to be explicit.
    if y_pos.size == 0:
        return np.inf

    kl_pos = kl_divergence(x_pos, y_pos, method=method, eps=eps, k=k)

    # Weight by P's nonzero mass
    kl_total = kl_bern + (1.0 - p0) * float(kl_pos)
    return float(kl_total)

def kl_divergence(data1, data2, method='fd', eps=1e-12, k=5):
    """
    Compute KL divergence D_KL(P || Q) between samples data1 ~ P and data2 ~ Q.

    Supported methods
    -----------------
    - Histogram-based (original behavior): method in {'fd','doane','auto',...} or int
    - Wang?Kulkarni?Verdú kNN estimator: method == 'wkv_knn'

    Parameters
    ----------
    data1 : np.ndarray
        Samples for dataset 1 (e.g. ground truth), shape (n,) or (n, d).
    data2 : np.ndarray
        Samples for dataset 2 (e.g. model), shape (m,) or (m, d).
    method : str or int
        Histogram bin rule (np.histogram_bin_edges) OR 'wkv_knn'.
    eps : float
        Small constant to avoid log(0) / handle degeneracies.
    k : int
        Number of neighbors for kNN KL estimator (used when method='wkv_knn').

    Returns
    -------
    kl : float
        Estimated KL divergence D_KL(P || Q).

    Notes
    -----
    WKV kNN estimator (Wang, Kulkarni, Verdú, 2009) for KL:
        KL(P||Q) ? (d/n) * sum_i log(nu_i / rho_i) + log(m / (n-1))
    where:
        rho_i = distance from x_i to its k-th nearest neighbor among {x_j}_{j!=i} (P-samples)
        nu_i  = distance from x_i to its k-th nearest neighbor among {y_j} (Q-samples)
        d     = dimension
        n     = len(data1), m = len(data2)

    Requirements
    ------------
    For method='wkv_knn', requires scikit-learn.
    """

    # -------------------------
    # Wang-Kulkarni-Verdú kNN KL divergence
    # -------------------------
    if method == 'wkv_knn':
        x = np.asarray(data1)
        y = np.asarray(data2)

        # Ensure 2D arrays: (n, d)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n, d = x.shape
        m = y.shape[0]

        if n < 2:
            raise ValueError("WKV kNN KL requires at least 2 samples in data1.")
        if m < 1:
            raise ValueError("WKV kNN KL requires at least 1 sample in data2.")
        if k < 1:
            raise ValueError("k must be >= 1.")
        if k >= n:
            raise ValueError(f"k must be < len(data1) (got k={k}, len(data1)={n}).")
        if k > m:
            raise ValueError(f"k must be <= len(data2) (got k={k}, len(data2)={m}).")

        # Small jitter to break ties / avoid zero distances (important for rectified rates w/ repeats)
        # Scale jitter to data magnitude to be minimally invasive.
        scale = max(np.std(x), np.std(y), 1.0)
        rng = np.random.default_rng(0)
        xj = x + rng.normal(0.0, eps * scale, size=x.shape)
        yj = y + rng.normal(0.0, eps * scale, size=y.shape)

        # rho_i: k-th NN distance within X excluding self -> query k+1 and take the (k+1)th (0 is self)
        nn_x = NearestNeighbors(metric='minkowski', p=2, algorithm='auto')
        nn_x.fit(xj)
        dist_x, _ = nn_x.kneighbors(xj, n_neighbors=k + 1, return_distance=True)
        rho = dist_x[:, -1]  # k-th neighbor excluding self

        # nu_i: k-th NN distance from X to Y
        nn_y = NearestNeighbors(metric='minkowski', p=2, algorithm='auto')
        nn_y.fit(yj)
        dist_y, _ = nn_y.kneighbors(xj, n_neighbors=k, return_distance=True)
        nu = dist_y[:, -1]

        # Guard against zeros (can happen if eps=0 and duplicate points exist)
        rho = np.maximum(rho, eps)
        nu = np.maximum(nu, eps)

        kl = (d / n) * np.sum(np.log(nu / rho)) + np.log(m / (n - 1))
        # Assert kl is non-negative (WKV estimator can be slightly negative due to finite sample bias; clip to zero)
        kl = max(kl, 0.0)
        return float(kl)

    # -------------------------
    # Histogram-based KL
    # -------------------------
    data1 = np.asarray(data1).ravel()
    data2 = np.asarray(data2).ravel()

    try:
        edges1 = np.histogram_bin_edges(data1, bins=method)
        edges2 = np.histogram_bin_edges(data2, bins=method)
        if edges1.size > 10000 or edges2.size > 10000:
            raise ValueError("Too many bins")
    except Exception:
        print(f"WARNING: {method} Method Failed for kl_divergence. Using simpler binning method.")
        edges1 = np.histogram_bin_edges(data1, bins=100)
        edges2 = np.histogram_bin_edges(data2, bins=100)

    unified_edges = np.union1d(edges1, edges2)
    unified_edges = np.sort(unified_edges)

    hist1, _ = np.histogram(data1, bins=unified_edges, density=False)
    hist2, _ = np.histogram(data2, bins=unified_edges, density=False)

    p = hist1.astype(float) / (hist1.sum() + eps)
    q = hist2.astype(float) / (hist2.sum() + eps)

    p = np.where(p == 0, eps, p)
    q = np.where(q == 0, eps, q)

    kl = np.sum(rel_entr(p, q))
    return float(kl)

def compute_MSE(data1, data2, method=None):
    """ Computes the mean-squrared error between two datasets. Datsets must be equal in size. """
    
    #Check data size
    if np.shape(data1) != np.shape(data2):
        raise("ERROR: Dataset must be equal size for MSE")
    else:
        N = np.size(data1)
        return (1/N)*(np.sum((np.array(data1) - np.array(data2))**2))

def correlation_eigenspectrum(data, method='pearson', rowvar=False, sort_desc=True, eps=1e-12, max_components=None):
    """
    Compute the eigenspectrum of the pairwise correlation matrix.

    Parameters
    ----------
    data : np.ndarray
      Data matrix. By default (rowvar=False), expected shape is
      (n_samples, n_cells), e.g. stimuli x cells.
    method : str
      Correlation method. Currently supports 'pearson'.
      Extra values (e.g. 'fd') are accepted for compatibility and treated as
      'pearson'.
    rowvar : bool
      Passed to np.corrcoef. If False, each column is a variable (cell).
    sort_desc : bool
      If True, return eigenvalues in descending order.
    eps : float
      Small threshold to remove near-zero variance variables.
    max_components : int or None
      If provided, return only the first max_components eigenvalues (after sorting).
      Useful for comparing top-k components and avoiding size mismatches.

    Returns
    -------
    eigvals : np.ndarray
      Eigenvalues of the correlation matrix.
    """
    x = np.asarray(data)
    if x.ndim != 2:
      raise ValueError(f"correlation_eigenspectrum expects a 2-D array, got shape {x.shape}")

    # Keep compatibility with the generic 'method' argument used elsewhere.
    if method not in [None, 'pearson', 'fd', 'doane', 'auto']:
      raise ValueError(f"Unsupported method='{method}' for correlation_eigenspectrum. Use 'pearson'.")

    if rowvar:
      var_std = np.std(x, axis=1)
      keep = var_std > eps
      x_use = x[keep, :]
    else:
      var_std = np.std(x, axis=0)
      keep = var_std > eps
      x_use = x[:, keep]

    if x_use.size == 0:
      return np.array([0.0])

    corr = np.corrcoef(x_use, rowvar=rowvar)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 1.0)

    eigvals = np.linalg.eigvalsh(corr)
    if sort_desc:
      eigvals = eigvals[::-1]

    if max_components is not None and max_components > 0:
      eigvals = eigvals[:max_components]

    return eigvals

def eigenspectrum_mse(model, data=None, method='pearson', gt_eigvals=None, rowvar=False, eps=1e-12, max_components=None):
    """
    Compute MSE between model and target correlation eigenspectra.

    Parameters
    ----------
    model : np.ndarray
      Model outputs (typically stimuli x cells).
    data : np.ndarray or None
      Ground-truth outputs. Required when gt_eigvals is None.
    method : str
      Included for compatibility with other cost functions and compute_cost.
      Passed through to correlation_eigenspectrum.
    gt_eigvals : np.ndarray or None
      Optional precomputed ground-truth eigenspectrum. If provided, `data`
      is not needed and ground-truth computation is skipped.
    rowvar : bool
      Passed to correlation_eigenspectrum.
    eps : float
      Small threshold used in correlation_eigenspectrum.
    max_components : int or None
      If provided, compare only the first max_components eigenvalues.
      Helps avoid size mismatches and focuses on dominant components.

    Returns
    -------
    float
      Mean squared error between ordered eigenspectra.
    """
    model_eigvals = correlation_eigenspectrum(
      model,
      method=method,
      rowvar=rowvar,
      sort_desc=True,
      eps=eps,
      max_components=max_components,
    )

    if gt_eigvals is None:
      if data is None:
        raise ValueError("Provide either data or precomputed gt_eigvals.")
      gt_eigvals = correlation_eigenspectrum(
        data,
        method=method,
        rowvar=rowvar,
        sort_desc=True,
        eps=eps,
        max_components=max_components,
      )
    elif max_components is not None and max_components > 0:
      # Truncate precomputed gt_eigvals if max_components is specified
      gt_eigvals = gt_eigvals[:max_components]

    gt_eigvals = np.asarray(gt_eigvals)
    
    # Handle shape mismatches by taking the minimum length
    # This occurs when subtypes have different numbers of cells
    min_len = min(model_eigvals.shape[0], gt_eigvals.shape[0])
    if min_len == 0:
      raise ValueError(
        f"Eigenspectrum has zero length: model {model_eigvals.shape} vs gt {gt_eigvals.shape}. "
        "Check that data has sufficient non-zero-variance variables."
      )
    
    # Truncate both to the minimum length for fair comparison
    model_eigvals = model_eigvals[:min_len]
    gt_eigvals = gt_eigvals[:min_len]

    return compute_MSE(model_eigvals, gt_eigvals)

def compute_linearized_covariance(ssn, r_star, g, model_input=None, eigval_check=True, W=None):
    """
    Compute steady-state covariance of the linearized SSN around a fixed point.

    Parameters
    ----------
    ssn : SSN
        SSN model object.
    r_star : array-like, shape (N,)
        Fixed-point firing-rate vector.
    g : array-like, shape (N, N)
        Noise matrix L such that Q = L @ L.T.
    model_input : array-like, shape (N,), optional
        Input vector h. If None, defaults to ssn.h[:, -1].
    eigval_check : bool
        If True, print a warning if any Jacobian eigenvalues have positive real parts.
    W : ndarray, shape (N, N), optional
        Pre-computed adjacency matrix. If None, calls ssn.construct_W().
        Pass this when calling in a loop over stimuli to avoid redundant construction.

    Returns
    -------
    C : ndarray, shape (N, N)
        Covariance matrix from Lyapunov equation.
    """

    # Ensure vector/matrix shapes
    r_star = np.asarray(r_star).reshape(-1)
    L = np.asarray(g)

    # Default input to final time point
    if model_input is None:
        h = np.asarray(ssn.h[:, -1]).reshape(-1)
    else:
        h = np.asarray(model_input).reshape(-1)

    # 1) Construct adjacency matrix (or use pre-computed)
    if W is None:
        W = ssn.construct_W()
    n_cells = len(ssn.nodes)

    # Basic validation
    if r_star.shape[0] != n_cells:
        raise ValueError(f"r_star must have length {n_cells}, got {r_star.shape[0]}")
    if h.shape[0] != n_cells:
        raise ValueError(f"model_input must have length {n_cells}, got {h.shape[0]}")
    if L.shape != (n_cells, n_cells):
        raise ValueError(f"g must have shape ({n_cells}, {n_cells}), got {L.shape}")

    # 2) Gather per-cell parameters by model type (k, n, c, tau)
    model_type_masks = {}
    for model_name in ssn.node_types["model_type_name"]:
        model_type_masks[model_name] = np.array(ssn.nodes["model_name"] == model_name)

    c = np.zeros(n_cells)
    k = np.zeros(n_cells)
    n = np.zeros(n_cells)
    tau = np.zeros(n_cells)

    for model_name in ssn.node_types["model_type_name"]:
        row = ssn.node_types.loc[ssn.node_types["model_type_name"] == model_name]
        c[model_type_masks[model_name]] = row["c"].values[0]
        k[model_type_masks[model_name]] = row["k"].values[0]
        n[model_type_masks[model_name]] = row["n"].values[0]
        tau[model_type_masks[model_name]] = row["tau"].values[0]

    # 3) Compute gain D
    i_arg = W @ r_star + c * h
    D = k * n * np.maximum(i_arg, 0) ** (n - 1)

    # 4) Compute Jacobian J using broadcasting (avoids allocating dense diagonal matrices)
    J = (1.0 / tau)[:, None] * (D[:, None] * W - np.eye(n_cells))

    # 5) Eigenvalue check (only compute eigenvalues when requested)
    if eigval_check:
        eigvals = np.linalg.eigvals(J)
        unstable = eigvals.real > 0
        if np.any(unstable):
            print(
                f"Warning: {np.sum(unstable)} Jacobian eigenvalue(s) have positive real part. "
                f"Max Re(lambda)={np.max(eigvals.real):.4e}"
            )

    # 6) Solve Lyapunov equation: J C + C J^T + Q = 0
    Q = L @ L.T
    C = solve_continuous_lyapunov(J, -Q)
    return C


def noise_eigenspectrum_mse(ssn, r_star, g, gt_eigvals, W=None, max_components=None, model_input=None):
    """Compute MSE between top eigenvalues of linearized noise covariance and ground truth.

    Parameters
    ----------
    ssn : SSN
        SSN model object (needs node parameters and edges).
    r_star : array-like, shape (N,)
        Fixed-point firing-rate vector.
    g : array-like, shape (N, N)
        Noise matrix L such that Q = L @ L.T.
    gt_eigvals : array-like
        Ground-truth noise covariance eigenvalues (descending order).
    W : ndarray, shape (N, N), optional
        Pre-computed adjacency matrix.  Pass this to avoid redundant construction.
    max_components : int or None
        Number of top eigenvalues to compare.  If None, compare as many as available.
    model_input : array-like, shape (N,), optional
        External input vector h.  If None, defaults to ssn.h[:, -1].

    Returns
    -------
    float
        Mean squared error between the top-k eigenvalues of the model noise
        covariance and *gt_eigvals*.
    """
    C = compute_linearized_covariance(ssn, r_star, g, model_input=model_input,
                                      eigval_check=False, W=W)
    eigvals_model = np.linalg.eigvalsh(C)[::-1]  # descending real eigenvalues (C is symmetric)
    gt_eigvals = np.sort(np.asarray(gt_eigvals).ravel())[::-1]

    # Determine how many components to compare
    n_avail = min(len(eigvals_model), len(gt_eigvals))
    if max_components is not None and max_components > 0:
        k = min(max_components, n_avail)
    else:
        k = n_avail

    return compute_MSE(eigvals_model[:k], gt_eigvals[:k])


def eigenspectrum_cov_mse(eigvals_model, eigvals_gt, method=None, max_components=None):
    """
    Compute MSE between model and target covariance eigenspectra.

    Parameters
    ----------
    eigvals_model : array-like
      Model covariance eigenspectrum.
    eigvals_gt : array-like
      Ground-truth covariance eigenspectrum.
    method : str
      Included for compatibility with other cost functions and compute_cost.
      Passed through to correlation_eigenspectrum.
    max_components : int or None
      If provided, compare only the first max_components eigenvalues.

    Returns
    -------
    float
      Mean squared error between ordered eigenspectra.
    """

    # Sort in descending order
    eigvals_model = np.sort(eigvals_model)[::-1]
    eigvals_gt = np.sort(eigvals_gt)[::-1]

    # Set max components
    if max_components is not None and max_components > 0:
       if max_components > len(eigvals_model) or max_components > len(eigvals_gt):
           print(
               f"Warning: max_components={max_components} exceeds number of available eigenvalues "
               f"(model has {len(eigvals_model)}, gt has {len(eigvals_gt)}). "
               "Using smaller value."
           )
           max_components = np.min([len(eigvals_model), len(eigvals_gt)])
    elif max_components is not None and max_components <= 0:
        print(f"Warning: max_components={max_components} is not positive. Ignoring and using all components.")
        max_components = np.min([len(eigvals_model), len(eigvals_gt)])
    else:
        max_components = np.min([len(eigvals_model), len(eigvals_gt)])

    eigvals_model = eigvals_model[:max_components]
    eigvals_gt = eigvals_gt[:max_components]

    return compute_MSE(eigvals_model, eigvals_gt)

def participation_ratio(eigvals):
    """
    Compute the participation ratio of an eigenspectrum.

    Parameters
    ----------
    eigvals : array-like
      Eigenspectrum (e.g., eigenvalues of a covariance matrix).

    Returns
    -------
    float
      Participation ratio, defined as (sum(eigvals) ** 2) / sum(eigvals ** 2).
      Measures the effective dimensionality of the eigenspectrum.
    """
    eigvals = np.asarray(eigvals)
    pr = (np.sum(eigvals) ** 2) / (np.sum(eigvals ** 2) + 1e-12)
    return pr

def eigenspectrum_cov_pr(eigvals_model, eigvals_gt, method='cov'):
    """
    Compute the participation ratio of the covariance eigenspectrum and return the mse 
    between model and ground truth.
    
    Parameters
    ----------
    eigvals_model : array-like
      Model covariance eigenspectrum.
    eigvals_gt : array-like
      Ground-truth covariance eigenspectrum.
    method : str
      Included for compatibility with other cost functions and compute_cost.
      Passed through to correlation_eigenspectrum.

    Returns
    -------
    float
      MSE between model and ground-truth participation ratios.
    """
    
    pr_model = participation_ratio(eigvals_model)
    pr_gt = participation_ratio(eigvals_gt)

    return compute_MSE(pr_model, pr_gt)

def eigenspectrum_cov_alignment(eigvals_model, eigvals_gt, method='dot', max_components=None):
    """
    Compute the alignment between model and target covariance eigenspectra.

    Parameters
    ----------
    eigvals_model : array-like
      Model covariance eigenspectrum.
    eigvals_gt : array-like
      Ground-truth covariance eigenspectrum.
    method : str
      Included for compatibility with other cost functions and compute_cost.
      Passed through to correlation_eigenspectrum.
    max_components : int or None
      If provided, compare only the first max_components eigenvalues.

    Returns
    -------
    float
      Cosine similarity between model and ground-truth eigenspectra.
    """

    # Sort in descending order
    eigvals_model = np.sort(eigvals_model)[::-1]
    eigvals_gt = np.sort(eigvals_gt)[::-1]

    # Set max components
    if max_components is not None and max_components > 0:
       if max_components > len(eigvals_model) or max_components > len(eigvals_gt):
           print(
               f"Warning: max_components={max_components} exceeds number of available eigenvalues "
               f"(model has {len(eigvals_model)}, gt has {len(eigvals_gt)}). "
               "Using smaller value."
           )
           max_components = np.min([len(eigvals_model), len(eigvals_gt)])
    elif max_components is not None and max_components <= 0:
        print(f"Warning: max_components={max_components} is not positive. Ignoring and using all components.")
        max_components = np.min([len(eigvals_model), len(eigvals_gt)])
    else:
        max_components = np.min([len(eigvals_model), len(eigvals_gt)])

    eigvals_model = eigvals_model[:max_components]
    eigvals_gt = eigvals_gt[:max_components]

    # Compute cosine similarity (alignment)
    dot_product = np.dot(eigvals_model, eigvals_gt)
    if method == 'cosine':
        norm_product = np.linalg.norm(eigvals_model) * np.linalg.norm(eigvals_gt) + 1e-12
        alignment = dot_product / norm_product
    elif method == 'dot':
        alignment = dot_product
    else:
        raise ValueError(f"Unsupported method='{method}' for eigenspectrum_cov_alignment. Use 'cosine' or 'dot'.")

    return alignment

#%% Model fitting utilities

def compute_cost(model, data, method='fd', cost_function=kl_divergence):
    """
    Compute the cost between corresponding columns of model and data using a given cost function.
    
    Parameters:
    - model: np.array, shape (N_model, N_c), model-generated data
    - data: np.array, shape (N_data, N_c), real-world data
    - method: if int, number of bins for histogram estimation; if string, the method for choosing bin edges (default 'fd' for Freedman Diaconis Estimator)
    - cost_function: function, function that computes cost between two distributions (default: KL divergence)
    
    Returns:
    - cost_values: np.array, shape (N_c,), cost values for each column
    """
    # Validate inputs: model and data should have the same number of columns
    if model.ndim < 2 or data.ndim < 2:
      raise ValueError(f"compute_cost expects 2-D arrays. Got model.ndim={model.ndim}, data.ndim={data.ndim}")
    if model.shape[1] != data.shape[1]:
      raise ValueError(
        f"compute_cost: column mismatch — model has {model.shape[1]} columns but data has {data.shape[1]} columns. "
        f"Ensure input conditions (inputs-dir) align with target files (targets-dir) and that both are ordered consistently."
      )

    N_c = model.shape[1]  # Number of columns
    cost_values = np.zeros(N_c)  # Store cost for each column

    for i in range(N_c):
      cost_values[i] = cost_function(np.squeeze(model[:, i]), np.squeeze(data[:, i]), method=method)

    return np.sum(cost_values)

def expected_improvement(mu, sigma, best_cost):
    """
    Standard EI formula for Bayesian Optimization.
    best_cost is the best (lowest) cost so far.
    Z = (best_cost - mu)/sigma
    EI = (best_cost - mu)*Phi(Z) + sigma*phi(Z) if improvement>0 else 0
    """
    eps = 1e-12
    Z = (best_cost - mu) / (sigma + eps)

    phi = stats.norm.pdf(Z)
    Phi = stats.norm.cdf(Z)

    improvement = best_cost - mu
    ei = improvement * Phi + sigma * phi
    #ei[improvement < 0] = 0.0
    return ei