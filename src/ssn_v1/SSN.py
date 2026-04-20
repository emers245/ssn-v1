#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:18:10 2024

@author: Joe

SSN: A class for building and running a stabilized supralinear network

"""

import numpy as np
import json
import pandas as pd
import os, sys
import warnings
try:
    from . import SSN_utils
except Exception:
    import SSN_utils
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, interp1d, interpn
import scipy.sparse
import h5py
import paramiko
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
import tempfile
from typing import Dict, Tuple
from tqdm import tqdm
import torch
import torchsde
from scipy.optimize import OptimizeResult

# Custom exception for numerical instability
class NumericalInstabilityError(Exception):
    """Raised when numerical overflow or instability occurs during simulation."""
    pass
import getpass
import copy

class SSN:
    """
    An SSN class for building and analyzing stabilized supralinear networks (SSN).

    Default Attributes
    ----------
    network_name : str
        Unique identifier for the network.
    rand_seed : int
        Seed for random number generation to ensure reproducibility.
    parameters : dict
        Dictionary to hold network parameters.

    Methods
    -------

    __init__(network_name, rand_seed=0)
        Initializes the network with a unique name and an optional random seed.

    set_rand_seed(seed=None)
        Sets the random seed for reproducibility.
        
    add_nodes(params={}, useConfig=True)
        Adds nodes to the network with optional parameters and configurations.

    add_edges(params={}, useConfig=True, seed=None)
        Adds edges between nodes with specified configurations and randomization.

        
    load_inputs(**kwargs)
        Load inputs from local or remote location. Currently loads NPY files.
        
    run(**kwargs)
        Run simulation based on simulation parameters specified in
        self.parameters.
        
    save_outputs(**kwargs)
        Save ouputs in HDF5 format.
        
    load_config(file_path)
        Load network configuration file
        
    spatial_organization()
        Set sptatial organization of nodes
        
    visualize_graph()
        Visualize the network as a graph of nodes and edges
        
    construct_W()
        Construct an adjacency matrix from self.edges()
        
    load_target_data
        Load some target data to fit the model to
        
    save(file_path)
        Saves a copy of the current object as a joblib file
        
    load(file_path)
        Load an existing SSN object from a joblib file directly

    copy()
        Creates and returns a deep copy of the current SSN object.

    _fetch_remote_file(remote_server, username, password, remote_path, local_path)
        Fetch a file from a remote server using SSH and SFTP protocols.

    _push_remote_file(remote_server, username, password, local_path, remote_path)
    """
    
    
    def __init__(self, network_name, rand_seed = None, map_seed = None, verbose = True):
        """
        Initializes the SSN with a unique name and an optional random seed.

        Parameters
        ----------
        network_name : str
            A unique identifier for the network.
        rand_seed : int, optional
            Random seed for reproducibility (default is None and will be set by the config).
        map_seed : int, optional
            Random seed for spatial node layout (default is None).
        verbose : bool, optional
            If True, print informational messages and progress bars (default is True).

        Attributes Created
        ------------------
        name: str
            Network name.
        parameters : dict
            Initialized as an empty dictionary to store network parameters.
        seed: int
            Random seed.
        map_seed: int
            Random seed for node layout.
        verbose: bool
            Controls whether to print progress and informational messages.
        """
        
        self.name = network_name
        self.seed = rand_seed
        self.parameters = {}
        self.map_seed = map_seed
        self.verbose = verbose
        
    def set_rand_seed(self, seed = None, map_seed = None):
        """
        Sets the random seed for reproducibility.

        Parameters
        ----------
        seed : int, optional
            Seed value for the random number generator. If None, no seed is set.
        map_seed : int, optional
            Seed for spatial node layout generation. If None, no map seed is set.

        Attributes Modified
        -------------------
        seed : int
            Updated with the provided seed value if seed is not None.
        map_seed : int
            Updated with the provided map_seed value if map_seed is not None.
        """
        
        if seed is not None:
            self.seed = seed
        if map_seed is not None:
            self.map_seed = map_seed
        np.random.seed(self.seed)
    
    def add_nodes(self, params={}, useConfig=True, seed=None):
        """
        Adds nodes to the network with optional parameters adjustments.

        Parameters
        ----------
        params : dict, optional
            Node-specific parameters (default is an empty dictionary).
        useConfig : bool, optional
            Whether to use a configuration file for node properties (default is True).

        Attributes Created
        -------------------
        nodes : pandas.DataFrame
            Logs information about nodes based on the provided parameters.
        """

        if useConfig:

            if not hasattr(self, 'manifest'):
                try:
                    self.load_config(self.config, seed=seed) # the random seed will default to `seed` unless it is None, in which case it wille default to the network seed from the config file
                except:
                    raise Exception(f"SSN ERROR: Config file not found for {self.name}")
            else:
                if self.verbose:
                    print("Config file already loaded. Adding nodes...")

            # Set random seed
            if seed:
                np.random.seed(seed)
            else:
                if self.seed is None:
                    raise ValueError("SSN ERROR: SSN.add_nodes Random seed not set. Please set the random seed before adding nodes.")
                else:
                    np.random.seed(self.seed)

            # Create pandas dataframes for all the nodes and model types
            components_dirs = []
            for model_dir in self.components.keys():
                components_path = SSN_utils.resolve_path(self.components[model_dir], self.manifest)
                components_dirs.append(components_path)
            self.node_types = pd.DataFrame()
            node_types_list = []
            node_list = []
            node_index = 0
            for m_i, model_type in enumerate(self.parameters['nodes']['models']):
                config_file = f'config.{model_type}.json'
                found_paths = SSN_utils.find_file(config_file, components_dirs)
                if len(found_paths) > 1:
                    print("WARNING: Same model type found in more that one components directory!")
                config_path = os.path.join(found_paths[0])
                with open(config_path, 'r') as file:
                    model_config = json.load(file)
                # Dynamically create node entry from config parameters
                new_model = {
                    'model_type_id': model_config.get('cell_type_id', None),
                    'model_type_name': model_config.get('cell_type', None)
                }

                # Add other parameters from the config file
                for param, value in model_config.items():
                    if param not in ['cell_type_id', 'cell_type']:  # Avoid duplicating entries
                        new_model[param] = value

                node_types_list.append(new_model)

                # Add nodes to network
                for i in range(self.parameters['nodes']['Ncells'][model_type]):
                    new_node = {
                        "node_index": node_index,
                        "model_index": model_config.get('cell_type_id', None),
                        "model_name": model_type,
                        'ei': new_model['ei'],
                        'x': None,  # placeholder for x location
                        'y': None,  # placeholder for y location
                        'z': None   # placeholder for z location
                    }
                    node_list.append(new_node)
                    node_index += 1

            self.node_types = pd.DataFrame(node_types_list)
            self.nodes = pd.DataFrame(node_list)

            # Add in information about spatial organization
            self.spatial_organization()

        else:
            print("SSN ERROR: This functionality if not available yet. Configuration files are needed to add nodes.")

    def add_edges(self, params={}, useConfig=True, seed=None):
        """
        Adds edges between nodes with specified configurations and randomization.
        Uses vectorized NumPy broadcasting for improved performance.

        Parameters
        ----------
        params : dict, optional
            Edge-specific parameters (default is an empty dictionary).
        useConfig : bool, optional
            Whether to use a configuration file for edge properties (default is True).
        seed : int, optional
            Seed value for edge randomization (default is None).

        Attributes Created
        -------------------
        edges : pandas.DataFrame
            Logs information about edges based on the provided parameters.
        """

        # Invalidate cached W matrix since edges are being rebuilt
        if hasattr(self, 'W_sparse'):
            del self.W_sparse

        if useConfig:
            if not hasattr(self, 'manifest'):
                try:
                    self.load_config(self.config, seed=seed)
                except:
                    raise Exception(f"SSN ERROR: Config file not found for {self.name}")
            else:
                if self.verbose:
                    print("Config file already loaded. Adding edges...")

            # Set random seed
            if seed:
                np.random.seed(seed)
            else:
                if self.seed is None:
                    raise ValueError("SSN ERROR: SSN.add_nodes Random seed not set. Please set the random seed before adding nodes.")
                else:
                    np.random.seed(self.seed)

            # Set helper function for boundary conditions (supports broadcasting)
            boundary_cfg = self.parameters['edges'].get('boundary_conditions', None)

            def _pairwise_distance(post_x, post_y, pre_x, pre_y):
                """Compute pairwise distances. Inputs can be broadcast-compatible arrays,
                e.g. post_x shape (Npost, 1) and pre_x shape (1, Npre) to get (Npost, Npre)."""
                if boundary_cfg is None:
                    return np.sqrt((post_x - pre_x) ** 2 + (post_y - pre_y) ** 2)

                b_type = boundary_cfg.get('type', '').lower()

                if b_type == 'euclidean':
                    return np.sqrt((post_x - pre_x) ** 2 + (post_y - pre_y) ** 2)

                if b_type == 'toroidal':
                    extent = np.asarray(boundary_cfg.get('extent', self.parameters['nodes']['spatial_config']['scaleXY']))
                    dx = np.abs(post_x - pre_x)
                    dy = np.abs(post_y - pre_y)
                    dx = np.minimum(dx, extent[0] - dx)
                    dy = np.minimum(dy, extent[1] - dy)
                    return np.sqrt(dx ** 2 + dy ** 2)

                return np.sqrt((post_x - pre_x) ** 2 + (post_y - pre_y) ** 2)

            # Set up variables
            st = self.parameters['edges']['spatial_tuning']
            ft = self.parameters['edges']['func_tuning']
            edge_types = list(st.keys())

            # Reorganize node parameters
            cellLocs_x = {}
            cellLocs_y = {}
            cellOris = {}
            Ntype = {}
            node_index = {}
            for nt in list(self.node_types['model_type_name']):
                cellLocs_x[nt] = self.nodes[self.nodes['model_name'] == nt]['x'].values
                cellLocs_y[nt] = self.nodes[self.nodes['model_name'] == nt]['y'].values
                cellOris[nt] = self.nodes[self.nodes['model_name'] == nt]['orientation'].values
                Ntype[nt] = np.sum(self.nodes['model_name'] == nt)
                node_index[nt] = self.nodes[self.nodes['model_name'] == nt].index.values
            Nall = len(self.nodes)

            # --- Optimization 1: Vectorized connection probabilities ---
            pW = np.zeros([Nall, Nall])

            for et in st:
                pre_nt = st[et]['presyn_type']
                post_nt = st[et]['postsyn_type']
                pre_idx = node_index[pre_nt]
                post_idx = node_index[post_nt]

                # Since json files do not support np.inf, swap "inf" strings
                if st[et]['sigma'] == "inf":
                    st[et]['sigma'] = np.inf
                if ft[et]['sigma_ori'] == "inf":
                    ft[et]['sigma_ori'] = np.inf

                # Vectorized pairwise distance: (Npost, Npre)
                distance = _pairwise_distance(
                    cellLocs_x[post_nt][:, np.newaxis],
                    cellLocs_y[post_nt][:, np.newaxis],
                    cellLocs_x[pre_nt][np.newaxis, :],
                    cellLocs_y[pre_nt][np.newaxis, :],
                )

                # Vectorized connection probability: (Npost, Npre)
                spatial_prob = SSN_utils.Gaussian(distance, 0, st[et]['sigma'])
                ori_prob = SSN_utils.circGauss(
                    cellOris[pre_nt][np.newaxis, :],
                    cellOris[post_nt][:, np.newaxis],
                    ft[et]['sigma_ori'],
                    cyc=np.pi,
                )
                conn_prob = st[et]['kappa'] * spatial_prob * ori_prob

                pW[np.ix_(post_idx, pre_idx)] = conn_prob

            if self.verbose:
                print("Connection probabilities computed (vectorized).")

            randomNum = np.random.rand(Nall, Nall)
            pWbi = pW >= randomNum

            # --- Optimization 2: Vectorized weight assignment ---
            wAll = np.zeros([Nall, Nall])
            for et in st:
                pre_nt = st[et]['presyn_type']
                post_nt = st[et]['postsyn_type']
                pre_idx = node_index[pre_nt]
                post_idx = node_index[post_nt]
                conn_weight = st[et]['sigma_j'] * np.abs(st[et]['j']) * np.random.randn(Ntype[post_nt], Ntype[pre_nt]) + st[et]['j']
                wAll[np.ix_(post_idx, pre_idx)] = conn_weight

            if self.verbose:
                print("Weights computed (vectorized).")

            # --- Optimization 3: Vectorized E/I sign clipping ---
            W = pWbi * wAll
            Eindex = self.nodes[self.nodes['ei'] == 'e'].index.values
            Iindex = self.nodes[self.nodes['ei'] == 'i'].index.values
            W[:, Eindex] = np.maximum(W[:, Eindex], 0)
            W[:, Iindex] = np.minimum(W[:, Iindex], 0)

            # Flag if nans present in W
            if np.sum(np.isnan(W) > 0):
                print("WARNING: Found NaNs in W.")

            # Do not allow singleton nodes if indicated
            if not self.parameters['edges']['allow_singleton_nodes']:

                for et in st:
                    pre_nt = st[et]['presyn_type']
                    pre_ei = self.node_types[self.node_types['model_type_name'] == pre_nt]['ei'].values[0]
                    post_nt = st[et]['postsyn_type']
                    pre_nt_index = self.nodes[self.nodes['model_name'] == pre_nt].index.values
                    post_nt_index = self.nodes[self.nodes['model_name'] == post_nt].index.values
                    W_submat = W[np.ix_(post_nt_index, pre_nt_index)]
                    in_weight_sum = np.sum(W_submat, axis=1)
                    pW_submat = pW[np.ix_(post_nt_index, pre_nt_index)]
                    singletons = np.where(in_weight_sum == 0)[0]

                    if len(singletons) > 0:
                        if self.verbose:
                            print(f"Adding connections for {len(singletons)} singletons ({et})...")

                        for singl_id in singletons:
                            connection_id = np.argmax(pW_submat[singl_id, :])
                            weight = 0
                            idx = 0

                            if pre_ei == 'e':
                                while weight <= 0 and idx < 100:
                                    weight = st[et]['sigma_j'] * np.abs(st[et]['j']) * np.random.randn() + st[et]['j']
                                    idx += 1
                                    if idx >= 100:
                                        print(f"Failed to converge on non-zero weight for connection type {et}. Check edge connection probabilities.")
                            elif pre_ei == 'i':
                                while weight >= 0 and idx < 100:
                                    weight = st[et]['sigma_j'] * np.abs(st[et]['j']) * np.random.randn() + st[et]['j']
                                    idx += 1
                                    if idx >= 100:
                                        print(f"Failed to converge on non-zero weight for connection type {et}. Check edge connection probabilities.")
                            else:
                                raise ValueError(f"Unknown 'ei' parameter: {pre_ei}")

                            W_submat[singl_id, connection_id] = weight

                        W[np.ix_(post_nt_index, pre_nt_index)] = W_submat

            # If standardization is requested, rescale weights
            if self.parameters['edges']['standardize_recurrence']:
                for et in st:
                    W_orig = W
                    pre_nt = st[et]['presyn_type']
                    post_nt = st[et]['postsyn_type']
                    pre_nt_index = self.nodes[self.nodes['model_name'] == pre_nt].index.values
                    post_nt_index = self.nodes[self.nodes['model_name'] == post_nt].index.values
                    in_weight_sum = np.sum(W_orig[np.ix_(post_nt_index, pre_nt_index)], axis=1)
                    in_weight_sum[in_weight_sum == np.nan] = 10 ^ -16
                    W[np.ix_(post_nt_index, pre_nt_index)] = np.abs(self.parameters['edges']['spatial_tuning'][et]['j']) * (W_orig[np.ix_(post_nt_index, pre_nt_index)] / np.abs(in_weight_sum[:, np.newaxis]))

            # --- Optimization 4: Vectorized edge DataFrame construction ---
            edges = np.nonzero(W)
            edges_post = edges[0]
            edges_pre = edges[1]
            Nedges = len(edges_post)

            node_index_vals = self.nodes['node_index'].values
            model_name_vals = self.nodes['model_name'].values
            ei_vals = self.nodes['ei'].values

            self.edges = pd.DataFrame({
                "edge_index": np.arange(Nedges),
                "weight": W[edges_post, edges_pre],
                "pre_model_index": node_index_vals[edges_pre],
                "post_model_index": node_index_vals[edges_post],
                "pre_model_name": model_name_vals[edges_pre],
                "post_model_name": model_name_vals[edges_post],
                "pre_ei": ei_vals[edges_pre],
                "post_ei": ei_vals[edges_post],
            })

            if self.verbose:
                print(f"Edge list created: {Nedges} edges (vectorized).")

        else:
            print("SSN ERROR: This functionality if not available yet. Configuration files are needed to add edges.")

    def load_inputs(self, file_path=None, local=True, remote_server=None, username=None, password=None):
        """
        Loads external inputs to a new attribute from .npy or .h5 file.
        
        Parameters
        ----------
        file_path : str
            Path to .npy or .h5 file containing external inputs. Default is None.
        local : bool, optional
            If True, directs the method to look for the file locally (default
            is True).
        remote_server : str, optional
            If exists, specifies a remote server to access to locate files.
            Server access is executed via SSH protocol (default is None).
        username : str, optional
            If exists, specifies the username for server access (default is
            None).
        password : str, optional
            If exists, specified the password for server access (default is 
            None).
            

        Attributes Created
        -------------------
        inputs: numpy.array
            A timeseries array of external inputs of shape NX x NY x NT x Nori
            where NX and NY are spatial dimensions, NT is a temporal dimension,
            and Nori is a functional dimension, here specified as orientation.
        """

        # Find inputs file if one not provided
        if file_path is None:
            # Get file path from self.inputs_config
            if not hasattr(self, 'inputs_config'):
                raise ValueError("Error: No input file path specified and no inputs_config found.")
            else:
                base_dir = self.manifest["$BASE_DIR"]
                inputs_dir = self.manifest["$INPUTS_DIR"].replace("$BASE_DIR", base_dir)
                file_path = self.inputs_config['file_name'].replace("$INPUTS_DIR", inputs_dir)
        
        if local:
            # Load inputs locally
            # Check file extension
            if file_path.endswith('.h5') or file_path.endswith('.hdf5'):
                with h5py.File(file_path, 'r') as h5file:
                    # Assuming the dataset is named 'inputs'; adjust as necessary
                    self.inputs = h5file['stimulus'][:]
            elif file_path.endswith('.npy'):
                self.inputs = np.load(file_path)
            else:
                raise ValueError("Unsupported file format. Please provide a .npy or .h5 file.")
        else:
            if remote_server is None:
                try:
                    remote_server = self.inputs_server
                except AttributeError:
                    raise ValueError("Remote server address and username are required for remote loading.")

            temp_file_path = None
            try:
                _, ext = os.path.splitext(file_path)
                with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_file:
                    temp_file_path = tmp_file.name

                self._fetch_remote_file(
                    remote_server=remote_server,
                    username=username,
                    password=password,
                    remote_path=file_path,
                    local_path=temp_file_path,
                )

                if ext in ('.h5', '.hdf5'):
                    with h5py.File(temp_file_path, 'r') as h5file:
                        self.inputs = h5file['stimulus'][:]
                elif ext == '.npy':
                    self.inputs = np.load(temp_file_path)
                else:
                    raise ValueError("Unsupported file format. Please provide a .npy or .h5 file.")
            except Exception as e:
                print(f"Failed to load {file_path} from {remote_server}: {e}")
                raise
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                
    def connect_inputs(self, inputs=None, input_coord_range = [[-16,16],[-16,16]], seed=None, mapping_method=None):
        """
        Connects external inputs to nodes in network based on input type.
        
        Parameters
        ----------
        inputs: Array of inputs to connect to model. If None, default to
            self.inputs if available.
        input_coord_range : array-like
            Specifies the coordinate range for the spatial dimensions. Array
            should be formatted [[dim1_min, dim2_max], [dim2_min, dim2_max]]
            (default is [[-16,16],[-16,16]]). Only used for "current_field" type.
        seed : int, optional
            Random seed for reproducibility. If None, use self.seed (default is None).
        mapping_method : dict, optional
            Specifies the mapping methods for spatial and functional dimensions.

        Attributes Created
        -------------------
        h: numpy.array
            An array of mapped time-series inputs. The first dimension indexes 
            nodes and the second dimension indexes time points.
        """

        if not hasattr(self, 'manifest'): #check if config loaded
            try:
                self.load_config(self.config, seed=seed)  # the random seed will default to `seed` unless it is None, in which case it wille default to the network seed from the config file
            except:
                raise Exception(f"SSN ERROR: Config file not found for {self.name}")
        else:
            if self.verbose:
                print("Config file already loaded. Adding edges...")
        
        # Get input type from config
        if not hasattr(self, 'stim_params'):
            input_type = "current_field"  # Default to current_field if no stim_params
        else:
            input_type = self.stim_params["stim_type"]

        # Check mapping_method
        if mapping_method is None:
            # Check network config for mapping methods
            if 'extrinsic' in self.parameters:
                if 'mapping_methods' in self.parameters['extrinsic']:
                    mapping_method = self.parameters['extrinsic']['mapping_methods']
                else:
                    if self.verbose:
                        print("No mapping methods found. Using default nearest.")
                    mapping_method = {'spatial': {'type': 'nearest'}, 'functional': {'type': 'nearest'}}
            else:
                if self.verbose:
                    print("No mapping methods found. Using default nearest.")
                mapping_method = {'spatial': {'type': 'nearest'}, 'functional': {'type': 'nearest'}}
        
        if input_type == "currents":
            # Input is already in Ncells x Ntime format
            if inputs is None:
                if hasattr(self, "inputs"):
                    self.h = self.inputs
                else:
                    raise ValueError("Error: No input data found.")
            else:
                self.h = inputs
                
        elif input_type == "current_field":
            # Map spatial current field to network nodes (existing functionality)
            
            # Connect feedforward inputs to nodes in network
            # First two dimension of inputs should be two spatial dimensions (x,y)
            # Time is the third index
            # Orientation is the last index

            # Assume that the array is linearly spaced and that the spatial bounds are specified
            # If coordinate range exists in self.stim_params, use that instead
            if hasattr(self, 'stim_params'):
                if "x_coord_range" in self.stim_params:
                    input_coord_range[0] = self.stim_params["x_coord_range"]
                if "y_coord_range" in self.stim_params:
                    input_coord_range[1] = self.stim_params["y_coord_range"]
                # If orientation channels specified in self.stim_params, use that instead
                if "orientation_channels" in self.stim_params:
                    input_coord_ori = np.asarray(self.stim_params["orientation_channels"], dtype=float)
                    # Detect if degrees
                    if np.max(input_coord_ori) > 2*np.pi:
                        input_coord_ori = np.deg2rad(input_coord_ori)
                else:
                    input_coord_ori = np.linspace(0,np.pi-(np.pi/np.shape(self.inputs)[3]),np.shape(self.inputs)[3])
            else:
                if self.verbose:
                    print("No stim_params found. Using default coordinate ranges and orientation channels.")
                input_coord_ori = np.linspace(0,np.pi-(np.pi/np.shape(self.inputs)[3]),np.shape(self.inputs)[3])
            
            # Input coordinate space
            if inputs is None:
                if hasattr(self,"inputs"):
                    input_coord_x = np.linspace(input_coord_range[0][0],input_coord_range[0][1],np.shape(self.inputs)[0])
                    input_coord_y = np.linspace(input_coord_range[1][0],input_coord_range[1][1],np.shape(self.inputs)[1])
                else:
                    raise ValueError("Error: No input data found.")
            else:
                input_coord_x = np.linspace(input_coord_range[0][0],input_coord_range[0][1],np.shape(inputs)[0])
                input_coord_y = np.linspace(input_coord_range[1][0],input_coord_range[1][1],np.shape(inputs)[1])
                
            # Lookup coordinates
            node_coord_x = self.nodes.x
            node_coord_y = self.nodes.y
            n_nodes = len(self.nodes)
            node_input_map = np.zeros((2,n_nodes))
            
            for x_i, coord_x in enumerate(node_coord_x):
                node_input_map[0][x_i] = int(np.argmin(np.abs(coord_x - input_coord_x)))
            for y_i, coord_y in enumerate(node_coord_y):
                node_input_map[1][y_i] = int(np.argmin(np.abs(coord_y - input_coord_y)))
            
            # Lookup orientation preference
            ori_vec = self.nodes.orientation
            
            def _map_current_field_to_inputs(
                source_inputs,
                node_coord_x,
                node_coord_y,
                node_orientations,
                input_coord_x,
                input_coord_y,
                input_coord_ori,
                mapping_method=None,
            ):
                node_coord_x = np.asarray(node_coord_x)
                node_coord_y = np.asarray(node_coord_y)
                node_orientations = np.asarray(node_orientations)
                mapping_method = mapping_method or {}
                spatial_cfg = mapping_method.get("spatial", {})
                functional_cfg = mapping_method.get("functional", {})
                spatial_type = spatial_cfg.get("type", "nearest").lower()
                spatial_method = spatial_cfg.get("method", "linear").lower()
                spatial_kwargs = spatial_cfg.get("kwargs", {})
                functional_type = functional_cfg.get("type", "nearest").lower()
                functional_method = functional_cfg.get("method", "linear").lower()
                functional_kwargs = functional_cfg.get("kwargs", {})

                NX, NY, NT, N_ori = source_inputs.shape
                n_nodes = len(node_coord_x)
                spatial_samples = np.zeros((n_nodes, NT, N_ori))

                def _nearest_index(value, grid):
                    return int(np.argmin(np.abs(value - grid)))

                if spatial_type == "nearest":
                    x_idx = np.array([_nearest_index(x, input_coord_x) for x in node_coord_x], dtype=int)
                    y_idx = np.array([_nearest_index(y, input_coord_y) for y in node_coord_y], dtype=int)
                    for node_idx in range(n_nodes):
                        spatial_samples[node_idx] = source_inputs[x_idx[node_idx], y_idx[node_idx], :, :]
                elif spatial_type == "interp":
                    points = (input_coord_x, input_coord_y)
                    coords = np.column_stack((node_coord_x, node_coord_y))
                    if spatial_method == "linear":
                        for t in range(NT):
                            for ori_idx in range(N_ori):
                                spatial_samples[:, t, ori_idx] = interpn(
                                    points,
                                    source_inputs[:, :, t, ori_idx],
                                    coords,
                                    method="linear",
                                    **spatial_kwargs,
                                )
                    elif spatial_method == "cubic_spline":
                        for t in range(NT):
                            for ori_idx in range(N_ori):
                                frame = source_inputs[:, :, t, ori_idx]
                                cs_x = CubicSpline(input_coord_x, frame, axis=0, **spatial_kwargs)
                                x_interp = cs_x(node_coord_x)
                                for node_idx in range(n_nodes):
                                    cs_y = CubicSpline(input_coord_y, x_interp[node_idx], **spatial_kwargs)
                                    spatial_samples[node_idx, t, ori_idx] = cs_y(node_coord_y[node_idx])
                    else:
                        raise ValueError(f"Unknown spatial interpolation method '{spatial_method}'.")
                else:
                    raise ValueError(f"Unknown spatial mapping type '{spatial_type}'.")

                mapped_inputs = np.zeros((n_nodes, NT))
                orientation_period = np.pi

                if functional_type == "nearest":
                    for node_idx in range(n_nodes):
                        ori_idx = int(
                            np.argmin(
                                np.abs(
                                    SSN_utils.circD(
                                        node_orientations[node_idx],
                                        input_coord_ori,
                                        orientation_period,
                                    )
                                )
                            )
                        )
                        mapped_inputs[node_idx] = spatial_samples[node_idx, :, ori_idx]
                elif functional_type == "interp":
                    input_coord_ori = np.asarray(input_coord_ori, dtype=float)
                    sorted_idx = np.argsort(input_coord_ori)
                    input_coord_ori = input_coord_ori[sorted_idx]
                    spatial_samples = spatial_samples[:, :, sorted_idx]

                    unique_ori, unique_idx = np.unique(input_coord_ori, return_index=True)
                    input_coord_ori = unique_ori
                    spatial_samples = spatial_samples[:, :, unique_idx]

                    ori_extended = np.concatenate([input_coord_ori, input_coord_ori + orientation_period])
                    for node_idx in range(n_nodes):
                        ori_val = np.mod(node_orientations[node_idx], orientation_period)
                        data = spatial_samples[node_idx]
                        data_extended = np.concatenate([data, data], axis=1)
                        if functional_method == "cubic_spline":
                            cs_ori = CubicSpline(
                                ori_extended,
                                data_extended,
                                axis=1,
                                **functional_kwargs,
                            )
                            mapped_inputs[node_idx] = cs_ori(ori_val)
                        elif functional_method == "linear":
                            interp_fn = interp1d(
                                ori_extended,
                                data_extended,
                                axis=1,
                                kind="linear",
                                **functional_kwargs,
                            )
                            mapped_inputs[node_idx] = interp_fn(ori_val)
                        else:
                            raise ValueError(
                                f"Unknown functional interpolation method '{functional_method}'."
                            )
                else:
                    raise ValueError(f"Unknown functional mapping type '{functional_type}'.")

                return mapped_inputs
            
            source_inputs = self.inputs if inputs is None else inputs
            mapped_inputs = _map_current_field_to_inputs(
                source_inputs=source_inputs,
                node_coord_x=node_coord_x.values,
                node_coord_y=node_coord_y.values,
                node_orientations=self.nodes.orientation.values,
                input_coord_x=input_coord_x,
                input_coord_y=input_coord_y,
                input_coord_ori=input_coord_ori,
                mapping_method=mapping_method,
            )
            
            # Determine input vector
            self.h = mapped_inputs
            
        elif input_type == "feature_map":
            # Map feature maps through receptive fields to generate currents
            
            # Get inputs dimensions and setup
            if inputs is None:
                if hasattr(self, "inputs"):
                    feature_inputs = self.inputs
                else:
                    raise ValueError("Error: No input data found.")
            else:
                feature_inputs = inputs
            
            NX, NY, NT, N_ori = feature_inputs.shape
            n_nodes = len(self.nodes)
            
            # Get RF and tuning parameters from network config
            rf_type = self.parameters['extrinsic']['rf_type']
            rf_params = self.parameters['extrinsic']['rf_params']
            tuning_params = self.parameters['extrinsic']['tuning_params']

            # Set random seed for reproducible RF generation
            if seed:
                np.random.seed(seed)
            else:
                if self.seed is None:
                    raise ValueError("SSN ERROR: SSN.add_nodes Random seed not set. Please set the random seed before adding nodes.")
                else:
                    np.random.seed(self.seed)

            # Assume that the array is linearly spaced and that the spatial bounds are specified
            # If coordinate range exists in self.stim_params, use that instead
            if hasattr(self, 'stim_params'):
                if "x_coord_range" in self.stim_params:
                    input_coord_x = self.stim_params["x_coord_range"]
                else:
                    input_coord_x = np.linspace(input_coord_range[0][0], input_coord_range[0][1], NX)
                if "y_coord_range" in self.stim_params:
                    input_coord_y = self.stim_params["y_coord_range"]
                else:
                    input_coord_y = np.linspace(input_coord_range[1][0], input_coord_range[1][1], NY)
                # If orientation channels specified in self.stim_params, use that instead
                if "orientation_channels" in self.stim_params:
                    input_coord_ori = self.stim_params["orientation_channels"]
                else:
                    input_coord_ori = np.linspace(0, np.pi - (np.pi / N_ori), N_ori)
            else:
                if self.verbose:
                    print("No stim_params found. Using default coordinate ranges and orientation channels.")
                input_coord_x = np.linspace(input_coord_range[0][0], input_coord_range[0][1], NX)
                input_coord_y = np.linspace(input_coord_range[1][0], input_coord_range[1][1], NY)
                input_coord_ori = np.linspace(0, np.pi - (np.pi / N_ori), N_ori)
            
            # Create meshgrid for spatial coordinates
            X_input, Y_input = np.meshgrid(input_coord_x, input_coord_y, indexing='ij')
            
            # Initialize output currents
            currents = np.zeros((n_nodes, NT))
            
            # Generate RFs and tuning for each node
            for node_idx in range(n_nodes):
                node_row = self.nodes.iloc[node_idx]
                node_x = node_row['x']
                node_y = node_row['y']
                node_ori = node_row['orientation']
                
                # Generate spatial RF parameters
                if tuning_params['mu_mean'] is None:
                    # Use node's orientation preference
                    rf_ori_mean = node_ori
                else:
                    # Use specified mean orientation
                    rf_ori_mean = tuning_params['mu_mean']
                
                # Sample RF parameters from distributions
                if rf_type == 'gaussian':
                    rf_size = np.random.normal(rf_params['rf_size_mean'], rf_params['rf_size_std'])
                    rf_size = max(rf_size, 0.1)  # Ensure positive size
                    
                    # Create Gaussian spatial profile
                    spatial_profile = np.exp(-((X_input - node_x)**2 + (Y_input - node_y)**2) / (2 * rf_size**2))
                    
                elif rf_type == 'circular':
                    rf_size = np.random.normal(rf_params['rf_size_mean'], rf_params['rf_size_std'])
                    rf_size = max(rf_size, 0.1)  # Ensure positive size
                    
                    # Create circular spatial profile (step function)
                    distance = np.sqrt((X_input - node_x)**2 + (Y_input - node_y)**2)
                    spatial_profile = (distance <= rf_size).astype(float)
                    
                else:
                    raise ValueError(f"Unknown RF type: {rf_type}. Supported types are 'gaussian' and 'circular'.")
                
                # Sample orientation tuning parameters
                ori_center = np.random.normal(rf_ori_mean, tuning_params['mu_std'])
                ori_width = np.random.normal(tuning_params['sigma_mean'], tuning_params['sigma_std'])
                ori_width = max(ori_width, 1.0)  # Ensure positive width
                ori_scale = np.random.normal(tuning_params['scale_mean'], tuning_params['scale_std'])
                ori_offset = np.random.normal(tuning_params['offset_mean'], tuning_params['offset_std'])
                
                # Create orientation tuning curve (circular Gaussian)
                ori_diff = input_coord_ori - np.radians(ori_center)
                # Handle circular nature of orientation (wrap around at pi)
                ori_diff = np.abs(ori_diff)
                ori_diff = np.minimum(ori_diff, np.pi - ori_diff)
                
                # Convert ori_width from degrees to radians
                ori_width_rad = np.radians(ori_width)
                ori_tuning = ori_scale * np.exp(-(ori_diff**2) / (2 * ori_width_rad**2)) + ori_offset
                ori_tuning = np.maximum(ori_tuning, 0)  # Ensure non-negative
                
                # Compute weighted input for this node across time
                for t in range(NT):
                    # Apply spatial RF to each orientation channel
                    weighted_input = 0
                    for ori_idx in range(N_ori):
                        # Spatial integration: sum over RF-weighted feature map
                        spatial_response = np.sum(spatial_profile * feature_inputs[:, :, t, ori_idx])
                        
                        # Apply orientation tuning
                        weighted_input += spatial_response * ori_tuning[ori_idx]
                    
                    currents[node_idx, t] = weighted_input
            
            # Set the final currents
            self.h = currents
            
        else:
            raise ValueError(f"Unknown input type: {input_type}. Supported types are 'currents', 'current_field', and 'feature_map'.")

    def run(self, params = {}, useConfig = True, seed = None, saveLog=False, events=None):
        """
        Runs numerical approximations of the solutions to the SSN system based
        on simulation parameters. Leverages scipy.integrate.solve_ivp for 
        numerical integration.
        
        Parameters
        ----------
        params : dict, optional
            Specify simulation parameter directly, otherwise use parameters 
            from self.parameters (default is {}).
        useConfig : bool, optional
            If True, use parameters specified in config file (default is True).
        seed : int, optional
            If exists, specify random seed, otherwise use self.seed (default is
            None).
        saveLog : bool, optional
            If True, save solver output to log file (default is False).
        events : callable or list of callables, optional
            Event functions to monitor during integration. Events can be used to
            detect specific conditions (e.g., threshold crossings) and optionally
            terminate integration early. Set event.terminal = True to stop 
            integration when the event is triggered. See scipy.integrate.solve_ivp
            documentation for details (default is None).
            

        Attributes Created
        -------------------
        outputs: scipy.integrate.OdeResult
            A scipy object containing the results of the numerical 
            approximation.
            
            This contains the following attributes:
                t : ndarray
                    Time points.
                y : ndarray
                    Values of solutions at t.
                sol : OdeSolution or None
                    Found solution as OdeSolution instance; None if 
                    dense_output was set to False.
                t_events : list of ndarray or None
                    Contains for each event type a list of arrays at which an 
                    event of that type event was detected. None if events was 
                    None.
                y_events : list of ndarray or None
                    For each value of t_events, the corresponding value of the 
                    solution. None if events was None.
                nfev : int
                    Number of evaluations of the right-hand side.
                njev : int
                    Number of evaluations of the Jacobian.
                nlu : int
                    Number of LU decompositions.
                status : int
                    Reason for algorithm termination:
                        -1: Integration step failed.
                        0: The solver successfully reached the end of tspan.
                        1: A termination event occurred.
                message : str
                    Human-readable description of the termination reason.
                success : bool
                    True if the solver reached the interval end or a 
                    termination event occurred (status >= 0).

        
        """
        
        # Run the network simulation

        # Set random seed
        if seed:
            np.random.seed(seed)
        else:
            if self.seed is None:
                raise ValueError("SSN ERROR: SSN.add_nodes Random seed not set. Please set the random seed before adding nodes.")
            else:
                np.random.seed(self.seed)
        
        def init_r(init_params,node_types):
            
            if init_params['type'] == 'uniform':
                r_initial = np.ones(W.shape[0])
                for nt in node_types.unique():
                    r_initial[node_types == nt] = init_params[nt]
                return(r_initial)
            elif init_params['type'] == 'random':
                r_initial = np.ones(W.shape[0])
                for nt in node_types.unique():
                    if init_params[nt]['sampling_type'] == 'uniform':
                        r_initial[node_types == nt] = (init_params[nt]['upper']-init_params[nt]['lower'])*np.random.rand(np.sum(node_types==nt))+init_params[nt]['lower']
                    elif init_params[nt]['sampling_type'] == 'normal':
                        r_initial[node_types == nt] = init_params[nt]['std']*np.random.randn(np.sum(node_types==nt))+init_params[nt]['mean']
                    else:
                        print("ERROR run: unknown initialization sampling method")
                return(r_initial)
            elif init_params['type'] == 'custom':
                r_initial = np.ones(W.shape[0])
                for nt in node_types.unique():
                    r_initial[node_types == nt] = init_params[nt]
                return(r_initial)
            else:
                print("ERROR run: unknown initialization type")
        
        if useConfig:
            if not hasattr(self, 'run_params'):
                print("run_params attribute not found. Loading from config file...")
                try:
                    self.load_config(self.config)
                except:
                    print(f"SSN ERROR: Config file not found for {self.name}")
            
            # Adjacency matrix (use cached if available)
            W = self.construct_W()
            # Use sparse representation for efficient matrix-vector multiply in ODE RHS
            if hasattr(self, 'W_sparse'):
                W_sparse = self.W_sparse
            else:
                W_sparse = scipy.sparse.csr_matrix(W)
            
            # Initial firing rates
            if 'r_init' in self.run_params:
                r_init = init_r(self.run_params['r_init'],self.nodes['model_name'])
            else:
                raise ValueError("SSN ERROR: Initial firing rates not specified in run_params.")
    
            # Run time vector
            t = np.arange(0, self.run_params['tstop'], self.run_params['dt'])
            
            # Interpolate inputs
            if hasattr(self, 'stim_params'):
                T_input = self.stim_params['temporal']['params']['T'] # duration of input vector in ms
                t_step_input = self.stim_params['temporal']['params']['t_steps'] # time step of input vector in ms
            else:
                if self.verbose:
                    print("SSN Warning: No input stimulus parameters found. Assuming input duration and time step match runtime parameters.")
                T_input = self.run_params['tstop'] # Assume duration of input matches the runtime duration
                t_step_input = T_input / self.h.shape[1] # Assume time step of input matches the runtime duration divided by number of input time points
            #print(len(np.arange(0, T_input, t_step_input)), self.h.shape)
            cs = CubicSpline(np.arange(0, T_input, t_step_input), self.h, axis = 1) # create a cubic spline of the input vector
    
            # Populate node parameters via vectorized map (avoids per-type DataFrame loop)
            nodes_df = self.nodes
            param_lookup = {}
            for _, row in self.node_types.iterrows():
                param_lookup[row['model_type_id']] = {
                    'tau': row['tau'], 'k': row['k'], 'c': row['c'], 'n': row['n']
                }
            nodes_df['tau'] = nodes_df['model_index'].map(lambda x: param_lookup[x]['tau'])
            nodes_df['k'] = nodes_df['model_index'].map(lambda x: param_lookup[x]['k'])
            nodes_df['c'] = nodes_df['model_index'].map(lambda x: param_lookup[x]['c'])
            nodes_df['n'] = nodes_df['model_index'].map(lambda x: param_lookup[x]['n'])
            
            tau = nodes_df['tau'].values
            k = nodes_df['k'].values
            c = nodes_df['c'].values
            n = nodes_df['n'].values

            # Save new values back to self.nodes
            self.nodes = nodes_df

            # Define the differential equation as a closure (captures tau, k, c, n, W_sparse, cs)
            def ssn_equations(t, r):
                h_t = cs(t) #evaluate the interpolated inputs at each time in vector t
                
                r_ss = k * np.power(np.maximum(c * h_t + W_sparse @ r, 0), n)
                drdt = (-r + r_ss) / tau
                return drdt
            
            if 'max_step' in self.run_params.keys():
                max_step = self.run_params['max_step']
            else:
                max_step = np.inf
            
            noise_matrix = self.run_params.get('noise_matrix', None)
            if noise_matrix is None:
                noise_matrix = self.run_params.get('noise_matrix_file', None)

            def _run_ode_solver():
                # run simulation
                # record stdout to log file
                if saveLog:
                    if hasattr(self, 'outputs_config') and 'log_file' in self.outputs_config:
                        log_filename = self.outputs_config['log_file']
                        outputs_dir = self.outputs_config['outputs_dir']
                        log_path = os.path.join(outputs_dir, log_filename)
                    else:
                        log_path = f"{self.name}_log.txt"

                    original_stdout = sys.stdout
                    original_stderr = sys.stderr
                    with open(log_path, 'w') as log_file:
                        sys.stdout = log_file
                        sys.stderr = log_file
                        with warnings.catch_warnings():
                            warnings.filterwarnings('error', category=RuntimeWarning, message='.*overflow.*')
                            try:
                                return solve_ivp(
                                    ssn_equations,
                                    [0, t[-1]],
                                    r_init,
                                    method=self.run_params['method'],
                                    max_step=max_step,
                                    events=events
                                )
                            except RuntimeWarning as e:
                                raise NumericalInstabilityError(f"Overflow detected during integration: {e}")
                            finally:
                                sys.stdout = original_stdout
                                sys.stderr = original_stderr
                else:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('error', category=RuntimeWarning, message='.*overflow.*')
                        try:
                            return solve_ivp(
                                ssn_equations,
                                [0, t[-1]],
                                r_init,
                                method=self.run_params['method'],
                                max_step=max_step,
                                events=events
                            )
                        except RuntimeWarning as e:
                            raise NumericalInstabilityError(f"Overflow detected during integration: {e}")

            def _load_noise_matrix(matrix_spec):
                if matrix_spec is None:
                    return None
                if isinstance(matrix_spec, np.ndarray):
                    return matrix_spec
                if isinstance(matrix_spec, list):
                    return np.array(matrix_spec)
                if isinstance(matrix_spec, str):
                    if not os.path.isfile(matrix_spec):
                        raise FileNotFoundError(f"SSN ERROR: noise matrix file not found: {matrix_spec}")
                    ext = os.path.splitext(matrix_spec)[1].lower()
                    if ext == '.npy':
                        return np.load(matrix_spec)
                    if ext == '.npz':
                        npz = np.load(matrix_spec)
                        if 'noise_matrix' in npz:
                            return npz['noise_matrix']
                        first_key = list(npz.keys())[0]
                        return npz[first_key]
                    if ext in ['.h5', '.hdf5']:
                        with h5py.File(matrix_spec, 'r') as f:
                            if 'noise_matrix' in f:
                                return np.array(f['noise_matrix'])
                            first_key = list(f.keys())[0]
                            return np.array(f[first_key])
                    raise ValueError("SSN ERROR: Unsupported noise matrix file extension. Use .npy, .npz, .h5, or .hdf5")
                raise ValueError("SSN ERROR: noise_matrix must be ndarray/list/path.")

            noise_matrix = _load_noise_matrix(noise_matrix)

            if noise_matrix is None:
                r_t = _run_ode_solver()
                self.outputs = r_t
                return

            noise_matrix = np.array(noise_matrix, dtype=float)
            if noise_matrix.ndim != 2 or noise_matrix.shape[0] != noise_matrix.shape[1]:
                raise ValueError("SSN ERROR: run_params['noise_matrix'] must be an N x N matrix.")
            if noise_matrix.shape[0] != W.shape[0]:
                raise ValueError(
                    f"SSN ERROR: noise matrix shape {noise_matrix.shape} must match number of nodes {W.shape[0]}."
                )

            L = noise_matrix

            method_requested = str(self.run_params.get('method', 'euler'))
            method_l = method_requested.lower()
            method_aliases = {
                'euler': 'euler',
                'milstein': 'milstein',
                'srk': 'srk',
                'heun': 'heun',
                'midpoint': 'midpoint',
                'reversible_heun': 'reversible_heun'
            }
            sde_method = method_aliases.get(method_l, method_l)
            supported_sde_methods = set(method_aliases.values())
            if sde_method not in supported_sde_methods:
                raise ValueError(
                    f"SSN ERROR: run_params['method']={method_requested} is not a supported torchsde method. "
                    f"Use one of: {sorted(supported_sde_methods)}"
                )

            torch_dtype = torch.float64 if str(self.run_params.get('dtype', '')).lower() in ['float64', 'torch.float64'] else torch.float32
            torch_device = torch.device(self.run_params.get('device', 'cpu'))
            batch_size = int(self.run_params.get('batch_size', 1))
            if batch_size < 1:
                raise ValueError("SSN ERROR: run_params['batch_size'] must be >= 1.")

            W_t = torch.tensor(W, dtype=torch_dtype, device=torch_device)
            tau_t = torch.tensor(tau, dtype=torch_dtype, device=torch_device)
            k_t = torch.tensor(k, dtype=torch_dtype, device=torch_device)
            c_t = torch.tensor(c, dtype=torch_dtype, device=torch_device)
            n_t = torch.tensor(n, dtype=torch_dtype, device=torch_device)
            L_t = torch.tensor(L, dtype=torch_dtype, device=torch_device)
            t_t = torch.tensor(t, dtype=torch_dtype, device=torch_device)
            r_init_arr = np.array(r_init, dtype=float).reshape(-1)
            if r_init_arr.shape[0] != W.shape[0]:
                raise ValueError(
                    f"SSN ERROR: initial rates size {r_init_arr.shape[0]} must match number of nodes {W.shape[0]}."
                )
            y0_base = torch.tensor(r_init_arr, dtype=torch_dtype, device=torch_device).unsqueeze(0)
            y0_t = y0_base.repeat(batch_size, 1)

            class _SSNSDE(torch.nn.Module):
                noise_type = 'additive'
                sde_type = 'ito'

                def __init__(self, cs_local):
                    super().__init__()
                    self.cs_local = cs_local

                def f(self, t_curr, y_curr):
                    h_t = self.cs_local(float(t_curr.item()))
                    h_t = torch.tensor(h_t, dtype=torch_dtype, device=torch_device)
                    recur = torch.matmul(y_curr, W_t.T)
                    r_ss = k_t * torch.pow(torch.clamp(c_t * h_t + recur, min=0.0), n_t)
                    return (-y_curr + r_ss) / tau_t

                def g(self, t_curr, y_curr):
                    batch_size = y_curr.shape[0]
                    return L_t.unsqueeze(0).expand(batch_size, -1, -1)

            sde = _SSNSDE(cs)
            dt_sde = float(self.run_params.get('dt', np.min(np.diff(t))))

            if events is None:
                ys = torchsde.sdeint(sde, y0_t, t_t, method=sde_method, dt=dt_sde)
                # ys shape: (T, B, N)
                y_batch = np.maximum(ys.detach().cpu().numpy().transpose(1, 2, 0), 0.0)  # (B, N, T)
                y_out = y_batch[0]  # backward-compatible primary output
                self.outputs = OptimizeResult({
                    't': np.array(t),
                    'y': y_out,
                    'y_batch': y_batch,
                    't_events': None,
                    'y_events': None,
                    'status': 0,
                    'message': 'The solver successfully reached the end of tspan.',
                    'success': True,
                    'solver': 'torchsde',
                    'method': sde_method,
                    'batch_size': batch_size
                })
                return

            event_list = events if isinstance(events, (list, tuple)) else [events]
            t_events = [[] for _ in event_list]
            y_events = [[] for _ in event_list]

            def _event_crossed(v0, v1, direction):
                if direction > 0:
                    return (v0 < 0 and v1 >= 0)
                if direction < 0:
                    return (v0 > 0 and v1 <= 0)
                return (v0 == 0) or (v1 == 0) or (v0 < 0 < v1) or (v1 < 0 < v0)

            y_series = [y0_t[0].detach().cpu().numpy()]
            y_series_batch = [y0_t.detach().cpu().numpy().copy()]
            t_series = [t[0]]
            y_prev = y0_t
            terminal_triggered = False
            terminal_message = 'The solver successfully reached the end of tspan.'

            for idx in range(1, len(t)):
                t_seg = torch.tensor([t[idx - 1], t[idx]], dtype=torch_dtype, device=torch_device)
                ys_seg = torchsde.sdeint(sde, y_prev, t_seg, method=sde_method, dt=dt_sde)
                y_curr = ys_seg[-1]

                y_prev_np = y_prev[0].detach().cpu().numpy()
                y_curr_np = y_curr[0].detach().cpu().numpy()

                for eidx, ev in enumerate(event_list):
                    v0 = float(ev(t[idx - 1], y_prev_np))
                    v1 = float(ev(t[idx], y_curr_np))
                    direction = getattr(ev, 'direction', 0)
                    if _event_crossed(v0, v1, direction):
                        t_events[eidx].append(float(t[idx]))
                        y_events[eidx].append(y_curr_np.copy())
                        if getattr(ev, 'terminal', False):
                            terminal_triggered = True
                            terminal_message = 'A termination event occurred.'

                y_series.append(y_curr_np.copy())
                y_series_batch.append(y_curr.detach().cpu().numpy().copy())
                t_series.append(t[idx])
                y_prev = y_curr

                if terminal_triggered:
                    break

            y_batch = np.maximum(np.stack(y_series_batch, axis=0).transpose(1, 2, 0), 0.0)  # (B, N, T)
            self.outputs = OptimizeResult({
                't': np.array(t_series),
                'y': np.maximum(np.array(y_series).T, 0.0),
                'y_batch': y_batch,
                't_events': [np.array(v) for v in t_events],
                'y_events': [np.array(v) for v in y_events],
                'status': 1 if terminal_triggered else 0,
                'message': terminal_message,
                'success': True,
                'solver': 'torchsde',
                'method': sde_method,
                'batch_size': batch_size
            })
            return
            
        else:
            print(f"SSN ERROR: This functionality if not available yet. Configuration files are needed to add nodes.")
    
    def save_outputs(self, output_path=None, local=True, remote_server=None, username=None, password=None, output_type='all'):
        """
        Saves outputs to an HDF5 file, along with associated JSON files containing simulation parameters.
        
        Parameters
        ----------
        output_path : str
            Path to output file location for the HDF5 file. Default is None.
        local : bool, optional
            If True, the file is saved locally. If False, the file is saved on a remote server via SSH.
            (Default is True.)
        remote_server : str, optional
            Remote server address. Required if local is False.
        username : str, optional
            Username for remote server access. Required if local is False.
        password : str, optional
            Password for remote server access.
        output_type : str or list, optional
            Controls which response data are saved to HDF5:
                - 'all' (default): save full time series (datasets 't' and 'y').
                - 'end': save only final timepoint of y (dataset 'y').
                - [t_start, t_end]: save mean firing rates over times within
                  t_start <= t <= t_end (dataset 'y').
        """

        def prepare_output_data(output_type):
            t_vals = np.asarray(self.outputs['t'])
            y_vals = np.asarray(self.outputs['y'])

            if output_type == 'all':
                return t_vals, y_vals

            if output_type == 'end':
                return None, y_vals[:, -1]

            if isinstance(output_type, (list, tuple, np.ndarray)):
                if len(output_type) != 2:
                    raise ValueError("output_type list must have length 2: [start_time, end_time]")
                t_start = float(output_type[0])
                t_end = float(output_type[1])
                if t_end < t_start:
                    raise ValueError("output_type time window must satisfy end_time >= start_time")

                t_mask = (t_vals >= t_start) & (t_vals <= t_end)
                if not np.any(t_mask):
                    raise ValueError(
                        f"No time points fall within output_type window [{t_start}, {t_end}]"
                    )

                y_mean = np.mean(y_vals[:, t_mask], axis=1)
                return None, y_mean

            raise ValueError("output_type must be 'all', 'end', or a length-2 list [start_time, end_time]")

        if output_path is None:
            # Look for output path in config
            outputs_dir = self.outputs_config['outputs_dir'].replace('$BASE_DIR', self.manifest['$BASE_DIR'])
            rates_h5 = self.outputs_config['rates_h5']
            output_path = os.path.join(outputs_dir, rates_h5)
        
        # Function to convert non-serializable objects for JSON
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert NumPy arrays to lists
            if isinstance(obj, (np.int32, np.int64, np.float32, np.float64)):
                return obj.item()  # Convert NumPy scalars to Python scalars
            if isinstance(obj, set):
                return list(obj)  # Convert sets to lists
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        # Determine the names of the JSON parameter files based on the output_path.
        parameters_file, ext = os.path.splitext(output_path)
        parameters_filename = parameters_file + '_parameters.json'
        simparameters_filename = parameters_file + '_sim_parameters.json'
        t_to_save, y_to_save = prepare_output_data(output_type)
        
        if local:
            output_file = output_path
            
            # Save the simulation outputs to the HDF5 file
            with h5py.File(output_file, 'w') as h5file:
                # Save output data according to output_type
                if t_to_save is not None:
                    h5file.create_dataset('t', data=t_to_save)
                h5file.create_dataset('y', data=y_to_save)
                # Save cell type indices
                h5file.create_dataset('model_index', data=self.nodes['model_index'])
        
                # Save scalar values
                h5file.attrs['success'] = self.outputs['success']
                h5file.attrs['status'] = self.outputs['status']
                h5file.attrs['nfev'] = self.outputs['nfev']
                h5file.attrs['njev'] = self.outputs['njev']
                h5file.attrs['nlu'] = self.outputs['nlu']
        
                # Save strings as attributes
                h5file.attrs['message'] = self.outputs['message']
        
                # Handle None values for attributes
                h5file.attrs['sol'] = "None" if self.outputs['sol'] is None else self.outputs['sol']
                h5file.attrs['t_events'] = "None" if self.outputs['t_events'] is None else self.outputs['t_events']
                h5file.attrs['y_events'] = "None" if self.outputs['y_events'] is None else self.outputs['y_events']
                
                # Save paths to the JSON parameter files as attributes
                h5file.attrs['parameters_file'] = parameters_filename
                h5file.attrs['sim_parameters_file'] = simparameters_filename
                h5file.attrs['output_type'] = str(output_type)
            
            # Save simulation parameters to JSON files locally
            with open(parameters_filename, 'w') as outfile:
                json.dump(self.parameters, outfile, default=convert_to_serializable)
            with open(simparameters_filename, 'w') as outfile:
                json.dump(self.run_params, outfile, default=convert_to_serializable)
        
        else:
            # Remote saving: ensure required remote parameters are provided.
            if remote_server is None or username is None:
                raise ValueError("Remote server address and username are required for remote saving.")
            
            # Create temporary local files for the HDF5 and JSON data.
            tmp_h5 = tempfile.NamedTemporaryFile(delete=False)
            tmp_h5_path = tmp_h5.name
            tmp_h5.close()  # We only need the name; h5py will open the file.
            
            tmp_params = tempfile.NamedTemporaryFile(delete=False, suffix='_parameters.json')
            tmp_params_path = tmp_params.name
            tmp_params.close()
            
            tmp_sim_params = tempfile.NamedTemporaryFile(delete=False, suffix='_sim_parameters.json')
            tmp_sim_params_path = tmp_sim_params.name
            tmp_sim_params.close()
            
            try:
                # Save the HDF5 file to the temporary location.
                with h5py.File(tmp_h5_path, 'w') as h5file:
                    if t_to_save is not None:
                        h5file.create_dataset('t', data=t_to_save)
                    h5file.create_dataset('y', data=y_to_save)
                    # Save cell type indices
                    h5file.create_dataset('model_index', data=self.nodes['model_index'])
        
                    h5file.attrs['success'] = self.outputs['success']
                    h5file.attrs['status'] = self.outputs['status']
                    h5file.attrs['nfev'] = self.outputs['nfev']
                    h5file.attrs['njev'] = self.outputs['njev']
                    h5file.attrs['nlu'] = self.outputs['nlu']
        
                    h5file.attrs['message'] = self.outputs['message']
        
                    h5file.attrs['sol'] = "None" if self.outputs['sol'] is None else self.outputs['sol']
                    h5file.attrs['t_events'] = "None" if self.outputs['t_events'] is None else self.outputs['t_events']
                    h5file.attrs['y_events'] = "None" if self.outputs['y_events'] is None else self.outputs['y_events']
        
                    # Write the JSON file names as attributes (so that the HDF5 file is aware of them)
                    h5file.attrs['parameters_file'] = parameters_filename
                    h5file.attrs['sim_parameters_file'] = simparameters_filename
                    h5file.attrs['output_type'] = str(output_type)
                
                # Save the JSON parameter files to the temporary locations.
                with open(tmp_params_path, 'w') as outfile:
                    json.dump(self.parameters, outfile, default=convert_to_serializable)
                with open(tmp_sim_params_path, 'w') as outfile:
                    json.dump(self.run_params, outfile, default=convert_to_serializable)

                # Push to remote server
                self._push_remote_file(remote_server, username, password, tmp_h5_path, output_path)
            
            except Exception as e:
                print(f"Failed to save outputs to remote server: {e}")
                raise
            
            finally:
                # Clean up temporary files.
                if os.path.exists(tmp_h5_path):
                    os.remove(tmp_h5_path)
                if os.path.exists(tmp_params_path):
                    os.remove(tmp_params_path)
                if os.path.exists(tmp_sim_params_path):
                    os.remove(tmp_sim_params_path)

    def load_config(self, file_path, base_dir = None, set_seed=True, set_map_seed=True, username=None, password=None, local_inputs=False):
        """
        Load parameters from a configuration file.
        
        Parameters
        ----------
        file_path : str
            Path to configuration file.
        base_dir : str, optional
            If exists, override base directory specified in configuration file
            (default is None).
        set_seed : bool, optional
            If True, set the random seed from the configuration file (default is
            False). If self.seed is None, the seed is always set from the config file.
        set_map_seed : bool, optional
            If True, set the map seed from the configuration file (default is False). 
            If self.map_seed is None, the map seed is always set from the config file.
        username : str, optional
            Username for server access if inputs are on a remote server (default is None).
        password : str, optional
            Password for server access if inputs are on a remote server (default is None).
            
        Attributes Created
        -------------------
        config: str
            Path to config file.
        manifest : dict
            File locations established by config file.
        run_params : dict
            Simulation parameters used in numerical integration.
        network_config : dict
            Location of files related to network configuration.
        inputs_config : dict
            Inputs configuration file locations and parameters.
        outputs_config : dict
            Outputs configuration file locations and parameters.
        components_config : dict
            Components configuration file locations and parameters.
        components: dict
            All components parameters.
         
        Attributes Modified
        -------------------
        parameters : dict
            All network parameters.
        
        """
        
        self.config = file_path
        with open(file_path, 'r') as file:
            c = json.load(file)
            self.manifest = c["manifest"]
            self.run_params = c["run"]
            self.network_config = c["network"]
            self.inputs_config = c["inputs"]
            self.outputs_config = c["outputs"]
            self.components_config = c["components"]
            
            # Get the network config file path
            if base_dir is None:
                base_dir = c["manifest"]["$BASE_DIR"]
            else:
                self.manifest['$BASE_DIR'] = base_dir #set base_dir manually
            network_dir = c["manifest"]["$NETWORK_DIR"].replace("$BASE_DIR", base_dir)
            network_config_path = c["network"]["network_config"].replace("$NETWORK_DIR", network_dir)

            # Get the input config file path
            inputs_dir = c["manifest"]["$INPUTS_DIR"].replace("$BASE_DIR", base_dir)
            if "inputs_config" in c["inputs"]:
                input_config_path = c["inputs"]["inputs_config"].replace("$INPUTS_DIR", inputs_dir)
            else:
                # Warning
                if self.verbose:
                    print("SSN WARNING: No input configuration file specified in config. Using default input configuration.")
                input_config_path = os.path.join(inputs_dir, f"./inputs/config.inputs.json")
            
            # Get server location if one included
            if "server" in c["inputs"] and not local_inputs:
                self.inputs_server = c["inputs"]["server"]
            else:
                self.inputs_server = None
            
        # Load the network configuration file
        with open(network_config_path, 'r') as file:
            c = json.load(file)
            self.parameters = c["network"]
            self.components = c["components"]

        # Load the input configuration file if it exists
        if self.inputs_server is None:
            if os.path.exists(input_config_path):
                with open(input_config_path, 'r') as file:
                    c = json.load(file)
                    self.stim_params = c["stimulus"]
                    self.stim_file = c["file_name"]
                    self.stim_model = c["model"]
            else:
                if self.verbose:
                    print(f"SSN WARNING: Input configuration file not found at {input_config_path}. Using default input configuration.")
        else:  
            if self.verbose:
                print(f"Trying to load input configuration from {self.inputs_server}...")
            if username is None:
                raise ValueError("Username is required to access the input configuration on the server.")
            if password is None:
                raise ValueError("Password is required to access the input configuration on the server.")

            temp_file_path = None
            try:
                _, ext = os.path.splitext(input_config_path)
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
                    temp_file_path = tmp_file.name
                
                self._fetch_remote_file(self.inputs_server, username, password, input_config_path, temp_file_path)

                with open(temp_file_path, 'r') as file:
                    c = json.load(file)
                    self.stim_params = c["stimulus"]
                    self.stim_file = c["file_name"]
                    self.stim_model = c["model"]
            
            except Exception as e:
                print(f"Failed to load input configuration from server: {e}")
            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

        # Set random seed from config
        if self.seed is None:
            self.seed = self.network_config['rand_seed']
        elif set_seed:
            self.seed = self.network_config['rand_seed']
        if self.map_seed is None:
            if 'map_seed' in self.network_config.keys():
                self.map_seed = self.network_config['map_seed']
            else:
                self.map_seed = None
        elif set_map_seed:
            self.map_seed = self.network_config['map_seed']
            
    def spatial_organization(self, seed = None):
        """
        Creates a spatial map and locations for the nodes to occupy.
        
        Parameters
        ----------
        seed : int, optional
            Random seed (default is None).
            
        Attributes Created
        -------------------
        grid : numpy.array
            Grid of discrete spatial locations. X coordinates occupy the first
            component of the first dimension, and Y coordinates occupy the
            second component of the first dimension.
        
        Attributes Modified
        -------------------
        nodes : pandas.DataFrame
            Adds in new columns tracking spatial position of nodes ('x','y',
            'x_i','y_i') and associated functional properties ('orientation').
        """
        
        def set_position(self):
            
            def make_grid(self):
                
                #Make Grid
                deltaXY = self.parameters['nodes']['spatial_config']['deltaXY']
                scaleXY = self.parameters['nodes']['spatial_config']['scaleXY']
                RFx = np.linspace(-scaleXY[0]/2,scaleXY[0]/2,int(scaleXY[0]/deltaXY)) #x locations of each receptive field in degrees of visual angle
                RFy = np.linspace(-scaleXY[1]/2,scaleXY[1]/2,int(scaleXY[1]/deltaXY)) #y locations of each receptive field in degrees of visual angle
                RFxy = np.meshgrid(RFx,RFy)
                RFxy = np.array([RFxy[0],RFxy[1]])
                self.grid = RFxy
                
                return(RFxy, scaleXY, RFx, RFy)
                
            RFxy, scaleXY, RFx, RFy = make_grid(self)
        
            #Sample from grid
        
            Ntotal = self.nodes.shape[0]
            
            if self.parameters['nodes']['spatial_config']['sampling'] == None:
                if self.verbose:
                    print("No spatial configuration selected")
                
            elif self.parameters['nodes']['spatial_config']['sampling'] == 'random':
                
                # Set random seed
                if seed:
                    np.random.seed(seed)
                else:
                    if self.seed is None:
                        raise ValueError("SSN ERROR: SSN.add_nodes Random seed not set. Please set the random seed before adding nodes.")
                    else:
                        np.random.seed(self.seed)
                
                if self.parameters['nodes']['spatial_config']['mask'] == 'circle':
                    #Apply mask
                    rad = np.min(scaleXY)/2 #radius of region that we are allowing cells to occupy
                    dist = np.sqrt(RFxy[0,:,:]**2 + RFxy[1,:,:]**2) #get radial distances from [0,0]
                    within_radius_indices = np.where(dist <= rad)
                    x_coords_within_radius = RFxy[0, within_radius_indices[0], within_radius_indices[1]]
                    y_coords_within_radius = RFxy[1, within_radius_indices[0], within_radius_indices[1]]
                    coords_within_radius = np.array([x_coords_within_radius, y_coords_within_radius])
                    
                    #Randomly select positions
                    num_coords = coords_within_radius.shape[1]
                    
                    if self.parameters['nodes']['spatial_config']['yoke_subtypes']:
    
                        #Check that there are equal numbers of cells in different subgroups
                        Ntype = self.parameters['nodes']['Ncells'][self.parameters['nodes']['models'][0]]
                        for model_type in self.parameters['nodes']['models']:
                            if Ntype != self.parameters['nodes']['Ncells'][model_type]:
                                raise Exception("ERROR: Yoked cell types requires having the same number of nodes per type!")
                        
                        if num_coords >= Ntotal:
                            indices = np.random.choice(num_coords, Ntype, replace=False)
                            cellLocs = coords_within_radius[:, indices]
                            cellLocs = np.tile(cellLocs,(1,len(self.parameters['nodes']['models']))) #repeat same coordinates for all cell types
                        else:
                            if self.verbose:
                                print("Not enough coordinates within radius to select unique points.")
                            # Handle this case as needed, for example by just taking as many as available:
                            selected_coords = coords_within_radius
                        cellInds = np.array([np.searchsorted(RFx,cellLocs[0,:]),np.searchsorted(RFy,cellLocs[1,:])]) #cell indicies in x-y coords
                        
                    else:
                        
                        if num_coords >= Ntotal:
                            indices = np.random.choice(num_coords, Ntotal, replace=False)
                            cellLocs = coords_within_radius[:, indices]
                        else:
                            if self.verbose:
                                print("Not enough coordinates within radius to select unique points.")
                            # Handle this case as needed, for example by just taking as many as available:
                            selected_coords = coords_within_radius
                        cellInds = np.array([np.searchsorted(RFx,cellLocs[0,:]),np.searchsorted(RFy,cellLocs[1,:])]) #cell indicies in x-y coords    
                        
                else:
                    
                    if self.parameters['nodes']['spatial_config']['yoke_subtypes']:
                        
                        #Check that there are equal numbers of cells in different subgroups
                        Ntype = self.parameters['nodes']['Ncells'][self.parameters['nodes']['models'][0]]
                        for model_type in self.parameters['nodes']['models']:
                            if Ntype != self.parameters['nodes']['Ncells'][model_type]:
                                raise Exception("ERROR: Yoked cell types requires having the same number of nodes per type!")
                           
                        cellLocs = np.array([np.random.choice(RFx,size=Ntype),np.random.choice(RFy,size=Ntype)]) #cell locations in x-y coords
                        cellLocs = np.tile(cellLocs,(1,len(self.parameters['nodes']['models']))) #repeat same coordinates for all cell types
                        cellInds = np.array([np.searchsorted(RFx,cellLocs[0,:]),np.searchsorted(RFy,cellLocs[1,:])]) #E cell indicies in x-y coords
                        
                    else:
                    
                        cellLocs = np.array([np.random.choice(RFx,size=Ntotal),np.random.choice(RFy,size=Ntotal)]) #cell locations in x-y coords
                        cellInds = np.array([np.searchsorted(RFx,cellLocs[0,:]),np.searchsorted(RFy,cellLocs[1,:])]) #cell indicies in x-y coords
                        
            elif self.parameters['nodes']['spatial_config']['sampling'] == 'grid':
                
                # Put a cell of each type at each grid position
                if Ntotal != (np.size(RFxy)/2)*len(self.node_types):
                    raise Exception("Invalid number of neurons for spatial sampling type 'grid'.")
                    
                # Assign cell location
                x_coords = RFxy[0, :, :].flatten() #extract coordinates
                y_coords = RFxy[1, :, :].flatten()
                x_full = np.tile(x_coords, len(self.node_types))  # lengthen to match number of cell types
                y_full = np.tile(y_coords, len(self.node_types))
                cellLocs = np.vstack([x_full, y_full])
                cellInds = np.array([np.searchsorted(RFx,cellLocs[0,:]),np.searchsorted(RFy,cellLocs[1,:])]) #cell indicies in x-y coords
            
            else:
                raise Exception("ERROR: Unknown spatial sampling type")
            
            self.nodes['x'] = cellLocs[0,:]
            self.nodes['y'] = cellLocs[1,:]
            self.nodes['x_i'] = cellInds[0,:]
            self.nodes['y_i'] = cellInds[1,:]
            
        def set_orientation(self):
            if 'kc' in self.parameters['nodes']['spatial_config']['ori_map']: #check if kc is specified
                kc = self.parameters['nodes']['spatial_config']['ori_map']['kc'] #(Nx*deltaXY)/Ncycles I'm matching roughly to the scale we want to simulate for Flexible Ferrets
            elif 'Ncycles' in self.parameters['nodes']['spatial_config']['ori_map']: #if kc not specified, specify from Ncycles
                Ncycles = self.parameters['nodes']['spatial_config']['ori_map']['Ncycles'] #number of orientation cycles per unit distance
                kc = Ncycles*2*np.pi #set kc from Ncycles
            nMap = self.parameters['nodes']['spatial_config']['ori_map']['nMap'] #period of ori waveform over space
            map_seed = self.map_seed
            if self.verbose:
                print(f"Creating OriMap - kc = {kc}, nMap = {nMap}, map_seed = {map_seed}")
            OriMap = SSN_utils.makeOriMap(kc,nMap,self.grid,seed=map_seed) #define orientation tuning map
            
            Ntotal = self.nodes.shape[0]
            cellInds = np.vstack((self.nodes['x_i'].values, self.nodes['y_i'].values))
            oris = OriMap[cellInds[0,:],cellInds[1,:]]
            self.nodes['orientation'] = oris
            self.OriMap = OriMap
            
        # Call the functions to set positions and orientations
        set_position(self)
        set_orientation(self)
        
    def visualize_graph(self, node_color='node_index', edge_color=None, marker_styles=None, linewidth=None, cmap='plasma', show_colorbar=False, circular_layout=False):
        
        """
        Creates network visualizations.
        
        Parameters
        ----------
        node_color : str or array-like or None, optional
            If a sting, node_color indicates the column name of self.nodes on which 
            to base the coloring of nodes in the visualization (e.g. 
            'node_index') bases coloring on the unique index of each node. If 
            array-like, node_color is set according to the values in the array. 
            The array must be the same length as the number of nodes in the 
            network. If None, use default coloring from matplotlib.pyplot 
            (default is 'node_index').
        edge_color : str or array-like or None, optional
            Same as node_color but for edges. If a string, edge_color indicates
            the column name of self.edges on which to base the coloring of 
            edges uns the visualization (e.g. 'edge_index'). If array-like, 
            edge_color is set according to the values in the array. The array
            must be the same length as the number of edges in the network. If
            None, use default coloring from matplotlib.pyplot (default is 
            None).
        marker_styles : dict, optional
            Marker styles defined for each model type 
            (e.g. {'model_type_1': 'o', 'model_type_2': 'sq'}) (default is 
            None).
        linewidth : float, optional
            Specifies linewidth of edges, otherwise uses default from
            matplotlib.pyplot (default is None).
        cmap : str, optional
            Specifies the colormap to use, otherwise uses default from
            matplotlib.pyplot (default is 'plasma').
        show_colorbar : bool, optional
            If True, shows colorbar (default is False).
        circular_layout : bool, optional
            If True, plot ignores spatial location of nodes and plots network
            on a circle (default is False).
        """
        
        # Determine if z-coordinate contains None
        if self.nodes['z'].isnull().any():
            plot_3d = False
        else:
            plot_3d = True

        # Create plot
        import matplotlib.pyplot as plt
        fig = plt.figure()
        if plot_3d:
            ax = fig.add_subplot(111, projection='3d')
        else:
            ax = fig.add_subplot(111)

        # If node_color or edge_color is a string, use it as a column name, otherwise use the passed array
        node_colors = self.nodes[node_color] if isinstance(node_color, str) else node_color
        edge_colors = self.edges[edge_color] if isinstance(edge_color, str) else edge_color

        # Get unique model types for node coloring
        model_types = self.nodes['model_name'].unique()

        # Set marker styles for each model type
        if marker_styles is None:
            marker_styles = {model: 'o' for model in model_types}
        else:
            marker_styles = {model: marker_styles.get(model, 'o') for model in model_types}

        # Calculate positions for circular layout if specified
        if circular_layout:
            n_nodes = len(self.nodes)
            theta = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
            radius = 1.0

            # Map node indices to circular positions
            position_map = {}
            start = 0
            for model_type in model_types:
                model_nodes = self.nodes[self.nodes['model_name'] == model_type]
                num_model_nodes = len(model_nodes)
                indices = model_nodes.index

                for i, idx in enumerate(indices):
                    position_map[idx] = (radius * np.cos(theta[start + i]), radius * np.sin(theta[start + i]), 0)
                
                start += num_model_nodes
        else:
            # Use spatial coordinates from the dataframe
            position_map = {idx: (row['x'], row['y'], row['z']) for idx, row in self.nodes.iterrows()}

        # Plot each model type separately
        for model_type in model_types:
            model_nodes = self.nodes[self.nodes['model_name'] == model_type]
            x_vals, y_vals, z_vals = [], [], []

            for idx in model_nodes.index:
                x_val, y_val, z_val = position_map[idx]
                x_vals.append(x_val)
                y_vals.append(y_val)
                z_vals.append(z_val)

            # Plot nodes
            if plot_3d:
                h = ax.scatter(x_vals, y_vals, z_vals, c=node_colors[model_nodes.index], label=model_type, marker=marker_styles[model_type], cmap=cmap)
            else:
                h = ax.scatter(x_vals, y_vals, c=node_colors[model_nodes.index], label=model_type, marker=marker_styles[model_type], cmap=cmap)

        if show_colorbar:
            plt.colorbar(h, label=node_color)

        # Plot edges as arrows and add edge type legend
        edge_legend_added = {'e': False, 'i': False}
        for _, edge in self.edges.iterrows():
            pre_idx = edge['pre_model_index']
            post_idx = edge['post_model_index']
            
            pre_x, pre_y, pre_z = position_map[pre_idx]
            post_x, post_y, post_z = position_map[post_idx]

            # Determine edge color
            if edge_color is not None:
                edge_c = edge_colors[edge.name]
            else:
                edge_c = 'black'

            if plot_3d:
                # Plot arrows for 3D plot
                ax.quiver(pre_x, pre_y, pre_z, post_x-pre_x, post_y-pre_y, post_z-pre_z,
                          color=edge_c, width=linewidth, arrow_length_ratio=0.1)
            else:
                # Plot arrows for 2D plot using quiver
                ax.quiver(pre_x, pre_y, post_x-pre_x, post_y-pre_y,
                          angles='xy', scale_units='xy', scale=1, color=edge_c, width=linewidth)

            # Add edge type to legend
            if edge_color is not None and not edge_legend_added[edge['pre_ei']]:
                ax.plot([], [], c=edge_c, label=f"Edge Type: {edge['pre_ei']}", linewidth=1)
                edge_legend_added[edge['pre_ei']] = True

        # Set labels and legend
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        if plot_3d:
            ax.set_zlabel('Z')

        ax.legend()
        plt.show()
        
    def construct_W(self):
        """
        Constructs the adjacency matrix from the edges attribute.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        W : numpy.array
            Adjacency matrix (Nnodes x Nnodes). Standard convention, columns 
            indicate outgoing connections and rows indicate incoming
            connections.
        
        Attributes Created
        ------------------
        W_sparse : scipy.sparse.csr_matrix
            Cached sparse (CSR) adjacency matrix for efficient matrix-vector
            multiplication.
        """
        
        # Return dense copy from cached sparse W if available
        if hasattr(self, 'W_sparse') and self.W_sparse is not None:
            return self.W_sparse.toarray()
        
        # Construct adjecency matrix from edges
          
        try:
            # Initialize the adjacency matrix with zeros
            N = len(self.nodes)
            W = np.zeros((N, N))
            
            # Vectorized assignment from edges DataFrame
            pre_indices = self.edges['pre_model_index'].values
            post_indices = self.edges['post_model_index'].values
            weights = self.edges['weight'].values
            W[post_indices, pre_indices] = weights
            
            # Cache sparse representation only
            self.W_sparse = scipy.sparse.csr_matrix(W)
            
            return W
        except:
            print("ERROR construct_W: Don't have nodes and/or edges")
            
    def load_target_data(self, file, group, name = 'None'):
        """
        Loads target data for model fitting. Data needs to be formatted so that
        it can be matched to nodes 1:1.
        
        Parameters
        ----------
        file : str
            Path to target data in HDF5 format.
        group : str
            Group in HDF5 file to locate data.
        name : str, optional
            Name of variable to search in HDF5 file. Otherwise, use default 
            from group (default is None).
            
        Attributes Created
        -------------------
        target_data : array-like
            Target time-series data.
        """
        
        # Load some data that the model can be fit to
        # Expects the data to be in HDF5 format
        
        if not hasattr(self, 'target_data'):
            self.target_data = {}
        
        try:
            target = h5py.File(file)
            try:
                data = target[group]
                if not name:
                    name = group.split('/')[-1]
                self.target_data[name] = data[:]
            except:
                print("ERROR: Group not found in target file.")
        except:
            print("ERROR: Unable to load target data")
        
        target.close()   
        
    def save(self, file_path=None):
        """
        Save the SSN object to a file. If no file path is provided, 
        a default filename based on the current date and time is used.
        
        Parameters
        ----------
        file_path : str, optional
            Path to output file, otherwise use default naming convention 
            (default is None).
        """
        if file_path is None:
            # Look for file name in self.network_config
            if 'file_name' in self.network_config.keys():
                base_dir = self.manifest["$BASE_DIR"]
                network_dir = self.manifest["$NETWORK_DIR"].replace("$BASE_DIR", base_dir)
                file_path = self.network_config['file_name'].replace("$NETWORK_DIR", network_dir)

            else:
                # Generate a default filename based on date and time
                current_time = datetime.now()
                file_path = current_time.strftime('%Y-%m-%d_%H-%M-%S') + '.joblib'
        
        if '.' not in file_path:  # Add extension if necessary
            file_path = file_path + '.joblib'
        
        # Save the SSN object to the specified file
        joblib.dump(self, file_path)
        print(f"SSN object saved to {file_path}")

    def copy(self):
        new_ssn = SSN(self.name, self.seed, self.map_seed)
        try:
            new_ssn.parameters = copy.deepcopy(self.parameters)
        except:
            print("SSN WARNING: Could not deepcopy parameters.")
        try:
            new_ssn.network_config = copy.deepcopy(getattr(self, "network_config", None))
        except:
            print("SSN WARNING: Could not deepcopy network_config.")
        try:
            new_ssn.inputs_config = copy.deepcopy(getattr(self, "inputs_config", None))
        except:
            print("SSN WARNING: Could not deepcopy inputs_config.")
        try:
            new_ssn.outputs_config = copy.deepcopy(getattr(self, "outputs_config", None))
        except:
            print("SSN WARNING: Could not deepcopy outputs_config.")
        try:
            new_ssn.components = copy.deepcopy(getattr(self, "components", None))
        except:
            print("SSN WARNING: Could not deepcopy components.")
        try:
            new_ssn.manifest = copy.deepcopy(getattr(self, "manifest", None))
        except:
            print("SSN WARNING: Could not deepcopy manifest.")
        try:
            new_ssn.node_types = getattr(self, "node_types", pd.DataFrame()).copy(deep=True)
        except:
            print("SSN WARNING: Could not deepcopy node_types.")
        try:
            new_ssn.nodes = getattr(self, "nodes", pd.DataFrame()).copy(deep=True)
        except:
            print("SSN WARNING: Could not deepcopy nodes.")
        try:
            new_ssn.edges = getattr(self, "edges", pd.DataFrame()).copy(deep=True)
        except:
            print("SSN WARNING: Could not deepcopy edges.")
        try:
            new_ssn.h = np.copy(self.h) if hasattr(self, "h") else None
        except:
            print("SSN WARNING: Could not copy h.")
        try:
            new_ssn.inputs = np.copy(self.inputs) if hasattr(self, "inputs") else None
        except:
            print("SSN WARNING: Could not copy inputs.")
        try:
            new_ssn.outputs = copy.deepcopy(getattr(self, "outputs", None))
        except:
            print("SSN WARNING: Could not deepcopy outputs.")
        try:
            new_ssn.stim_params = copy.deepcopy(getattr(self, "stim_params", None))
        except:
            print("SSN WARNING: Could not deepcopy stim_params.")
        try:
            new_ssn.run_params = copy.deepcopy(getattr(self, "run_params", None))
        except:
            print("SSN WARNING: Could not deepcopy run_params.")
        try:
            new_ssn.OriMap = np.copy(getattr(self, "OriMap", np.array([])))
        except:
            print("SSN WARNING: Could not copy OriMap.")
        return new_ssn

    def _fetch_remote_file(self, remote_server, username, password, remote_path, local_path):
        """
        Fetch a file from a remote server using SFTP.
        
        Parameters
        ----------
        remote_server : str
            Address of the remote server.
        username : str
            Username for remote server access.
        password : str
            Password for remote server access.
        remote_path : str
            Path to the file on the remote server.
        local_path : str
            Path to save the file locally.
        """
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(remote_server, username=username, password=password)
            sftp = ssh.open_sftp()
            sftp.get(remote_path, local_path)
            sftp.close()
            ssh.close()
            if self.verbose:
                print(f"File fetched from {remote_server}:{remote_path} to {local_path}")
        except Exception as e:
            print(f"Failed to fetch file from remote server: {e}")
            raise
    
    def _push_remote_file(self, remote_server, username, password, local_path, remote_path):
        """
        Push a file to a remote server using SFTP.
        
        Parameters
        ----------
        remote_server : str
            Address of the remote server.
        username : str
            Username for remote server access.
        password : str
            Password for remote server access.
        local_path : str
            Path to the file locally.
        remote_path : str
            Path to save the file on the remote server.
        """
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(remote_server, username=username, password=password)
            sftp = ssh.open_sftp()
            sftp.put(local_path, remote_path)
            sftp.close()
            ssh.close()
            if self.verbose:
                print(f"File pushed from {local_path} to {remote_server}:{remote_path}")
        except Exception as e:
            print(f"Failed to push file to remote server: {e}")
            raise
    
    @staticmethod
    def load(file_path):
        """
        Load an SSN object from a file.
        
        Parameters
        ----------
        file_path : str
            Path to input file.
        """
        # Return the loaded SSN object
        return joblib.load(file_path)

#%% Test usage
import time
from pathlib import Path
import os
if __name__ == "__main__":

    def cd_to_ei2000():
        try:
            start = Path(__file__).resolve()
        except NameError:
            start = Path.cwd()

        repo_dir = None
        # look for a directory named 'model_sandbox' in the path or as a child of any parent
        for p in [start] + list(start.parents):
            if p.name == "model_sandbox":
                repo_dir = p
                break
            candidate = p / "model_sandbox"
            if candidate.is_dir():
                repo_dir = candidate
                break

        if repo_dir is None:
            raise FileNotFoundError("Could not find 'model_sandbox' in current path or its parents.")

        target = repo_dir / "SSN" / "EI2000"
        if not target.is_dir():
            raise FileNotFoundError(f"Target directory not found: {target}")

        os.chdir(str(target))

    cd_to_ei2000()
    
    newSSN = SSN("ei_model")
    newSSN.load_config('config.20250521.flexCRF.json') #newSSN.load_config('./config.driftingGrating_1.json')
    newSSN.add_nodes()
    
    #Visualize nodes in space
    fig, ax = plt.subplots()
    orimap = ax.scatter(newSSN.nodes['x'].values,newSSN.nodes['y'].values,c=newSSN.nodes['orientation'].values,cmap='hsv')
    cbar = plt.colorbar(orimap)
    cbar.ax.set_ylabel("Orientation (rad)")
    ax.set_aspect('equal')
    
    #newSSN.spatial_organization()
    
    newSSN.add_edges()
    
    newSSN.node_types.style
    
    #Visualize network as graph: This takes very long    
    #edge_colors = np.where(newSSN.edges['pre_ei'] == 'e', 'r', 'b')
    #newSSN.visualize_graph(node_color = 'orientation', marker_styles = {'E':'^', 'I':'o'}, edge_color=edge_colors, linewidth=0.0001, cmap='hsv', show_colorbar=True, circular_layout=False)
    
    #load inputs  #TAKES SEVERAL MINUTES!!!
    import getpass
    # username = input("Enter username for remote server scat.cmrr.umn.edu: ")
    # password = getpass.getpass("Enter password: ")
    # newSSN.load_inputs("/home/scat-raid4/share/FlexibleFerrets/L4_drive/driftingGrating/driftingGrating_0.0.npy",
    #                     local=False,
    #                     remote_server="scat.cmrr.umn.edu",
    #                     username=username,
    #                     password=password)
    newSSN.load_inputs("./inputs/flexCRF/FullFieldGrating_orientation=0.0_contrast=1.0.h5",
    local=True)
    
    #%% Run Simulation
    newSSN.connect_inputs()
    
    newSSN.run()
    
    #%% Plot
    Ncells = np.shape(newSSN.outputs.y)[0]
    
    #plot cell responses over time
    fig, ax = plt.subplots()
    ax.plot(np.tile(np.expand_dims(newSSN.outputs.t,axis=0),[Ncells,1]).T,newSSN.outputs.y.T)
    ax.set_ylabel("r")
    ax.set_xlabel("Time (ms)")
    