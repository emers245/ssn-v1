#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 13:06:22 2024

@author: Joe

Design Stim: This is a function repository for code that creates artificial
stimuli for the SSN.

"""

import numpy as np
import matplotlib.pyplot as plt
try:
    from . import SSN_utils
except ImportError:
    from ssn_v1 import SSN_utils
from scipy.special import i0

def create_stimulus_from_config(input_config_path, network_config_path, output_path=None, seed=None):
    """
    Main entry point that reads config and dispatches appropriately
    """
    import json
    
    with open(input_config_path, 'r') as f:
        input_config = json.load(f)
    with open(network_config_path, 'r') as f:
        network_config = json.load(f)

    # set random seed
    if seed is None:
        seed = 123  # Default seed
    
    # Dispatch based on stimulus type
    if config['stimulus_type'] == 'generated':
        stimulus_type = config['stimulus_params']['type']
        
        if stimulus_type == 'grating':
            inputs = generate_grating_stimulus(config)
        elif stimulus_type == 'center_surround_grating':
            inputs = generate_center_surround_grating(config)
        elif stimulus_type == 'plaid':
            inputs = generate_plaid_stimulus(config)
        else:
            raise ValueError(f"Unknown generated stimulus type: {stimulus_type}")
            
    elif config['stimulus_type'] == 'movie':
        inputs = load_movie_stimulus(config)
    else:
        raise ValueError(f"Unknown stimulus_type: {config['stimulus_type']}")
    
    # Save as .npy file
    if output_path is None:
        output_path = config['file_name']
    
    np.save(output_path, inputs)
    return output_path

def generate_grating_stimulus(input_config, network_config, seed=None, tuning_func=SSN_utils.circGauss):
    """
    Generate grating stimulus from config parameters
    
    Expected config structure:
    input_config = {
        "orientation": 45,  # degrees
        "contrast": 0.5,
        "spatial_frequency": 0.1,  # cycles per degree
        "center_x": 0,  # stimulus center in degrees
        "center_y": 0,
        "orientation_channels": [0, 18, 36, 54, 72, 90, 108, 126, 144, 162],
        "mask": {
            "type": "circular",  # "circular", "raisedCosine", "none"
            "radius": 10,  # for circular mask in degrees
            "width": 20, "height": 15  # for rectangular mask
        },
        "temporal": {
            "type": "transient_sustained",  # or "rectangular"
            "params": {
                # For transient_sustained:
                "T": 10000, "t_steps": 50, "m1": 170, "s1": 100,
                "A1": 1, "m2": -200, "s2": 200, "A2": 1.6, "C": 0.8, "t_delay": 0
                # For rectangular:
                # "onset_time": 1000, "offset_time": 8000, "amplitude": 1.0
            }
        }
    }
    network_config = {"extrinsic": {
            "rf_params": {
                "rf_size_mean": 2.0,  # RF radius in degrees
                "rf_size_std": 0.2,  # st. dev. of RF radius across cells in degrees
            },
            "tuning_params": {
                "mu_mean": None,  # mean FF orientation preference (deg) (usually None, overwritten by node ori pref)
                "mu_std": 10,  # st. dev. of FF orientation preference across cells (deg)
                "sigma_mean": 40,  # mean FF orientation tuning width (deg)
                "sigma_std": 26,  # st. dev. of orientation tuning width (deg)
                "scale_mean": 1.0,  # mean magnitude of FF input
                "scale_std": 0.2,  # st. dev. of magnitude of FF input
                "offset_mean": 0.5,  # mean baseline input for orientation tuning curve
                "offset_std": 0.05,  # st. dev. of baseline input for orientation tuning curve
            }
        }

    seed = seed if seed is not None else 123
    tuning_func = tuning function to use for orientation preference
    """
    
    # Set seed
    if seed is None:
        seed = 123

    # Set up spatial geometry
    scaleXY = network_config["nodes"]["spatial_config"]["scaleXY"]
    scaleX = scaleXY[0]
    scaleY = scaleXY[1]
    degrees_per_pixel = network_config["nodes"]["spatial_config"]["deltaXY"]
    NX, NY = (int(scaleX // degrees_per_pixel), int(scaleY // degrees_per_pixel))

    # Create spatial coordinate system
    x_coords, y_coords = create_spatial_coordinates(NX, NY, degrees_per_pixel)
    
    # Apply spatial mask if specified
    if input_config['mask']['type'] != 'none':
        mask = create_spatial_mask(
            x_coords, y_coords,
            mask_params=input_config['mask'],
            center_x=input_config['center_x'],
            center_y=input_config['center_y']
        )
    else:
        mask = None
    
    # Generate temporal profile
    temporal_profile = create_temporal_profile(input_config['temporal'])

    # Generate orientation tuning kernels for each grid position
    tuning_params = network_config['extrinsic']['tuning_params'].copy()
    tuning_params['mu_mean'] = input_config['orientation']  # Align tuning to stimulus
    orientation_kernels = generate_orientation_kernels(
        theta=input_config['orientation_channels'],
        NX=NX, NY=NY, seed=seed, tuning_func=tuning_func,
        **tuning_params
    )
    
    # Generate receptive field masks for each grid position
    sum_mask = np.sum(mask)
    rf_overlaps = compute_rf_overlaps(
        x_coords, y_coords, mask,
        rf_params=network_config['extrinsic']['rf_params'],
        rf_type=network_config['extrinsic']['rf_type'],
        seed=seed
    )
    
    # Combine spatial, temporal, and orientation components
    inputs = combine_stimulus_components(
        mask, temporal_profile, orientation_kernels, rf_overlaps
    )
    
    return inputs

def generate_Nplaid_stimulus(input_config, network_config, seed=None, tuning_func=SSN_utils.circGauss):
    """
    Generate an N-component plaid (sum of N oriented gratings).

    input_config differences from generate_plaid_stimulus:
      - "orientation": list of N orientations (degrees)
      - "contrast": list of N contrasts
      - "spatial_frequency": list of N spatial frequencies

    network_config is assumed to have the same structure as in generate_plaid_stimulus.
    The sum of contrasts must be <= 1.0.
    """

    # Validate contrasts
    contrasts = input_config.get('contrast', [])
    if sum(contrasts) > 1.0:
        print("Warning: Sum of contrasts across N components is greater than 1.0. Normalizing inputs now...")
        contrasts = [ c / sum(contrasts) * 0.999 for c in contrasts ]  # Normalize to sum to 0.999 to avoid exceeding 1 due to rounding
        #raise ValueError("Sum of contrasts across N components is greater than 1.0")

    # Set seed
    if seed is None:
        seed = 123

    # Spatial geometry
    scaleXY = network_config["nodes"]["spatial_config"]["scaleXY"]
    scaleX = scaleXY[0]
    scaleY = scaleXY[1]
    degrees_per_pixel = network_config["nodes"]["spatial_config"]["deltaXY"]
    NX, NY = (int(scaleX // degrees_per_pixel), int(scaleY // degrees_per_pixel))

    # Coordinates and mask
    x_coords, y_coords = create_spatial_coordinates(NX, NY, degrees_per_pixel)
    if input_config['mask']['type'] != 'none':
        mask = create_spatial_mask(
            x_coords, y_coords,
            mask_params=input_config['mask'],
            center_x=input_config['center_x'],
            center_y=input_config['center_y']
        )
    else:
        mask = np.ones((NX, NY))

    # Temporal profile
    temporal_profile = create_temporal_profile(input_config['temporal'])

    orientations = input_config.get('orientation', [])
    orientation_channels = input_config['orientation_channels']

    N = len(orientations)
    if N == 0:
        return np.zeros((NX, NY, len(temporal_profile), len(orientation_channels)))

    # Prepare storage for kernels and rf overlaps
    orientation_kernels_list = []
    rf_overlaps_list = []

    for m in range(N):
        ori = orientations[m]
        tuning_params_m = network_config['extrinsic']['tuning_params'].copy()
        tuning_params_m['mu_mean'] = ori
        kernels_m = generate_orientation_kernels(
            theta=orientation_channels,
            NX=NX, NY=NY, seed=seed, tuning_func=tuning_func,
            **tuning_params_m
        )
        orientation_kernels_list.append(kernels_m)

        rf_m = compute_rf_overlaps(
            x_coords, y_coords, mask,
            rf_params=network_config['extrinsic']['rf_params'],
            rf_type=network_config['extrinsic']['rf_type'],
            seed=seed
        )
        c = contrasts[m] if m < len(contrasts) else 0.0
        rf_m = rf_m * c
        rf_overlaps_list.append(rf_m)

    # Combine components to form total spatial response per orientation
    N_ori = orientation_kernels_list[0].shape[2]
    total_spatial = np.zeros((NX, NY, N_ori))
    for m in range(N):
        total_spatial += rf_overlaps_list[m][:, :, None] * orientation_kernels_list[m]

    # Apply temporal profile to get final inputs: shape (NX, NY, NT, N_ori)
    NT = len(temporal_profile)
    inputs = total_spatial[:, :, None, :] * temporal_profile[None, None, :, None]

    return inputs

def generate_center_surround_grating(input_config, network_config, seed=None, tuning_func=SSN_utils.circGauss):
    """
    Generate center-surround grating stimulus from config parameters
    
    Expected config structure:
    input_config = {
        "center_orientation": 45,  # degrees
        "surround_orientation": 135,  # degrees  
        "center_contrast": 0.8,
        "surround_contrast": 0.5,
        "spatial_frequency": 0.1,  # cycles per degree (currently unused)
        "center_x": 0,  # stimulus center in degrees
        "center_y": 0,
        "orientation_channels": [0, 18, 36, 54, 72, 90, 108, 126, 144, 162],
        "center_mask": {
            "type": "circular",  # "circular", "rectangular", "none"
            "radius": 5.0,  # center radius in degrees
        },
        "surround_mask": {
            "type": "circular",  # typically circular for surround
            "inner_radius": 5.0,  # inner edge of surround (should match center radius)
            "outer_radius": 15.0,  # outer edge of surround in degrees
        },
        "temporal": {
            "type": "transient_sustained",  # or "rectangular"
            "params": {
                "T": 10000, "t_steps": 50, "m1": 170, "s1": 100,
                "A1": 1, "m2": -200, "s2": 200, "A2": 1.6, "C": 0.8, "t_delay": 0
            }
        }
    }
    network_config = {"extrinsic": {
            "rf_params": {
                "rf_size_mean": 2.0,  # RF radius in degrees
                "rf_size_std": 0.2,  # st. dev. of RF radius across cells in degrees
            },
            "tuning_params": {
                "mu_mean": None,  # mean FF orientation preference (deg) (usually None, overwritten by node ori pref)
                "mu_std": 10,  # st. dev. of FF orientation preference across cells (deg)
                "sigma_mean": 40,  # mean FF orientation tuning width (deg)
                "sigma_std": 26,  # st. dev. of orientation tuning width (deg)
                "scale_mean": 1.0,  # mean magnitude of FF input
                "scale_std": 0.2,  # st. dev. of magnitude of FF input
                "offset_mean": 0.5,  # mean baseline input for orientation tuning curve
                "offset_std": 0.05,  # st. dev. of baseline input for orientation tuning curve
            }
        }

    seed = seed if seed is not None else 123
    tuning_func = tuning function to use for orientation preference
    """

    # Set seed
    if seed is None:
        seed = 123
    
    # Set up spatial geometry
    scaleXY = network_config["nodes"]["spatial_config"]["scaleXY"]
    scaleX = scaleXY[0]
    scaleY = scaleXY[1]
    degrees_per_pixel = network_config["nodes"]["spatial_config"]["deltaXY"]
    NX, NY = (int(scaleX // degrees_per_pixel), int(scaleY // degrees_per_pixel))

    # Create spatial coordinate system
    x_coords, y_coords = create_spatial_coordinates(NX, NY, degrees_per_pixel)
    
    # Create center mask
    center_mask = create_center_surround_mask(
        x_coords, y_coords,
        mask_params=input_config['center_mask'],
        mask_type='center',
        center_x=input_config['center_x'],
        center_y=input_config['center_y']
    )
    
    # Create surround mask (annular region)
    surround_mask = create_center_surround_mask(
        x_coords, y_coords,
        mask_params=input_config['surround_mask'],
        mask_type='surround',
        center_x=input_config['center_x'],
        center_y=input_config['center_y']
    )
    
    # Generate temporal profile
    temporal_profile = create_temporal_profile(input_config['temporal'])
    
    # Generate orientation tuning kernels for center orientation
    tuning_params_center = network_config['extrinsic']['tuning_params'].copy()
    tuning_params_center['mu_mean'] = input_config['center_orientation']  # Align tuning to center orientation
    orientation_kernels_ctr = generate_orientation_kernels(
        theta=input_config['orientation_channels'],
        NX=NX, NY=NY, seed=seed, tuning_func=tuning_func,
        **tuning_params_center
    )

    # Generate orientation tuning kernels for surround orientation
    tuning_params_surround = network_config['extrinsic']['tuning_params'].copy()
    tuning_params_surround['mu_mean'] = input_config['surround_orientation']  # Align tuning to surround orientation
    orientation_kernels_sur = generate_orientation_kernels(
        theta=input_config['orientation_channels'],
        NX=NX, NY=NY, seed=seed, tuning_func=tuning_func,
        **tuning_params_surround
    )
    
    # Compute RF overlaps for center and surround separately
    center_rf_overlaps = compute_rf_overlaps(
        x_coords, y_coords, center_mask,
        rf_params=network_config['extrinsic']['rf_params'],
        rf_type=network_config['extrinsic']['rf_type'],
        seed=seed
    )
    
    surround_rf_overlaps = compute_rf_overlaps(
        x_coords, y_coords, surround_mask,
        rf_params=network_config['extrinsic']['rf_params'],
        rf_type=network_config['extrinsic']['rf_type'],
        seed=seed
    )
    
    # Apply contrast scaling to RF overlaps
    center_rf_overlaps = center_rf_overlaps * input_config['center_contrast']
    surround_rf_overlaps = surround_rf_overlaps * input_config['surround_contrast']
    
    # Combine center and surround components with different orientation preferences
    inputs = combine_center_surround_components(
        center_rf_overlaps, surround_rf_overlaps,
        temporal_profile, orientation_kernels_ctr,
        orientation_kernels_sur,
        center_orientation=input_config['center_orientation'],
        surround_orientation=input_config['surround_orientation'],
        orientation_channels=input_config['orientation_channels']
    )
    
    return inputs

def generate_plaid_stimulus(input_config, network_config, seed=None, tuning_func=SSN_utils.circGauss):
    """
    Generate plaid stimulus from config parameters (two overlapping gratings)
    
    Expected config structure:
    input_config = {
        "orientation_1": 45,  # degrees for first grating
        "orientation_2": 135,  # degrees for second grating
        "contrast_1": 0.5,  # contrast for first grating
        "contrast_2": 0.5,  # contrast for second grating
        "spatial_frequency": 0.1,  # cycles per degree (currently unused)
        "center_x": 0,  # stimulus center in degrees
        "center_y": 0,
        "orientation_channels": [0, 18, 36, 54, 72, 90, 108, 126, 144, 162],
        "mask": {
            "type": "circular",  # "circular", "rectangular", "none"
            "radius": 10,  # for circular mask in degrees
            "width": 20, "height": 15  # for rectangular mask
        },
        "temporal": {
            "type": "transient_sustained",  # or "rectangular"
            "params": {
                "T": 10000, "t_steps": 50, "m1": 170, "s1": 100,
                "A1": 1, "m2": -200, "s2": 200, "A2": 1.6, "C": 0.8, "t_delay": 0
            }
        }
    }
    network_config = {"extrinsic": {
            "rf_params": {
                "rf_size_mean": 2.0,  # RF radius in degrees
                "rf_size_std": 0.2,  # st. dev. of RF radius across cells in degrees
            },
            "tuning_params": {
                "mu_mean": None,  # mean FF orientation preference (deg) (usually None, overridden by grating orientations)
                "mu_std": 10,  # st. dev. of FF orientation preference across cells (deg)
                "sigma_mean": 40,  # mean FF orientation tuning width (deg)
                "sigma_std": 26,  # st. dev. of orientation tuning width (deg)
                "scale_mean": 1.0,  # mean magnitude of FF input
                "scale_std": 0.2,  # st. dev. of magnitude of FF input
                "offset_mean": 0.5,  # mean baseline input for orientation tuning curve
                "offset_std": 0.05,  # st. dev. of baseline input for orientation tuning curve
            }
        }
    seed = seed if seed is not None else 123
    """
    # Set seed
    if seed is None:
        seed = 123
    
    # Set up spatial geometry
    scaleXY = network_config["nodes"]["spatial_config"]["scaleXY"]
    scaleX = scaleXY[0]
    scaleY = scaleXY[1]
    degrees_per_pixel = network_config["nodes"]["spatial_config"]["deltaXY"]
    NX, NY = (int(scaleX // degrees_per_pixel), int(scaleY // degrees_per_pixel))

    # Create spatial coordinate system
    x_coords, y_coords = create_spatial_coordinates(NX, NY, degrees_per_pixel)
    
    # Apply spatial mask if specified
    if input_config['mask']['type'] != 'none':
        mask = create_spatial_mask(
            x_coords, y_coords,
            mask_params=input_config['mask'],
            center_x=input_config['center_x'],
            center_y=input_config['center_y']
        )
    else:
        mask = np.ones((NX, NY))
    
    # Generate temporal profile
    temporal_profile = create_temporal_profile(input_config['temporal'])
    
    # Generate orientation tuning kernels for each grating separately
    tuning_params_1 = network_config['extrinsic']['tuning_params'].copy()
    tuning_params_1['mu_mean'] = input_config['orientation_1']  # Align tuning to first grating orientation
    orientation_kernels_1 = generate_orientation_kernels(
        theta=input_config['orientation_channels'],
        NX=NX, NY=NY, seed=seed, tuning_func=tuning_func,
        **tuning_params_1
    )

    tuning_params_2 = network_config['extrinsic']['tuning_params'].copy()
    tuning_params_2['mu_mean'] = input_config['orientation_2']  # Align tuning to second grating orientation
    orientation_kernels_2 = generate_orientation_kernels(
        theta=input_config['orientation_channels'],
        NX=NX, NY=NY, seed=seed, tuning_func=tuning_func,
        **tuning_params_2
    )
    
    # Compute RF overlaps for each grating (same spatial mask for both)
    rf_overlaps_1 = compute_rf_overlaps(
        x_coords, y_coords, mask,
        rf_params=network_config['extrinsic']['rf_params'],
        rf_type=network_config['extrinsic']['rf_type'],
        seed=seed
    )
    
    rf_overlaps_2 = compute_rf_overlaps(
        x_coords, y_coords, mask,
        rf_params=network_config['extrinsic']['rf_params'],
        rf_type=network_config['extrinsic']['rf_type'],
        seed=seed
    )
    
    # Apply contrast scaling to RF overlaps
    rf_overlaps_1 = rf_overlaps_1 * input_config['contrast_1']
    rf_overlaps_2 = rf_overlaps_2 * input_config['contrast_2']
    
    # Combine both grating components
    inputs = combine_plaid_components(
        rf_overlaps_1, rf_overlaps_2,
        temporal_profile, orientation_kernels_1, orientation_kernels_2,
        orientation_1=input_config['orientation_1'],
        orientation_2=input_config['orientation_2'],
        orientation_channels=input_config['orientation_channels']
    )
    
    return inputs

def combine_plaid_components(rf_overlaps_1, rf_overlaps_2, temporal_profile, 
                            orientation_kernels_1, orientation_kernels_2,
                            orientation_1, orientation_2, orientation_channels):
    """
    Combine two grating components with different orientation preferences to create plaid
    """
    NX, NY = rf_overlaps_1.shape
    NT = len(temporal_profile)
    N_ori = orientation_kernels_1.shape[2]
    
    inputs = np.zeros((NX, NY, NT, N_ori))
    
    for i in range(NX):
        for j in range(NY):
            for k in range(N_ori):
                # Each grating has its own orientation tuning from kernels
                grating_1_response = rf_overlaps_1[i, j] * orientation_kernels_1[i, j, k]
                grating_2_response = rf_overlaps_2[i, j] * orientation_kernels_2[i, j, k]
                
                # Combine both gratings (additive)
                total_spatial_response = grating_1_response + grating_2_response
                
                # Apply temporal profile
                inputs[i, j, :, k] = total_spatial_response * temporal_profile
    
    return inputs

def create_center_surround_mask(X, Y, mask_params, mask_type, center_x=0, center_y=0):
    """
    Create center or surround mask
    
    Args:
        X, Y: Coordinate arrays
        mask_params: Mask parameters from config
        mask_type: 'center' or 'surround'
        center_x, center_y: Center position
    """
    distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    
    if mask_type == 'center':
        if mask_params['type'] == 'circular':
            mask = distance <= mask_params['radius']
        elif mask_params['type'] == 'rectangular':
            x_in_bounds = np.abs(X - center_x) <= mask_params['width']/2
            y_in_bounds = np.abs(Y - center_y) <= mask_params['height']/2
            mask = x_in_bounds & y_in_bounds
        else:  # 'none'
            mask = np.ones_like(X, dtype=bool)
    
    elif mask_type == 'surround':
        if mask_params['type'] == 'circular':
            inner_mask = distance > mask_params['inner_radius']
            outer_mask = distance <= mask_params['outer_radius']
            mask = inner_mask & outer_mask
        else:
            raise ValueError(f"Surround mask type '{mask_params['type']}' not supported")
    
    return mask.astype(float)

def combine_center_surround_components(center_rf_overlaps, surround_rf_overlaps, 
                                     temporal_profile, orientation_kernels_ctr,
                                     orientation_kernels_sur,
                                     center_orientation, surround_orientation, 
                                     orientation_channels):
    """
    Combine center and surround components with different orientation preferences
    """
    NX, NY = center_rf_overlaps.shape
    NT = len(temporal_profile)
    N_ori = orientation_kernels_ctr.shape[2]
    
    inputs = np.zeros((NX, NY, NT, N_ori))
    
    for i in range(NX):
        for j in range(NY):
            for k in range(N_ori):
                # Center and surround already have proper orientation tuning from kernels
                center_response = center_rf_overlaps[i, j] * orientation_kernels_ctr[i, j, k]
                surround_response = surround_rf_overlaps[i, j] * orientation_kernels_sur[i, j, k]
                
                # Combine center and surround
                total_spatial_response = center_response + surround_response
                
                # Apply temporal profile
                inputs[i, j, :, k] = total_spatial_response * temporal_profile
    
    return inputs

def load_movie_stimulus(config):
    """Load movie stimulus from file"""
    movie_file = config['stimulus_params']['movie_file']
    movie_data = np.load(movie_file)
    return movie_data

def create_spatial_coordinates(NX, NY, degrees_per_pixel):
    """Create x,y coordinate arrays in degrees"""
    x_extent = NX * degrees_per_pixel
    y_extent = NY * degrees_per_pixel
    
    x = np.linspace(-x_extent/2, x_extent/2, NX)
    y = np.linspace(-y_extent/2, y_extent/2, NY)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    return X, Y

def create_spatial_mask(X, Y, mask_params, center_x=0, center_y=0):
    """Create spatial mask (circular, rectangular, or none)"""
    if mask_params['type'] == 'circular':
        distance = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        mask = distance <= mask_params['radius']
        
    elif mask_params['type'] == 'rectangular':
        x_in_bounds = np.abs(X - center_x) <= mask_params['width']/2
        y_in_bounds = np.abs(Y - center_y) <= mask_params['height']/2
        mask = x_in_bounds & y_in_bounds
        
    else:  # 'none'
        mask = np.ones_like(X, dtype=bool)
    
    return mask.astype(float)

def create_temporal_profile(temporal_params):
    """Create temporal profile based on type"""
    if temporal_params['type'] == 'transient_sustained':
        return transient_sustained(**temporal_params['params'])
        
    elif temporal_params['type'] == 'rectangular':
        params = temporal_params['params']
        T = params.get('T', 10000)
        t_steps = params.get('t_steps', 50)
        onset_time = params['onset_time']
        offset_time = params.get('offset_time', T)
        amplitude = params.get('amplitude', 1.0)
        
        t = np.arange(0, T, t_steps)
        profile = np.zeros_like(t)
        onset_idx = np.argmin(np.abs(t - onset_time))
        offset_idx = np.argmin(np.abs(t - offset_time))
        profile[onset_idx:offset_idx] = amplitude
        
        return profile

    elif temporal_params['type'] == 'ornstein_uhlenbeck':
        return ornstein_uhlenbeck(**temporal_params['params'])
    
    else:
        raise ValueError(f"Unknown temporal profile type: {temporal_params['type']}")

def ornstein_uhlenbeck(T=10000, t_steps=50, theta=0.01, mu=0.0, sigma=0.1, x0=0.0, size=None, seed=None):
    """
    Generate a 1D Ornstein-Uhlenbeck process.

    The discretization used is the exact update for the OU SDE:
        dX = theta * (mu - X) dt + sigma dW

    Inputs:
      T (float): total duration (ms)
      t_steps (float): time step (ms)
      theta (float): mean-reversion rate per ms
      mu (float): long-run mean
      sigma (float): diffusion scale per sqrt(ms)
      x0 (float): initial value
            size (int|tuple|None): leading output shape for parallel OU processes.
                    Time is always the last dimension.
            seed (int|None): random seed for reproducibility

    Outputs:
            x (np.ndarray): OU trajectory array with shape (*size, n_t), where
                    n_t = len(np.arange(0, T, t_steps)). If size is None, shape is (n_t,).
    """

    if T <= 0:
        raise ValueError("T must be > 0")
    if t_steps <= 0:
        raise ValueError("t_steps must be > 0")
    if theta < 0:
        raise ValueError("theta must be >= 0")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")

    if size is None:
        leading_shape = ()
    elif np.isscalar(size):
        leading_shape = (int(size),)
    else:
        leading_shape = tuple(size)
    if any(dim < 0 for dim in leading_shape):
        raise ValueError("All dimensions in size must be >= 0")

    t = np.arange(0, T, t_steps)
    n_steps = len(t)
    x = np.zeros(leading_shape + (n_steps,))
    if n_steps == 0:
        return x

    x[..., 0] = x0
    if n_steps == 1:
        return x

    rng = np.random.default_rng(seed)
    dt = float(t_steps)

    if theta == 0:
        noise_scale = sigma * np.sqrt(dt)
        for i in range(1, n_steps):
            x[..., i] = x[..., i - 1] + noise_scale * rng.normal(size=leading_shape)
        return x

    decay = np.exp(-theta * dt)
    innovation_std = sigma * np.sqrt((1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta))

    for i in range(1, n_steps):
        x[..., i] = mu + (x[..., i - 1] - mu) * decay + innovation_std * rng.normal(size=leading_shape)

    return x

def compute_rf_overlaps(X, Y, mask, rf_params, rf_type, seed=None):
    """
    Compute receptive field overlap with stimulus for each grid position
    
    For each grid position, compute how much of the stimulus falls within
    its receptive field (circular Gaussian)
    """
    NX, NY = X.shape
    rf_overlaps = np.zeros((NX, NY))

    # Set seed
    if seed is None:
        seed = 123
    np.random.seed(seed)  # For reproducibility
    
    # Sample RF sizes for each position
    rf_sizes = np.random.normal(
        rf_params['rf_size_mean'], 
        rf_params['rf_size_std'], 
        size=(NX, NY)
    )
    rf_sizes = np.maximum(rf_sizes, 0.1)  # Ensure positive

    if mask is not None:

        if rf_type == 'gaussian':

            for i in range(NX):
                for j in range(NY):
                    # RF center is at this grid position
                    rf_center_x = X[i, j]
                    rf_center_y = Y[i, j]
                    rf_size = rf_sizes[i, j]
                    
                    # Compute Gaussian RF profile
                    distance = np.sqrt((X - rf_center_x)**2 + (Y - rf_center_y)**2)
                    rf_profile = np.exp(-(distance**2) / (2 * rf_size**2))
                    
                    # Compute overlap with stimulus
                    rf_overlaps[i, j] = np.sum(np.abs(mask) * rf_profile)

            # Normalize by maximum possible overlap
            rf_overlaps = rf_overlaps / (np.max(rf_overlaps) if np.max(rf_overlaps) > 0 else 1)

        elif rf_type == 'circular':

            for i in range(NX):
                for j in range(NY):
                    # RF center is at this grid position
                    rf_center_x = X[i, j]
                    rf_center_y = Y[i, j]
                    rf_size = rf_sizes[i, j]
                    
                    # Create circular RF mask
                    distance = np.sqrt((X - rf_center_x)**2 + (Y - rf_center_y)**2)
                    rf_mask = distance <= rf_size
                    
                    # Compute overlap with stimulus
                    rf_area = np.sum(rf_mask)
                    #print(f"RF area: {rf_area}")
                    rf_overlaps[i, j] = np.sum(np.abs(mask) * rf_mask) / rf_area if rf_area > 0 else 0
                    #print(f"RF overlap at ({i},{j}): {rf_overlaps[i, j]}")
        else:
            raise ValueError(f"Unknown rf_type: {rf_type}")

    else:

        rf_overlaps = np.ones((NX, NY))
    
    return rf_overlaps

def combine_stimulus_components(mask, temporal_profile, orientation_kernels, rf_overlaps):
    """
    Combine spatial grating, temporal profile, orientation tuning, and RF overlaps
    into final 4D input array (X, Y, T, Orientation)
    """
    if mask is None:
        mask = np.ones(rf_overlaps.shape)

    NX, NY = mask.shape
    NT = len(temporal_profile)
    N_ori = orientation_kernels.shape[2]
    
    # Initialize output
    inputs = np.zeros((NX, NY, NT, N_ori))
    
    # For each grid position and orientation
    for i in range(NX):
        for j in range(NY):
            for k in range(N_ori):
                # Spatial response scaled by RF overlap and orientation tuning
                spatial_response = rf_overlaps[i, j] * orientation_kernels[i, j, k]
                
                # Apply temporal profile
                inputs[i, j, :, k] = spatial_response * temporal_profile
    
    return inputs

def transient_sustained(T=10000.,t_steps=50,m1=170.,s1=100.,A1=1.,m2=-200.,s2=200.,A2=1.6,C=0.8,t_delay=0.):
    """
    Creates a temporal input with an initial onset transient and a longer
    sustained response. The response is based on a different of Gaussians.
    
    Inputs:
      T (float): length of stimulus (ms)
      t_steps (int): duration of time step (ms)
      m1 (float): mean of Gaussian 1
      s1 (float): st. dev. of Gaussian 1
      A1 (float): amplitude of Gaussian 1
      m2 (float): mean of Gaussian 2
      s2 (float): st. dev. of Gaussian 2
      A2 (float): amplitude of Gaussian 2
      C (float): offset of DoG
    
    Outputs:
      t_response (arraylike): temporal response
    """
    
    t = np.arange(0,T,t_steps)
    i_delay = np.argmin(np.abs(t-t_delay))
    t_kernel = np.maximum(SSN_utils.DoG(t,m1,m2,s1,s2,A1,A2,C),0)
    t_response = np.zeros(len(t))
    end_t = np.argmin(np.abs(t-(T-t_delay)))
    t_response[i_delay:] = t_kernel[:min(end_t+1, len(t)-i_delay)]
    
    return(t_response)

def generate_orientation_kernels(theta, NX, NY, mu_mean, mu_std, sigma_mean, sigma_std, scale_mean, scale_std, offset_mean, offset_std, seed=None, radians=False, tuning_func=SSN_utils.circGauss):
    """
    Generates orientation kernels using Gaussian distributions for each spatial position.

    Parameters:
    NX (int): Number of spatial positions along the x-axis.
    NY (int): Number of spatial positions along the y-axis.
    theta (array-like): Array of orientation angles (in degrees) to compute Gaussian activations for.
    mu_mean (float): Mean of the distribution used to generate the center (mu) of the Gaussian.
    mu_std (float): Standard deviation of the distribution for mu.
    sigma_mean (float): Mean of the distribution used to generate the standard deviation (sigma) of the Gaussian.
    sigma_std (float): Standard deviation of the distribution for sigma.
    scale_mean (float): Mean of the distribution for the scale factor of the Gaussian.
    scale_std (float): Standard deviation of the distribution for the scale factor.
    offset_mean (float): Mean of the distribution for the offset value of the Gaussian.
    offset_std (float): Standard deviation of the distribution for the offset value.
    seed (int, optional): Seed for random number generation to ensure reproducibility. Default is None.
    radians (bool, optional): If True, inputs are in radians; otherwise in degrees. Default is False.
    tuning_func (callable, optional): Function to compute the tuning curve. Default is SSN_utils.circGauss.
        Must contain the arguments (theta, mu, sigma, period). The period must be a keyword argument named cyc.

    Returns:
    numpy.ndarray: A 3D array of shape (NX, NY, len(theta)) representing orientation kernels at each spatial position.

    Description:
    This function generates a set of orientation kernels for each spatial position (x, y) by sampling the parameters of a Gaussian distribution. 
    The orientation kernels approximate the selectivity of neurons for different stimulus orientations and can be used for modeling responses 
    in visual cortex. For each spatial position, a set of parameters (mu, sigma, scale, offset) is sampled to compute the Gaussian activation 
    for each orientation in `theta`. Parameters are sampled from normal distributions, with resampling for certain conditions (e.g., sigma or 
    scale being negative) to ensure valid values.
    """
    
    # Set random seed for reproducibility
    if not seed:
        np.random.seed(123)  # Ensure that the random kernels are the same for each run
    else:
        np.random.seed(seed)
    
    # Initialize the orientation kernels array
    orientation_kernels = np.zeros((NX, NY, len(theta)))

    # Check for infs
    if sigma_std == 'inf':
        sigma_std = np.inf
    if sigma_mean == 'inf':
        sigma_mean = np.inf
    
    # Loop through each spatial position
    for x in range(NX):
        for y in range(NY):
            # Randomly sample parameters for the Gaussian kernel
            if radians:
                sigma = np.random.normal(sigma_mean, sigma_std)
                mu = np.random.normal(mu_mean, mu_std)
            else:
                sigma = np.radians(np.random.normal(sigma_mean, sigma_std))
                mu = np.radians(np.random.normal(mu_mean, mu_std))
            scale = np.random.normal(scale_mean, scale_std)
            offset = np.random.normal(offset_mean, offset_std)
            
            # Ensure parameters are valid (resample if not)
            mu = np.mod(mu, np.pi)  # Keep mu within [0, pi] range
            while sigma < 0:
                if radians:
                    sigma = np.random.normal(sigma_mean, sigma_std)
                else:
                    sigma = np.radians(np.random.normal(sigma_mean, sigma_std))
            while scale < 0:
                scale = np.random.normal(scale_mean, scale_std)
            while offset < 0:
                offset = np.random.normal(offset_mean, offset_std)
            
            # Compute Gaussian activation for each orientation
            if radians:
                theta_rad = theta
            else:
                theta_rad = np.radians(theta)
            orientation_kernels[x, y, :] = scale * tuning_func(theta_rad, mu, sigma, cyc=np.pi) + offset

    return orientation_kernels

def drive_orientation(NX,NY,ori_kernel_params,t_params):
    """
    Drive a single orientation representation in the network
    

    Inputs:
        NX (int): number of x positions
        NY (int): number of y positions
        ori_kernel_params (dict): a dictionary of parameters for the orientation kernel
            kernelType (function): the type of kernel to use (must be a known function)
            args (list): positional arguments for the kernel. The first argument must be the orientation channels used
            kwargs (dict): key word arguments for the kernel
        t_params (dict): a dictionary of parameters for the temporal kernel
            kernelType (function): the type of kernel to use (must be a known function)
            args (list): positional arguments for the kernel
            kwargs (dict): key word arguments for the kernel

    Outputs:
        inputs (arraylike): array of inputs to network (X,Y,T,Theta)

    """
    
    # Make temporal kernel
    tkernel = t_params['kernelType']
    t_args = t_params['args']
    t_kwargs = t_params['kwargs']
    t_response = tkernel(*t_args,**t_kwargs)
    
    # Make 2D grid of responses
    xy_response = np.tile(t_response, (NX, NY, 1))
    
    # Make orientation kernel
    orikernel = ori_kernel_params['kernelType']
    ori_args = ori_kernel_params['args']
    ori_kwargs = ori_kernel_params['kwargs']
    ori_scaling = orikernel(*ori_args,**ori_kwargs)
    
    # Change shape of ori_scaling if necessary
    if len(np.shape(ori_scaling)) == 1:
        ori_scaling = ori_scaling[np.newaxis, np.newaxis, np.newaxis, :]
    if len(np.shape(ori_scaling)) == 3:
        ori_scaling = ori_scaling[:, :, :, np.newaxis]
        ori_scaling = np.transpose(ori_scaling, (0,1,3,2))
    
    # Apply orientation tuning to input array
    Ntheta = np.shape(ori_scaling)[3]
    inputs = np.tile(xy_response[:, :, :, np.newaxis], (1, 1, 1, Ntheta))
    
    # Scale the 4D array
    inputs = inputs * ori_scaling
    
    return(inputs)

# theta = np.linspace(0,180-180/20,20)
# ori_kernel_params = {"kernelType": generate_orientation_kernels, "args": [theta,2**4,2**4,0,0,40,26,1.0,0.2,0.5,0.05], "kwargs": {"seed": 123}}

# t_params = {"kernelType": transient_sustained, "args": [], "kwargs": {"T":10000., "t_steps":50, "m1": 170., "s1": 100., "A1": 1, "m2": -200., "s2": 200., "A2": 1.6, "C": 0.8, "t_delay": 0.}}

# inputs = drive_orientation(2**4,2**4,ori_kernel_params=ori_kernel_params,t_params=t_params)
    
if __name__ == "__main__":

    # Testing environment

    # An example input config file
    input_config = {
        "file_name": "grating_stimulus.npy",
        "stimulus": {
            "orientation": 45,  # degrees
            "contrast": 0.5,
            "spatial_frequency": 0.1,  # cycles per degree
            "center_x": 0,  # stimulus center in degrees
            "center_y": 0,
            "orientation_channels": [0, 18, 36, 54, 72, 90, 108, 126, 144, 162],
            "mask": {
                "type": "circular",  # "circular", "rectangular", "none"
                "radius": 7.0,  # center radius in degrees
            },
            "stim_type": "current_field",
            "temporal": {
                "type": "transient_sustained",  # or "rectangular"
                "params": {
                    "T": 10000, "t_steps": 50, "m1": 170, "s1": 100,
                    "A1": 1, "m2": -200, "s2": 200, "A2": 1.6, "C": 0.8, "t_delay": 0
                }
            }
        },
        "model": {
            "model_type": "designStim",
        }
    }
    network_config = {
        "nodes": {
            "spatial_config": {
                "scaleXY": {
                    0: 16,
                    1: 16
                },
                "deltaXY": 0.125
            }
        },
        "extrinsic": {
            "rf_params": {
                "rf_size_mean": 2.0,  # RF radius in degrees
                "rf_size_std": 0.0,  # st. dev. of RF radius across cells in degrees
            },
            "rf_type": "circular",
            "tuning_params": {
                "mu_mean": None,  # mean FF orientation preference (deg) (usually None, overwritten by node ori pref)
                "mu_std": 10,  # st. dev. of FF orientation preference across cells (deg)
                "sigma_mean": 40,  # mean FF orientation tuning width (deg)
                "sigma_std": 26,  # st. dev. of orientation tuning width (deg)
                "scale_mean": 1.0,  # mean magnitude of FF input
                "scale_std": 0.2,  # st. dev. of magnitude of FF input
                "offset_mean": 0.5,  # mean baseline input for orientation tuning curve
                "offset_std": 0.05,  # st. dev. of baseline input for orientation tuning curve
            }
        }
    }

    # Test each function for generating a drifting grating stimulus

    # Test create_spatial_coordinates
    # Set up spatial geometry
    scaleXY = network_config["nodes"]["spatial_config"]["scaleXY"]
    scaleX = scaleXY[0]
    scaleY = scaleXY[1]
    degrees_per_pixel = network_config["nodes"]["spatial_config"]["deltaXY"]
    NX, NY = (int(scaleX // degrees_per_pixel), int(scaleY // degrees_per_pixel))

    # Create spatial coordinate system
    x_coords, y_coords = create_spatial_coordinates(NX, NY, degrees_per_pixel)
    print("Spatial coordinates created.")
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(x_coords, cmap='gray')
    ax[0].set_xticks(np.arange(0, x_coords.shape[1], 10), labels=[f'{x:.2f}' for x in x_coords[::10,0]])
    ax[0].set_yticks(np.arange(0, x_coords.shape[0], 10), labels=[f'{y:.2f}' for y in y_coords[0,::10]])
    ax[0].set_title('X Coordinates')
    ax[1].imshow(y_coords, cmap='gray')
    ax[1].set_xticks(np.arange(0, y_coords.shape[1], 10), labels=[f'{x:.2f}' for x in x_coords[::10,0]])
    ax[1].set_yticks(np.arange(0, y_coords.shape[0], 10), labels=[f'{y:.2f}' for y in y_coords[0,::10]])
    ax[1].set_title('Y Coordinates')
    plt.show()

    # Test create_spatial_mask
    # Apply spatial mask if specified
    if input_config['stimulus']['mask']['type'] != 'none':
        mask = create_spatial_mask(
            x_coords, y_coords,
            mask_params=input_config['stimulus']['mask'],
            center_x=input_config['stimulus']['center_x'],
            center_y=input_config['stimulus']['center_y']
        )
    else:
        mask = np.ones((NX, NY))
    print("Spatial mask created.")
    fig, ax = plt.subplots()
    ax.imshow(mask, cmap='gray')
    ax.set_xticks(np.arange(0, x_coords.shape[1], 10), labels=[f'{x:.2f}' for x in x_coords[::10,0]])
    ax.set_yticks(np.arange(0, x_coords.shape[0], 10), labels=[f'{y:.2f}' for y in y_coords[0,::10]])
    ax.set_title('Spatial Mask')
    plt.show()

    # Test create_temporal_profile
    temporal_profile = create_temporal_profile(input_config['stimulus']['temporal'])
    print("Temporal profile created.")
    fig, ax = plt.subplots()
    ax.plot(temporal_profile)
    ax.set_xticks(np.arange(0, len(temporal_profile), 20), labels=[f'{t*input_config["stimulus"]["temporal"]["params"]["t_steps"]:.0f}' for t in np.arange(0, len(temporal_profile), 20)])
    ax.set_title('Temporal Profile')
    plt.show()

    # Test generate_orientation_kernels
    # Generate orientation tuning kernels for each grid position
    tuning_params = network_config['extrinsic']['tuning_params'].copy()
    tuning_params['mu_mean'] = input_config['stimulus']['orientation']  # Align tuning to stimulus
    orientation_kernels = generate_orientation_kernels(
        theta=input_config['stimulus']['orientation_channels'],
        NX=NX, NY=NY,
        **tuning_params
    )
    print("Orientation kernels created.")
    fig, ax = plt.subplots(len(input_config['stimulus']['orientation_channels'])//2, 1)
    for i in range(len(input_config['stimulus']['orientation_channels'])//2):
        ax[i].imshow(orientation_kernels[:, :, 2*i], cmap='gray')
        ax[i].set_xticks(np.arange(0, x_coords.shape[1], 10), labels=[f'{x:.2f}' for x in x_coords[::10,0]])
        ax[i].set_yticks(np.arange(0, x_coords.shape[0], 10), labels=[f'{y:.2f}' for y in y_coords[0,::10]])
        ax[i].set_title(f'Orientation Kernel (orientation {input_config["stimulus"]["orientation_channels"][2*i]}°)')
    plt.show()

    # Test compute_rf_overlaps
    rf_overlaps = compute_rf_overlaps(
        x_coords, y_coords, mask,
        rf_params=network_config['extrinsic']['rf_params'],
        rf_type=network_config['extrinsic']['rf_type']
    )
    print("RF overlaps computed.")
    fig, ax = plt.subplots()
    im = ax.imshow(rf_overlaps, cmap='gray', vmin=0, vmax=1)
    ax.set_xticks(np.arange(0, x_coords.shape[1], 10), labels=[f'{x:.2f}' for x in x_coords[::10,0]])
    ax.set_yticks(np.arange(0, x_coords.shape[0], 10), labels=[f'{y:.2f}' for y in y_coords[0,::10]])
    ax.set_title('RF Overlaps')
    cbar = plt.colorbar(im, ax=ax)
    plt.show()

    # Test combine_stimulus_components
    inputs = combine_stimulus_components(
        mask, temporal_profile, orientation_kernels, rf_overlaps
    )
    print("Stimulus components combined.")
    fig, ax = plt.subplots(len(input_config['stimulus']['orientation_channels'])//2, 1)
    t = 1000
    t_idx = 500 // input_config['stimulus']['temporal']['params']['t_steps']
    for i in range(len(input_config['stimulus']['orientation_channels'])//2):
        im = ax[i].imshow(inputs[:, :, t_idx, 2*i], cmap='gray', vmin=0, vmax=1.5)
        ax[i].set_xticks(np.arange(0, x_coords.shape[1], 10), labels=[f'{x:.2f}' for x in x_coords[::10,0]])
        ax[i].set_yticks(np.arange(0, x_coords.shape[0], 10), labels=[f'{y:.2f}' for y in y_coords[0,::10]])
        ax[i].set_title(f'Combined Stimulus (t={t}, orientation {input_config["stimulus"]["orientation_channels"][2*i]}°)')
        cbar = plt.colorbar(im, ax=ax[i])
    plt.show()
    fig, ax = plt.subplots()
    avg_input = np.mean(np.mean(np.mean(inputs[:, :, :, :], axis=2), axis=1), axis=0)
    ax.plot(input_config['stimulus']['orientation_channels'], avg_input)
    ax.set_xlabel('Orientation (degrees)')
    ax.set_ylabel('Average Input')
    ax.set_title('Average Input vs Orientation')
    plt.show()

    # Test full stimulus generation
    inputs = generate_grating_stimulus(input_config['stimulus'], network_config)
    print("Full grating stimulus generated.")
    fig, ax = plt.subplots(len(input_config['stimulus']['orientation_channels'])//2, 1)
    t = 1000
    t_idx = 500 // input_config['stimulus']['temporal']['params']['t_steps']
    for i in range(len(input_config['stimulus']['orientation_channels'])//2):
        im = ax[i].imshow(inputs[:, :, t_idx, i], cmap='gray', vmin=0, vmax=1.5)
        ax[i].set_xticks(np.arange(0, x_coords.shape[1], 10), labels=[f'{x:.2f}' for x in x_coords[::10,0]])
        ax[i].set_yticks(np.arange(0, x_coords.shape[0], 10), labels=[f'{y:.2f}' for y in y_coords[0,::10]])
        ax[i].set_title(f'Combined Stimulus (t={t}, orientation {input_config["stimulus"]["orientation_channels"][2*i]}°)')
        cbar = plt.colorbar(im, ax=ax[i])
    plt.show()
    fig, ax = plt.subplots()
    avg_input = np.mean(np.mean(np.mean(inputs[:, :, :, :], axis=2), axis=1), axis=0)
    ax.plot(input_config['stimulus']['orientation_channels'], avg_input)
    ax.set_xlabel('Orientation (degrees)')
    ax.set_ylabel('Average Input')
    ax.set_title('Average Input vs Orientation')
    plt.show()

    # Test all functions in generate_grating_stimulus
    input_config = {
        "file_name": "center_surround_stimulus.npy",
        "stimulus": {
            "center_orientation": 45,  # degrees
            "surround_orientation": 135,  # degrees  
            "center_contrast": 0.8,
            "surround_contrast": 0.5,
            "spatial_frequency": 0.1,  # cycles per degree (currently unused)
            "center_x": 0,  # stimulus center in degrees
            "center_y": 0,
            "orientation_channels": [0, 18, 36, 54, 72, 90, 108, 126, 144, 162],
            "center_mask": {
                "type": "circular",  # "circular", "rectangular", "none"
                "radius": 4.0,  # center radius in degrees
            },
            "surround_mask": {
                "type": "circular",  # typically circular for surround
                "inner_radius": 4.0,  # inner edge of surround (should match center radius)
                "outer_radius": 8.0,  # outer edge of surround in degrees
            },
            "temporal": {
                "type": "transient_sustained",  # or "rectangular"
                "params": {
                    "T": 10000, "t_steps": 50, "m1": 170, "s1": 100,
                    "A1": 1, "m2": -200, "s2": 200, "A2": 1.6, "C": 0.8, "t_delay": 0
                }
            }
        },
        "model": {
            "model_type": "designStim",
        }
    }

    # Test create_center_surround_mask
    # Set up spatial geometry
    scaleXY = network_config["nodes"]["spatial_config"]["scaleXY"]
    scaleX = scaleXY[0]
    scaleY = scaleXY[1]
    degrees_per_pixel = network_config["nodes"]["spatial_config"]["deltaXY"]
    NX, NY = (int(scaleX // degrees_per_pixel), int(scaleY // degrees_per_pixel))
    
    # Create spatial coordinate system
    x_coords, y_coords = create_spatial_coordinates(NX, NY, degrees_per_pixel)
    
    # Create center mask
    center_mask = create_center_surround_mask(
        x_coords, y_coords,
        mask_params=input_config['stimulus']['center_mask'],
        mask_type='center',
        center_x=input_config['stimulus']['center_x'],
        center_y=input_config['stimulus']['center_y']
    )
    print("Center mask created.")

    # Create surround mask (annular region)
    surround_mask = create_center_surround_mask(
        x_coords, y_coords,
        mask_params=input_config['stimulus']['surround_mask'],
        mask_type='surround',
        center_x=input_config['stimulus']['center_x'],
        center_y=input_config['stimulus']['center_y']
    )
    print("Surround mask created.")
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(center_mask, cmap='gray')
    ax[0].set_xticks(np.arange(0, x_coords.shape[1], 10), labels=[f'{x:.2f}' for x in x_coords[::10,0]])
    ax[0].set_yticks(np.arange(0, x_coords.shape[0], 10), labels=[f'{y:.2f}' for y in y_coords[0,::10]])
    ax[0].set_title('Center Mask')
    ax[1].imshow(surround_mask, cmap='gray')
    ax[1].set_xticks(np.arange(0, x_coords.shape[1], 10), labels=[f'{x:.2f}' for x in x_coords[::10,0]])
    ax[1].set_yticks(np.arange(0, x_coords.shape[0], 10), labels=[f'{y:.2f}' for y in y_coords[0,::10]])
    ax[1].set_title('Surround Mask')
    plt.show()

    # Test rf overlaps for center and surround

    # Generate temporal profile (reuse existing function)
    temporal_profile = create_temporal_profile(input_config['stimulus']['temporal'])
    
    # Generate orientation tuning kernels for each grid position (reuse existing function)
    tuning_params = network_config['extrinsic']['tuning_params']
    tuning_params['mu_mean'] = input_config['stimulus']['center_orientation']  # Align tuning to center orientation
    orientation_kernels_ctr = generate_orientation_kernels(
        theta=input_config['stimulus']['orientation_channels'],
        NX=NX, NY=NY,
        **tuning_params
    )

    tuning_params['mu_mean'] = input_config['stimulus']['surround_orientation']  # Align tuning to surround orientation
    orientation_kernels_sur = generate_orientation_kernels(
        theta=input_config['stimulus']['orientation_channels'],
        NX=NX, NY=NY,
        **tuning_params
    )

    # Compute RF overlaps for center and surround separately (reuse existing function)
    center_rf_overlaps = compute_rf_overlaps(
        x_coords, y_coords, center_mask,
        rf_params=network_config['extrinsic']['rf_params'],
        rf_type=network_config['extrinsic']['rf_type']
    )
    
    surround_rf_overlaps = compute_rf_overlaps(
        x_coords, y_coords, surround_mask,
        rf_params=network_config['extrinsic']['rf_params'],
        rf_type=network_config['extrinsic']['rf_type']
    )
    print("Center and surround RF overlaps computed.")
    fig, ax = plt.subplots(1,2)
    im0 = ax[0].imshow(center_rf_overlaps, cmap='gray', vmin=0, vmax=1)
    ax[0].set_xticks(np.arange(0, x_coords.shape[1], 10), labels=[f'{x:.2f}' for x in x_coords[::10,0]])
    ax[0].set_yticks(np.arange(0, x_coords.shape[0], 10), labels=[f'{y:.2f}' for y in y_coords[0,::10]])
    ax[0].set_title('Center RF Overlaps')
    cbar0 = plt.colorbar(im0, ax=ax[0])
    im1 = ax[1].imshow(surround_rf_overlaps, cmap='gray', vmin=0, vmax=1)
    ax[1].set_xticks(np.arange(0, x_coords.shape[1], 10), labels=[f'{x:.2f}' for x in x_coords[::10,0]])
    ax[1].set_yticks(np.arange(0, x_coords.shape[0], 10), labels=[f'{y:.2f}' for y in y_coords[0,::10]])
    ax[1].set_title('Surround RF Overlaps')
    cbar1 = plt.colorbar(im1, ax=ax[1])
    plt.show()

    # Test combine_center_surround_components
    center_rf_overlaps = center_rf_overlaps * input_config['stimulus']['center_contrast']
    surround_rf_overlaps = surround_rf_overlaps * input_config['stimulus']['surround_contrast']

    # Combine center and surround components with different orientation preferences
    inputs = combine_center_surround_components(
        center_rf_overlaps, surround_rf_overlaps,
        temporal_profile, orientation_kernels_ctr,
        orientation_kernels_sur,
        center_orientation=input_config['stimulus']['center_orientation'],
        surround_orientation=input_config['stimulus']['surround_orientation'],
        orientation_channels=input_config['stimulus']['orientation_channels']
    )
    print("Center and surround components combined.")
    fig, ax = plt.subplots(len(input_config['stimulus']['orientation_channels'])//2, 1)
    t = 1000
    t_idx = 500 // input_config['stimulus']['temporal']['params']['t_steps']
    for i in range(len(input_config['stimulus']['orientation_channels'])//2):
        im = ax[i].imshow(inputs[:, :, t_idx, 2*i], cmap='gray', vmin=0, vmax=1.5)
        ax[i].set_xticks(np.arange(0, x_coords.shape[1], 10), labels=[f'{x:.2f}' for x in x_coords[::10,0]])
        ax[i].set_yticks(np.arange(0, x_coords.shape[0], 10), labels=[f'{y:.2f}' for y in y_coords[0,::10]])
        ax[i].set_title(f'Combined Stimulus (t={t}, orientation {input_config["stimulus"]["orientation_channels"][2*i]}°)')
        cbar = plt.colorbar(im, ax=ax[i])
    plt.show()
    fig, ax = plt.subplots()
    # Create masked arrays where mask values of 0 are masked out
    # Expand center_mask to match inputs dimensions
    center_mask_expanded = center_mask[:,:,np.newaxis,np.newaxis]  # Shape: (1, 1, H, W)
    center_mask_expanded = np.broadcast_to(center_mask_expanded, inputs.shape)  # Shape: (H, W, T, N_ori)
    surround_mask_expanded = surround_mask[:,:,np.newaxis,np.newaxis]  # Shape: (1, 1, H, W)
    surround_mask_expanded = np.broadcast_to(surround_mask_expanded, inputs.shape)  # Shape: (H, W, T, N_ori)
    # Average only over non-masked (non-zero mask) values
    avg_input_ctr = [np.mean(inputs[:,:,:,o_i][center_mask_expanded[:,:,:,o_i] != 0]) for o_i in range(len(input_config['stimulus']['orientation_channels']))]
    avg_input_sur = [np.mean(inputs[:,:,:,o_i][surround_mask_expanded[:,:,:,o_i] != 0]) for o_i in range(len(input_config['stimulus']['orientation_channels']))]
    ax.plot(input_config['stimulus']['orientation_channels'], avg_input_ctr, label='Center')
    ax.plot(input_config['stimulus']['orientation_channels'], avg_input_sur, label='Surround')
    ax.set_xlabel('Orientation (degrees)')
    ax.set_ylabel('Average Input')
    ax.set_title('Average Input vs Orientation')
    ax.legend()
    plt.show()

    # Test full center-surround stimulus generation
    inputs = generate_center_surround_grating(input_config['stimulus'], network_config)
    print("Full center-surround stimulus generated.")
    fig, ax = plt.subplots(len(input_config['stimulus']['orientation_channels'])//2, 1)
    t = 1000
    t_idx = 500 // input_config['stimulus']['temporal']['params']['t_steps']
    for i in range(len(input_config['stimulus']['orientation_channels'])//2):
        im = ax[i].imshow(inputs[:, :, t_idx, 2*i], cmap='gray', vmin=0, vmax=1.5)
        ax[i].set_xticks(np.arange(0, x_coords.shape[1], 10), labels=[f'{x:.2f}' for x in x_coords[::10,0]])
        ax[i].set_yticks(np.arange(0, x_coords.shape[0], 10), labels=[f'{y:.2f}' for y in y_coords[0,::10]])
        ax[i].set_title(f'Combined Stimulus (t={t}, orientation {input_config["stimulus"]["orientation_channels"][2*i]}°)')
        cbar = plt.colorbar(im, ax=ax[i])
    plt.show()
    fig, ax = plt.subplots()
    # Average only over non-masked (non-zero mask) values
    avg_input_ctr = [np.mean(inputs[:,:,:,o_i][center_mask_expanded[:,:,:,o_i] != 0]) for o_i in range(len(input_config['stimulus']['orientation_channels']))]
    avg_input_sur = [np.mean(inputs[:,:,:,o_i][surround_mask_expanded[:,:,:,o_i] != 0]) for o_i in range(len(input_config['stimulus']['orientation_channels']))]
    ax.plot(input_config['stimulus']['orientation_channels'], avg_input_ctr, label='Center')
    ax.plot(input_config['stimulus']['orientation_channels'], avg_input_sur, label='Surround')
    ax.set_xlabel('Orientation (degrees)')
    ax.set_ylabel('Average Input')
    ax.set_title('Average Input vs Orientation')
    ax.legend()
    plt.show()

    # Test plaid stimulus generation
    input_config = {
        "file_name": "plaid_stimulus.npy",
        "stimulus": {
            "orientation_1": 45,  # degrees for first grating
            "orientation_2": 135,  # degrees for second grating
            "contrast_1": 0.6,  # contrast for first grating
            "contrast_2": 0.4,  # contrast for second grating
            "spatial_frequency": 0.1,  # cycles per degree (currently unused)
            "center_x": 0,  # stimulus center in degrees
            "center_y": 0,
            "orientation_channels": [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5],
            "mask": {
                "type": "circular",  # "circular", "rectangular", "none"
                "radius": 6.0,  # for circular mask in degrees
                "width": 20, "height": 15  # for rectangular mask
            },
            "temporal": {
                "type": "transient_sustained",  # or "rectangular"
                "params": {
                    "T": 10000, "t_steps": 50, "m1": 170, "s1": 100,
                    "A1": 1, "m2": -200, "s2": 200, "A2": 1.6, "C": 0.8, "t_delay": 0
                }
            }
        },
        "model": {
            "model_type": "designStim",
        }
    }
    network_config['extrinsic']['tuning_params']['sigma_mean'] = 20
    network_config['extrinsic']['tuning_params']['sigma_std'] = 0
    network_config['extrinsic']['tuning_params']['scale_mean'] = 1.0
    network_config['extrinsic']['tuning_params']['scale_std'] = 0.2
    network_config['extrinsic']['tuning_params']['offset_mean'] = 0
    network_config['extrinsic']['tuning_params']['offset_std'] = 0 

    # Test full plaid stimulus generation
    inputs = generate_plaid_stimulus(input_config['stimulus'], network_config)
    print("Full plaid stimulus generated.")
    
    # Visualize plaid stimulus
    fig, ax = plt.subplots(len(input_config['stimulus']['orientation_channels'])//2, 1, figsize=(8, 12))
    t = 1000
    t_idx = 500 // input_config['stimulus']['temporal']['params']['t_steps']
    for i in range(len(input_config['stimulus']['orientation_channels'])//2):
        im = ax[i].imshow(inputs[:, :, t_idx, 2*i], cmap='gray', vmin=0, vmax=2)
        ax[i].set_xticks(np.arange(0, x_coords.shape[1], 10), labels=[f'{x:.2f}' for x in x_coords[::10,0]])
        ax[i].set_yticks(np.arange(0, x_coords.shape[0], 10), labels=[f'{y:.2f}' for y in y_coords[0,::10]])
        ax[i].set_title(f'Plaid Stimulus (t={t}, orientation {input_config["stimulus"]["orientation_channels"][2*i]}°)')
        cbar = plt.colorbar(im, ax=ax[i])
    plt.tight_layout()
    plt.show()
    
    # Plot average input vs orientation for plaid
    fig, ax = plt.subplots()
    avg_input_plaid = np.mean(np.mean(np.mean(inputs[:, :, :, :], axis=2), axis=1), axis=0)
    ax.plot(input_config['stimulus']['orientation_channels'], avg_input_plaid, 'o-', label='Plaid')
    ax.axvline(x=input_config['stimulus']['orientation_1'], color='red', linestyle='--', alpha=0.7, label=f'Grating 1 ({input_config["stimulus"]["orientation_1"]}°)')
    ax.axvline(x=input_config['stimulus']['orientation_2'], color='blue', linestyle='--', alpha=0.7, label=f'Grating 2 ({input_config["stimulus"]["orientation_2"]}°)')
    ax.set_xlabel('Orientation (degrees)')
    ax.set_ylabel('Average Input')
    ax.set_title('Average Input vs Orientation (Plaid Stimulus)')
    ax.legend()
    plt.show()

    avg_input_dominant = avg_input_plaid[np.argmin(np.abs(np.array(input_config['stimulus']['orientation_channels']) - input_config['stimulus']['orientation_1']))]
    avg_input_non_dominant = avg_input_plaid[np.argmin(np.abs(np.array(input_config['stimulus']['orientation_channels']) - input_config['stimulus']['orientation_2']))]
    print(f"Average input at dominant orientation ({input_config['stimulus']['orientation_1']}°): {avg_input_dominant:.4f}")
    print(f"Average input at non-dominant orientation ({input_config['stimulus']['orientation_2']}°): {avg_input_non_dominant:.4f}")
    print(f"Input ratio (dominant/non-dominant): {avg_input_dominant/avg_input_non_dominant:.4f}")
    print(f"Actual Contrast Ratio (dominant/non-dominant): {input_config['stimulus']['contrast_1']/input_config['stimulus']['contrast_2']:.4f}")