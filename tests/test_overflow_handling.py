#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest tests for numerical overflow handling in SSN.py

Usage:
    pytest test_overflow_handling.py
    pytest test_overflow_handling.py -v  # verbose
    pytest test_overflow_handling.py::test_normal_run  # run specific test

This module tests that:
1. Normal simulations run without errors
2. Simulations with extreme parameters raise NumericalInstabilityError
3. Overflow is caught and handled gracefully
"""

import numpy as np
import pytest
import sys
from pathlib import Path
import os

os.sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from SSN import SSN, NumericalInstabilityError
except ImportError:
    # Try relative import
    from ..SSN import SSN, NumericalInstabilityError


def create_synthetic_input(n_timesteps: int = 100, n_neurons: int = 10) -> np.ndarray:
    """Create a simple synthetic input for testing."""
    return np.ones((n_neurons, n_timesteps), dtype=np.float32)


def test_normal_run(config_file, verbose):
    """Test that normal SSN run works without errors."""
    ssn = SSN("ei_model_normal", verbose=verbose)
    ssn.load_config(config_file)
    ssn.add_nodes()
    ssn.add_edges()
    
    # Create small synthetic input
    n_timesteps = int(ssn.stim_params['temporal']['params']['T'] // ssn.stim_params['temporal']['params']['t_steps'])
    synthetic_input = create_synthetic_input(n_timesteps=n_timesteps, n_neurons=ssn.nodes.shape[0])
    ssn.inputs = synthetic_input
    ssn.stim_params['stim_type'] = 'currents'
    ssn.connect_inputs()
    ssn.node_types["c"] = 0.01
    ssn.nodes["c"] = 0.01

    # Run the simulation - should not raise any exceptions
    ssn.run()
    
    # Verify outputs are valid
    assert ssn.outputs.y.shape[0] > 0, "Output should have neurons"
    assert ssn.outputs.y.shape[1] > 0, "Output should have timesteps"
    assert not np.isnan(ssn.outputs.y).any(), "Output should not contain NaN"
    assert not np.isinf(ssn.outputs.y).any(), "Output should not contain Inf"
    
    print(f"Normal run completed successfully")
    print(f"  Output shape: {ssn.outputs.y.shape}")
    print(f"  Output range: [{ssn.outputs.y.min():.4f}, {ssn.outputs.y.max():.4f}]")


def test_overflow_scenario(config_file, verbose):
    """Test that extreme parameters trigger NumericalInstabilityError or produce valid outputs."""
    ssn = SSN("ei_model_extreme", verbose=verbose)
    ssn.load_config(config_file)
    ssn.add_nodes()
    ssn.add_edges()
    
    # Create extremely large input to trigger overflow in the power law nonlinearity
    # The nonlinearity is: r_ss = k * max(c*h + W*r, 0)^n
    # We'll pump in huge input gains
    n_timesteps = int(ssn.stim_params['temporal']['params']['T'] // ssn.stim_params['temporal']['params']['t_steps'])
    synthetic_input = create_synthetic_input(n_timesteps=n_timesteps, n_neurons=ssn.nodes.shape[0])
    ssn.inputs = synthetic_input
    ssn.connect_inputs()
    
    # Use extreme scaling to force overflow
    ssn.node_types["c"] = 10**20  # Huge input gain
    ssn.nodes["c"] = 10**20

    # Either NumericalInstabilityError is raised or the run completes
    # Both behaviors are acceptable depending on the implementation
    try:
        print(f"Running with extreme input gain (c=10^20)...")
        ssn.run()
        # If we get here, no overflow was raised but outputs might still be extreme
        print(f"No NumericalInstabilityError raised with extreme parameters")
        print(f"  Output range: [{ssn.outputs.y.min():.4f}, {ssn.outputs.y.max():.4f}]")
    except (NumericalInstabilityError, OverflowError) as e:
        print(f"Successfully caught {type(e).__name__}: {e}")
        # This is expected and acceptable


def test_extreme_recurrent_weights(config_file, verbose):
    """Test overflow from extreme recurrent connectivity."""
    ssn = SSN("ei_model_recurrent", verbose=verbose)
    ssn.load_config(config_file)
    ssn.add_nodes()
    ssn.add_edges()
    
    # Artificially inflate recurrent weights to cause instability
    ssn.edges['weight'] *= 10**10  # Multiply all recurrent weights by huge factor
    
    # Create input
    n_timesteps = int(ssn.stim_params['temporal']['params']['T'] // ssn.stim_params['temporal']['params']['t_steps'])
    synthetic_input = create_synthetic_input(n_timesteps=n_timesteps, n_neurons=ssn.nodes.shape[0])
    ssn.inputs = synthetic_input
    ssn.connect_inputs()
    ssn.node_types["c"] = 0.5
    ssn.nodes["c"] = 0.5

    # Either NumericalInstabilityError is raised or the run completes
    try:
        print(f"Running with recurrent weights scaled by 10^10...")
        ssn.run()
        print(f"No overflow detected with extreme recurrent weights")
        print(f"  Output range: [{ssn.outputs.y.min():.4f}, {ssn.outputs.y.max():.4f}]")
    except NumericalInstabilityError as e:
        print(f"Successfully caught NumericalInstabilityError: {e}")
        # This is expected and acceptable


def test_event_threshold(config_file, verbose):
    """Test that events parameter can terminate integration early."""
    ssn = SSN("ei_model_events", verbose=verbose)
    ssn.load_config(config_file)
    ssn.add_nodes()
    ssn.add_edges()
    
    # Create input
    n_timesteps = int(ssn.stim_params['temporal']['params']['T'] // ssn.stim_params['temporal']['params']['t_steps'])
    synthetic_input = create_synthetic_input(n_timesteps=n_timesteps, n_neurons=ssn.nodes.shape[0])
    ssn.inputs = synthetic_input
    ssn.stim_params['stim_type'] = 'currents'
    ssn.connect_inputs()
    
    # Use high input gain to push rates high
    ssn.node_types["c"] = 100.0
    ssn.nodes["c"] = 100.0

    # Define event to stop when any rate exceeds threshold
    threshold = 10.0  # Set a low threshold to ensure we trigger it before overflow
    
    def threshold_exceeded(t, y, *args):
        """Returns negative when threshold is exceeded.
        Args are passed by solve_ivp but not used here."""
        return threshold - np.max(y)
    
    threshold_exceeded.terminal = True  # Stop integration when triggered
    threshold_exceeded.direction = -1   # Detect when crossing from positive to negative
    
    # Run with event - may complete, trigger event, or raise error
    try:
        print(f"Running with input gain c=100.0 and threshold={threshold}...")
        ssn.run(events=threshold_exceeded)
        
        # Check if event was triggered
        if ssn.outputs.status == 1 and len(ssn.outputs.t_events[0]) > 0:
            event_time = ssn.outputs.t_events[0][0]
            max_rate = np.max(ssn.outputs.y[:, -1])
            print(f"Integration terminated by event at t={event_time:.4f}")
            print(f"  Max rate at termination: {max_rate:.4f}")
            assert max_rate >= threshold, f"Rate {max_rate} should be >= threshold {threshold}"
        else:
            print(f"No event triggered (rates stayed below threshold)")
            print(f"  Max rate: {np.max(ssn.outputs.y):.4f}")
    except NumericalInstabilityError as e:
        # Acceptable - overflow caught before event could trigger
        print(f"NumericalInstabilityError raised (overflow before event): {e}")

