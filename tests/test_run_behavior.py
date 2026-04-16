#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Behavior tests for SSN.run ODE/SDE branching."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add SSN directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from SSN import SSN
except ImportError:
    from ..SSN import SSN


def _build_minimal_ssn(n_cells: int = 4, tstop: float = 1.0, dt: float = 0.1):
    """Create a minimal SSN instance with enough state to run()."""
    ssn = SSN("test_run", rand_seed=123, verbose=False)

    # Minimal node tables expected by run()
    ssn.nodes = pd.DataFrame(
        {
            "model_name": ["E"] * n_cells,
            "model_index": [0] * n_cells,
        }
    )
    ssn.node_types = pd.DataFrame(
        {
            "model_type_id": [0],
            "tau": [10.0],
            "k": [0.01],
            "c": [1.0],
            "n": [2.0],
        }
    )

    # Minimal recurrent matrix via construct_W()
    W = np.eye(n_cells) * 0.05
    ssn.construct_W = lambda: W

    # Minimal input current h(t)
    t = np.arange(0, tstop, dt)
    ssn.h = np.zeros((n_cells, len(t)), dtype=float)

    # Provide stimulus timing so CubicSpline axis lengths match exactly
    ssn.stim_params = {
        "temporal": {
            "params": {
                "T": tstop,
                "t_steps": dt,
            }
        }
    }

    # Minimal run params
    ssn.run_params = {
        "tstop": tstop,
        "dt": dt,
        "r_init": {"type": "uniform", "E": 0.0},
        "method": "RK45",
    }

    return ssn


def test_run_uses_solve_ivp_when_noise_matrix_absent():
    """Without noise_matrix, run() should follow ODE path and return solve_ivp-style output."""
    ssn = _build_minimal_ssn()
    ssn.run()

    assert hasattr(ssn, "outputs")
    assert hasattr(ssn.outputs, "t")
    assert hasattr(ssn.outputs, "y")
    assert ssn.outputs.y.shape[0] == len(ssn.nodes)
    assert ssn.outputs.y.shape[1] == len(ssn.outputs.t)
    assert bool(ssn.outputs.success)


def test_run_noise_matrix_invalid_method_raises_value_error():
    """With noise_matrix present, unsupported torchsde methods should raise clearly."""
    ssn = _build_minimal_ssn()
    n = len(ssn.nodes)
    ssn.run_params["noise_matrix"] = 0.1 * np.eye(n)
    ssn.run_params["method"] = "RK45"  # valid for solve_ivp, not torchsde

    with pytest.raises(ValueError, match="supported torchsde method"):
        ssn.run()


@pytest.mark.integration
def test_run_uses_torchsde_when_noise_matrix_present():
    """With noise_matrix and supported method, run() should execute SDE branch."""
    pytest.importorskip("torch")
    pytest.importorskip("torchsde")

    ssn = _build_minimal_ssn()
    n = len(ssn.nodes)
    ssn.run_params["noise_matrix"] = 0.05 * np.eye(n)
    ssn.run_params["method"] = "euler"

    ssn.run()

    assert hasattr(ssn, "outputs")
    assert hasattr(ssn.outputs, "t")
    assert hasattr(ssn.outputs, "y")
    assert ssn.outputs.y.shape[0] == n
    assert ssn.outputs.y.shape[1] == len(ssn.outputs.t)
    assert bool(ssn.outputs.success)
    assert ssn.outputs.get("solver", None) == "torchsde"
