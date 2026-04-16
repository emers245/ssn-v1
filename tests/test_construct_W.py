#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for SSN.construct_W — verifies vectorized implementation against
row-by-row reference."""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add SSN directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from SSN import SSN


def _build_ssn_with_edges(n_cells, density=0.3, seed=42):
    """Create a minimal SSN with a random edges DataFrame."""
    rng = np.random.default_rng(seed)

    ssn = SSN("test_construct_W", rand_seed=seed, verbose=False)
    ssn.nodes = pd.DataFrame({
        "node_index": np.arange(n_cells),
        "model_name": ["E"] * (n_cells // 2) + ["I"] * (n_cells - n_cells // 2),
        "ei": ["e"] * (n_cells // 2) + ["i"] * (n_cells - n_cells // 2),
    })

    # Generate random sparse connectivity
    n_edges = int(n_cells * n_cells * density)
    pre = rng.integers(0, n_cells, size=n_edges)
    post = rng.integers(0, n_cells, size=n_edges)
    weights = rng.standard_normal(n_edges)

    edge_list = []
    for i in range(n_edges):
        edge_list.append({
            "edge_index": i,
            "weight": weights[i],
            "pre_model_index": int(pre[i]),
            "post_model_index": int(post[i]),
            "pre_model_name": ssn.nodes["model_name"].iloc[pre[i]],
            "post_model_name": ssn.nodes["model_name"].iloc[post[i]],
            "pre_ei": ssn.nodes["ei"].iloc[pre[i]],
            "post_ei": ssn.nodes["ei"].iloc[post[i]],
        })
    ssn.edges = pd.DataFrame(edge_list)
    return ssn


def _construct_W_rowwise(ssn):
    """Reference row-by-row implementation (the original iterrows version)."""
    N = len(ssn.nodes)
    W = np.zeros((N, N))
    for _, row in ssn.edges.iterrows():
        pre_index = row['pre_model_index']
        post_index = row['post_model_index']
        weight = row['weight']
        W[post_index, pre_index] = weight
    return W


# ---- Tests ----

@pytest.mark.parametrize("n_cells,density", [
    (4, 1.0),
    (16, 0.5),
    (64, 0.3),
    (128, 0.2),
    (512, 0.1),
])
def test_construct_W_matches_rowwise(n_cells, density):
    """Vectorized construct_W must produce the same matrix as the row-by-row loop."""
    ssn = _build_ssn_with_edges(n_cells, density=density)
    W_vectorized = ssn.construct_W()
    W_reference = _construct_W_rowwise(ssn)
    np.testing.assert_array_equal(
        W_vectorized, W_reference,
        err_msg=f"Mismatch for n_cells={n_cells}, density={density}",
    )


def test_construct_W_empty_edges():
    """construct_W should return a zero matrix when there are no edges."""
    ssn = SSN("test_empty", rand_seed=0, verbose=False)
    n_cells = 8
    ssn.nodes = pd.DataFrame({
        "node_index": np.arange(n_cells),
        "model_name": ["E"] * n_cells,
        "ei": ["e"] * n_cells,
    })
    ssn.edges = pd.DataFrame({
        "edge_index": pd.Series([], dtype=int),
        "weight": pd.Series([], dtype=float),
        "pre_model_index": pd.Series([], dtype=int),
        "post_model_index": pd.Series([], dtype=int),
        "pre_model_name": pd.Series([], dtype=str),
        "post_model_name": pd.Series([], dtype=str),
        "pre_ei": pd.Series([], dtype=str),
        "post_ei": pd.Series([], dtype=str),
    })
    W = ssn.construct_W()
    assert W.shape == (n_cells, n_cells)
    assert np.all(W == 0.0)


def test_construct_W_duplicate_edges_last_wins():
    """When duplicate (post, pre) pairs exist, the last edge's weight should win
    (same semantics as the original iterrows loop)."""
    ssn = SSN("test_dup", rand_seed=0, verbose=False)
    n_cells = 4
    ssn.nodes = pd.DataFrame({
        "node_index": np.arange(n_cells),
        "model_name": ["E"] * n_cells,
        "ei": ["e"] * n_cells,
    })
    # Two edges to the same (post=1, pre=0) slot with different weights
    ssn.edges = pd.DataFrame({
        "edge_index": [0, 1],
        "weight": [1.0, 5.0],
        "pre_model_index": [0, 0],
        "post_model_index": [1, 1],
        "pre_model_name": ["E", "E"],
        "post_model_name": ["E", "E"],
        "pre_ei": ["e", "e"],
        "post_ei": ["e", "e"],
    })
    W = ssn.construct_W()
    W_ref = _construct_W_rowwise(ssn)
    np.testing.assert_array_equal(W, W_ref)
    # Both methods should end up with the last weight (5.0)
    assert W[1, 0] == 5.0


def test_construct_W_self_connections():
    """Self-connections (diagonal entries) should be handled correctly."""
    ssn = _build_ssn_with_edges(16, density=0.5, seed=99)
    # Add explicit self-connections
    self_edges = pd.DataFrame({
        "edge_index": range(16),
        "weight": np.ones(16) * 0.1,
        "pre_model_index": np.arange(16),
        "post_model_index": np.arange(16),
        "pre_model_name": ssn.nodes["model_name"].values,
        "post_model_name": ssn.nodes["model_name"].values,
        "pre_ei": ssn.nodes["ei"].values,
        "post_ei": ssn.nodes["ei"].values,
    })
    ssn.edges = pd.concat([ssn.edges, self_edges], ignore_index=True)

    W = ssn.construct_W()
    W_ref = _construct_W_rowwise(ssn)
    np.testing.assert_array_equal(W, W_ref)
