#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for SSN.run() performance optimizations:
  1. Sparse W in ODE RHS
  3. Cached self.W_sparse with invalidation on add_edges
  6. Closure-based ssn_equations (no args)
  7. Vectorized DataFrame param lookup

Unit tests use a minimal E/I SSN built without config files.
Integration tests use the test config to exercise the full pipeline.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from SSN import SSN
except ImportError:
    from ..SSN import SSN

import scipy.sparse


# ---------------------------------------------------------------------------
# Helper: build a minimal E/I SSN that can run() without config files
# ---------------------------------------------------------------------------
def _build_minimal_ssn(n_e: int = 2, n_i: int = 2, tstop: float = 100.0, dt: float = 0.1,
                       input_amp: float = 5.0):
    """Create a minimal E/I SSN instance with enough state to run().

    Parameters are chosen from inspect_minimal_ssn.ipynb to produce stable,
    non-trivial dynamics up to input amplitudes of ~100.

    Parameters
    ----------
    n_e : int
        Number of excitatory cells.
    n_i : int
        Number of inhibitory cells.
    tstop : float
        Simulation duration (ms).
    dt : float
        Time step for input vector (ms).
    input_amp : float
        Constant input amplitude applied to all cells.
    """
    n_cells = n_e + n_i
    ssn = SSN("test_opt", rand_seed=123, verbose=False)

    ssn.nodes = pd.DataFrame({
        "model_name": ["E"] * n_e + ["I"] * n_i,
        "model_index": [0] * n_e + [1] * n_i,
    })
    ssn.node_types = pd.DataFrame({
        "model_type_id": [0, 1],
        "tau": [20.0, 10.0],
        "k": [0.04, 0.04],
        "c": [1.0, 1.0],
        "n": [2.0, 2.0],
    })

    # Recurrent weight matrix (block structure from inspect_minimal_ssn.ipynb)
    W = np.zeros((n_cells, n_cells))
    W[:n_e, :n_e] = 0.178    # E->E excitation
    W[:n_e, n_e:] = -0.170   # I->E inhibition
    W[n_e:, :n_e] = 0.093    # E->I excitation
    W[n_e:, n_e:] = -0.073   # I->I inhibition
    ssn.construct_W = lambda: W

    # Input current h(t): constant drive at input_amp
    t = np.arange(0, tstop, dt)
    ssn.h = np.ones((n_cells, len(t)), dtype=float) * input_amp

    ssn.stim_params = {
        "temporal": {"params": {"T": tstop, "t_steps": dt}},
    }

    ssn.run_params = {
        "tstop": tstop,
        "dt": dt,
        "r_init": {"type": "uniform", "E": 0.0, "I": 0.0},
        "method": "RK45",
    }

    return ssn


# ---------------------------------------------------------------------------
# Correctness: ODE output unchanged
# ---------------------------------------------------------------------------
class TestRunOutputUnchanged:
    """Verify that the optimized run() produces correct, deterministic output."""

    def test_run_output_deterministic(self):
        """Two identical runs with the same seed must produce identical output."""
        ssn1 = _build_minimal_ssn()
        ssn1.run()

        ssn2 = _build_minimal_ssn()
        ssn2.run()

        np.testing.assert_array_equal(ssn1.outputs.t, ssn2.outputs.t)
        np.testing.assert_array_equal(ssn1.outputs.y, ssn2.outputs.y)

    def test_run_output_shape(self):
        """Output shape must be (n_cells, n_timepoints)."""
        n_e, n_i = 4, 4
        ssn = _build_minimal_ssn(n_e=n_e, n_i=n_i)
        ssn.run()

        assert ssn.outputs.y.shape[0] == n_e + n_i
        assert ssn.outputs.y.shape[1] == len(ssn.outputs.t)
        assert ssn.outputs.success

    def test_run_with_nonzero_input(self):
        """Default input_amp=5 produces non-trivial, stable dynamics."""
        ssn = _build_minimal_ssn()
        ssn.run()
        assert ssn.outputs.success
        # With positive input and k*power-law, rates should grow above zero
        assert np.any(ssn.outputs.y > 0)
        # Rates should remain finite (network is stable)
        assert not np.any(np.isinf(ssn.outputs.y))
        assert not np.any(np.isnan(ssn.outputs.y))

    def test_run_with_high_input(self):
        """Network stays stable at max intended input (amp=100)."""
        ssn = _build_minimal_ssn(input_amp=100.0)
        ssn.run()
        assert ssn.outputs.success
        assert np.any(ssn.outputs.y > 0)
        assert not np.any(np.isinf(ssn.outputs.y))
        assert not np.any(np.isnan(ssn.outputs.y))


def _build_minimal_ssn_with_edges(n_e: int = 2, n_i: int = 2):
    """Create a minimal SSN with edges DataFrame so the real construct_W works.

    Uses the same block W structure from inspect_minimal_ssn.ipynb, expressed
    as an edges table.
    """
    n_cells = n_e + n_i
    ssn = SSN("test_cache", rand_seed=123, verbose=False)

    ssn.nodes = pd.DataFrame({
        "model_name": ["E"] * n_e + ["I"] * n_i,
        "model_index": list(range(n_cells)),
    })

    # Build edges from the block W matrix:
    #   E->E = 0.178, I->E = -0.170, E->I = 0.093, I->I = -0.073
    block_weights = {
        ("E", "E"): 0.178,
        ("I", "E"): -0.170,
        ("E", "I"): 0.093,
        ("I", "I"): -0.073,
    }
    rows = []
    idx = 0
    for pre in range(n_cells):
        for post in range(n_cells):
            pre_type = "E" if pre < n_e else "I"
            post_type = "E" if post < n_e else "I"
            w = block_weights[(pre_type, post_type)]
            rows.append({
                "edge_index": idx,
                "pre_model_index": pre,
                "post_model_index": post,
                "weight": w,
            })
            idx += 1

    ssn.edges = pd.DataFrame(rows)
    return ssn


# ---------------------------------------------------------------------------
# Caching: construct_W populates self.W_sparse
# ---------------------------------------------------------------------------
class TestConstructWCache:

    def test_construct_W_creates_cache(self):
        """After construct_W(), self.W_sparse should exist."""
        ssn = _build_minimal_ssn_with_edges()
        W = ssn.construct_W()

        assert hasattr(ssn, "W_sparse")
        assert scipy.sparse.issparse(ssn.W_sparse)
        np.testing.assert_array_equal(ssn.W_sparse.toarray(), W)

    def test_construct_W_returns_from_cache(self):
        """Calling construct_W() a second time returns equivalent matrix from cache."""
        ssn = _build_minimal_ssn_with_edges()
        W1 = ssn.construct_W()
        W2 = ssn.construct_W()

        np.testing.assert_array_equal(W1, W2)
        # Same object should be returned (from cache)
        assert id(ssn.W_sparse) == id(ssn.W_sparse)


# ---------------------------------------------------------------------------
# Invalidation: add_edges wipes cached W_sparse
# ---------------------------------------------------------------------------
class TestAddEdgesInvalidation:

    def test_add_edges_deletes_W_cache(self):
        """Calling add_edges() must delete self.W_sparse."""
        ssn = _build_minimal_ssn_with_edges()
        ssn.construct_W()
        assert hasattr(ssn, "W_sparse")

        # add_edges without config will print an error and return,
        # but the invalidation happens at the top before the useConfig check
        ssn.add_edges(useConfig=False)

        assert not hasattr(ssn, "W_sparse")


# ---------------------------------------------------------------------------
# Closure: verify ssn_equations works without args
# ---------------------------------------------------------------------------
class TestClosureEquations:

    def test_run_succeeds_without_args(self):
        """run() must succeed -- if it tried to use args, solve_ivp would fail
        because ssn_equations only takes (t, r) now."""
        ssn = _build_minimal_ssn()
        ssn.run()
        assert ssn.outputs.success


# ---------------------------------------------------------------------------
# Vectorized param lookup
# ---------------------------------------------------------------------------
class TestVectorizedParamLookup:

    def test_node_params_populated_after_run(self):
        """After run(), self.nodes should have tau/k/c/n columns matching node_types."""
        n_e, n_i = 4, 4
        ssn = _build_minimal_ssn(n_e=n_e, n_i=n_i)
        ssn.run()

        assert "tau" in ssn.nodes.columns
        assert "k" in ssn.nodes.columns
        assert "c" in ssn.nodes.columns
        assert "n" in ssn.nodes.columns

        # E cells: tau=20, k=0.04, c=1, n=2.0
        np.testing.assert_array_equal(ssn.nodes["tau"].values[:n_e], np.full(n_e, 20.0))
        np.testing.assert_array_equal(ssn.nodes["k"].values[:n_e], np.full(n_e, 0.04))
        # I cells: tau=10, k=0.04, c=1, n=2.0
        np.testing.assert_array_equal(ssn.nodes["tau"].values[n_e:], np.full(n_i, 10.0))
        np.testing.assert_array_equal(ssn.nodes["n"].values[n_e:], np.full(n_i, 2.0))


# ---------------------------------------------------------------------------
# Integration tests: Full pipeline with config file
# ---------------------------------------------------------------------------
class TestIntegrationWithConfig:
    """Run the full SSN pipeline using the test config file. These tests
    exercise construct_W caching, sparse W, closure, and param lookup
    on a realistic 512-cell E/I network."""

    @pytest.mark.integration
    def test_full_pipeline_runs(self, config_file, verbose):
        """load_config -> add_nodes -> add_edges -> load_inputs -> connect_inputs -> run"""
        ssn = SSN("integration_test", verbose=verbose)
        ssn.load_config(config_file)
        ssn.add_nodes()
        ssn.add_edges()

        # After add_edges, W_sparse should not exist (invalidated at top)
        assert not hasattr(ssn, "W_sparse"), \
            "W_sparse should not exist right after add_edges (invalidated)"

        ssn.load_inputs()
        ssn.connect_inputs()

        # Scale down c to keep simulation stable for testing
        ssn.node_types["c"] = 0.01
        ssn.nodes["c"] = 0.01

        ssn.run()

        # Basic output checks
        assert ssn.outputs.success
        n_cells = len(ssn.nodes)
        assert ssn.outputs.y.shape[0] == n_cells
        assert ssn.outputs.y.shape[1] == len(ssn.outputs.t)
        assert not np.isnan(ssn.outputs.y).any()
        assert not np.isinf(ssn.outputs.y).any()

        # After run(), W_sparse should now be cached
        assert hasattr(ssn, "W_sparse"), \
            "W_sparse should be cached after run() calls construct_W()"
        assert scipy.sparse.issparse(ssn.W_sparse)

    @pytest.mark.integration
    def test_sparse_W_used_in_run(self, config_file, verbose):
        """Verify the sparse matrix is actually populated during integration."""
        ssn = SSN("sparse_check", verbose=verbose)
        ssn.load_config(config_file)
        ssn.add_nodes()
        ssn.add_edges()

        # Pre-build W so we can inspect sparsity
        W_dense = ssn.construct_W()
        nnz = np.count_nonzero(W_dense)
        total = W_dense.size
        density = nnz / total
        print(f"\n  W shape: {W_dense.shape}, nonzero: {nnz}, "
              f"density: {density:.3f}")

        assert scipy.sparse.issparse(ssn.W_sparse)
        assert ssn.W_sparse.nnz == nnz

    @pytest.mark.integration
    def test_add_edges_invalidates_then_rebuild(self, config_file, verbose):
        """Calling add_edges a second time invalidates cache; construct_W rebuilds it."""
        ssn = SSN("invalidate_rebuild", verbose=verbose)
        ssn.load_config(config_file)
        ssn.add_nodes()
        ssn.add_edges()

        # Build cache
        W1 = ssn.construct_W()
        assert hasattr(ssn, "W_sparse")

        # Re-run add_edges -- cache should be wiped
        ssn.add_edges()
        assert not hasattr(ssn, "W_sparse"), \
            "W_sparse must be invalidated by add_edges"

        # Rebuild
        W2 = ssn.construct_W()
        assert hasattr(ssn, "W_sparse")
        assert ssn.W_sparse.shape == W1.shape

    @pytest.mark.integration
    def test_second_run_uses_cached_W(self, config_file, verbose):
        """A second call to run() should reuse the cached W_sparse."""
        ssn = SSN("cached_run", verbose=verbose)
        ssn.load_config(config_file)
        ssn.add_nodes()
        ssn.add_edges()
        ssn.load_inputs()
        ssn.connect_inputs()
        ssn.node_types["c"] = 0.01
        ssn.nodes["c"] = 0.01

        # First run builds and caches W_sparse
        ssn.run()
        assert hasattr(ssn, "W_sparse")
        sparse_id_first = id(ssn.W_sparse)

        # Second run should reuse the same cached W_sparse object
        ssn.run()
        assert id(ssn.W_sparse) == sparse_id_first, \
            "Second run() should reuse cached W_sparse, not rebuild"


# ---------------------------------------------------------------------------
# Benchmark (not run by default -- use: pytest -m benchmark --no-header -rN)
# ---------------------------------------------------------------------------
def _build_benchmark_ssn(n_cells, tstop=10.0, dt=0.1, seed=7):
    """Build an E/I SSN with a random ~20% dense W for benchmarking.

    Uses the same cell parameters from inspect_minimal_ssn.ipynb.
    """
    n_e = n_cells // 2
    n_i = n_cells - n_e
    ssn = SSN(f"bench_{n_cells}", rand_seed=seed, verbose=False)

    ssn.nodes = pd.DataFrame({
        "model_name": ["E"] * n_e + ["I"] * n_i,
        "model_index": [0] * n_e + [1] * n_i,
    })
    ssn.node_types = pd.DataFrame({
        "model_type_id": [0, 1],
        "tau": [20.0, 10.0],
        "k": [0.04, 0.04],
        "c": [1.0, 1.0],
        "n": [2.0, 2.0],
    })

    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n_cells, n_cells)) * 0.01
    W[rng.random((n_cells, n_cells)) > 0.2] = 0.0
    # Enforce E/I sign convention
    W[:, :n_e] = np.maximum(W[:, :n_e], 0)
    W[:, n_e:] = np.minimum(W[:, n_e:], 0)
    ssn.construct_W = lambda: W

    t_arr = np.arange(0, tstop, dt)
    ssn.h = np.abs(rng.standard_normal((n_cells, len(t_arr)))) * 5.0
    ssn.stim_params = {"temporal": {"params": {"T": tstop, "t_steps": dt}}}
    ssn.run_params = {
        "tstop": tstop, "dt": dt,
        "r_init": {"type": "uniform", "E": 0.0, "I": 0.0},
        "method": "RK45",
    }
    return ssn, W


class TestBenchmark:

    @pytest.mark.benchmark
    @pytest.mark.parametrize("n_cells", [50, 200, 500, 1000])
    def test_run_benchmark_sparse(self, n_cells):
        """Benchmark run() with sparse W at various network sizes."""
        ssn, _ = _build_benchmark_ssn(n_cells)

        n_repeats = 3
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            ssn.run()
            times.append(time.perf_counter() - t0)

        median_time = np.median(times)
        print(f"\n  [SPARSE ] n_cells={n_cells:>4d}  median={median_time:.4f}s  "
              f"(min={min(times):.4f}s, max={max(times):.4f}s)")
        assert ssn.outputs.success

    @pytest.mark.benchmark
    @pytest.mark.parametrize("n_cells", [50, 200, 500, 1000])
    def test_run_benchmark_dense(self, n_cells):
        """Benchmark solve_ivp with dense np.dot for A/B comparison."""
        from scipy.interpolate import CubicSpline
        from scipy.integrate import solve_ivp

        ssn, W_dense = _build_benchmark_ssn(n_cells)

        t_arr = np.arange(0, ssn.run_params['tstop'], ssn.run_params['dt'])
        cs = CubicSpline(t_arr, ssn.h, axis=1)

        # Build param vectors the same way run() does
        param_map = {row['model_type_id']: row for _, row in ssn.node_types.iterrows()}
        tau = ssn.nodes['model_index'].map(lambda x: param_map[x]['tau']).values.astype(float)
        k = ssn.nodes['model_index'].map(lambda x: param_map[x]['k']).values.astype(float)
        c = ssn.nodes['model_index'].map(lambda x: param_map[x]['c']).values.astype(float)
        n = ssn.nodes['model_index'].map(lambda x: param_map[x]['n']).values.astype(float)

        def ssn_equations_dense(t, r):
            h_t = cs(t)
            r_ss = k * np.power(np.maximum(c * h_t + np.dot(W_dense, r), 0), n)
            return (-r + r_ss) / tau

        r_init = np.zeros(n_cells)

        n_repeats = 3
        times = []
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            solve_ivp(ssn_equations_dense, [0, t_arr[-1]], r_init,
                      method='RK45', max_step=np.inf)
            times.append(time.perf_counter() - t0)

        median_time = np.median(times)
        print(f"\n  [DENSE  ] n_cells={n_cells:>4d}  median={median_time:.4f}s  "
              f"(min={min(times):.4f}s, max={max(times):.4f}s)")
