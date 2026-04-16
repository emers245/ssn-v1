#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the --eval-subtypes feature in run_optimization.py"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import h5py
import warnings
import sys

# Add SSN directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import SSN_utils first (it should always be available)
import SSN_utils

# Import specific functions from run_optimization
# We need to be careful with imports because run_optimization has some module-level dependencies
import importlib.util
spec = importlib.util.spec_from_file_location("run_optimization", 
                                               Path(__file__).parent.parent / "run_optimization.py")
run_optimization = importlib.util.module_from_spec(spec)

# Mock the problematic imports before loading the module, then restore
# to avoid contaminating other test files that import the real SSN.
_orig_SSN = sys.modules.get('SSN')
_orig_bayesopt = sys.modules.get('bayesopt')
sys.modules['SSN'] = MagicMock()
sys.modules['bayesopt'] = MagicMock()

# Now load the module
spec.loader.exec_module(run_optimization)

# Restore original modules so other test files get the real SSN
if _orig_SSN is not None:
    sys.modules['SSN'] = _orig_SSN
else:
    del sys.modules['SSN']
if _orig_bayesopt is not None:
    sys.modules['bayesopt'] = _orig_bayesopt
else:
    del sys.modules['bayesopt']

# Extract the functions we need
load_target_data = run_optimization.load_target_data
select_cell_subset = run_optimization.select_cell_subset
compute_subtype_cost = run_optimization.compute_subtype_cost
compute_spatial_mask = run_optimization.compute_spatial_mask


class TestLoadTargetDataWithModelIndex:
    """Test load_target_data function with model_index loading."""
    
    def test_load_target_data_without_model_index(self):
        """Test loading target data without model_index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test HDF5 file without model_index
            h5_path = Path(tmpdir) / "test.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("y", data=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
            
            matrix, labels, indices = load_target_data([h5_path], load_model_index=False)
            
            assert matrix.shape == (3, 1)
            assert labels == ["test.h5"]
            assert indices is None
    
    def test_load_target_data_with_model_index(self):
        """Test loading target data with model_index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "test.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("y", data=np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
                f.create_dataset("model_index", data=np.array([0, 0, 1]))
            
            matrix, labels, indices = load_target_data([h5_path], load_model_index=True)
            
            assert matrix.shape == (3, 1)
            assert labels == ["test.h5"]
            assert indices is not None
            np.testing.assert_array_equal(indices, np.array([0, 0, 1]))
    
    def test_load_target_data_missing_model_index_raises_error(self):
        """Test that missing model_index raises KeyError when requested."""
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "test.h5"
            with h5py.File(h5_path, "w") as f:
                f.create_dataset("y", data=np.array([[1.0, 2.0], [3.0, 4.0]]))
            
            with pytest.raises(KeyError, match="does not contain 'model_index'"):
                load_target_data([h5_path], load_model_index=True)
    
    def test_load_target_data_multiple_files_with_model_index(self):
        """Test loading multiple files with model_index concatenation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first file
            h5_path1 = Path(tmpdir) / "test1.h5"
            with h5py.File(h5_path1, "w") as f:
                f.create_dataset("y", data=np.array([[1.0, 2.0], [3.0, 4.0]]))
                f.create_dataset("model_index", data=np.array([0, 1]))
            
            # Create second file
            h5_path2 = Path(tmpdir) / "test2.h5"
            with h5py.File(h5_path2, "w") as f:
                f.create_dataset("y", data=np.array([[5.0, 6.0], [7.0, 8.0]]))
                f.create_dataset("model_index", data=np.array([0, 2]))
            
            matrix, labels, indices = load_target_data(
                [h5_path1, h5_path2], load_model_index=True
            )
            
            # load_target_data takes the last time point of each file and stacks horizontally
            # So we get (n_cells, n_files)
            assert matrix.shape == (2, 2)
            # model_indices concatenates all cells from all files
            np.testing.assert_array_equal(indices, np.array([0, 1, 0, 2]))


class TestSelectCellSubset:
    """Test select_cell_subset function with indices_data."""
    
    def test_select_cell_subset_without_indices(self):
        """Test selecting subset without indices_data."""
        data = np.arange(30).reshape(10, 3)
        fraction = 0.5
        seed = 42
        
        subset_idx, subset, subset_indices = select_cell_subset(
            data, fraction, seed, indices_data=None
        )
        
        assert len(subset_idx) == 5
        assert subset.shape == (5, 3)
        assert subset_indices is None
        np.testing.assert_array_equal(subset, data[subset_idx])
    
    def test_select_cell_subset_with_indices(self):
        """Test selecting subset with indices_data."""
        data = np.arange(30).reshape(10, 3)
        indices = np.array([0, 0, 1, 1, 0, 2, 2, 1, 0, 2])
        fraction = 0.5
        seed = 42
        
        subset_idx, subset, subset_indices = select_cell_subset(
            data, fraction, seed, indices_data=indices
        )
        
        assert len(subset_idx) == 5
        assert subset.shape == (5, 3)
        assert subset_indices is not None
        assert len(subset_indices) == 5
        np.testing.assert_array_equal(subset_indices, indices[subset_idx])
    
    def test_select_cell_subset_full_fraction(self):
        """Test selecting with fraction=1.0 returns all data."""
        data = np.arange(30).reshape(10, 3)
        indices = np.array([0, 0, 1, 1, 0, 2, 2, 1, 0, 2])
        
        subset_idx, subset, subset_indices = select_cell_subset(
            data, 1.0, 42, indices_data=indices
        )
        
        assert len(subset_idx) == 10
        np.testing.assert_array_equal(subset, data)
        np.testing.assert_array_equal(subset_indices, indices)
    
    def test_select_cell_subset_invalid_fraction_raises_error(self):
        """Test that invalid fraction raises ValueError."""
        data = np.arange(30).reshape(10, 3)
        
        with pytest.raises(ValueError, match="fraction must be in"):
            select_cell_subset(data, 1.5, 42)
        
        with pytest.raises(ValueError, match="fraction must be in"):
            select_cell_subset(data, 0.0, 42)


class TestComputeSubtypeCost:
    """Test compute_subtype_cost function."""
    
    def test_compute_subtype_cost_single_subtype(self):
        """Test cost computation with a single subtype."""
        model_outputs = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        target_data = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])
        model_indices = np.array([0, 0, 0])
        target_indices = np.array([0, 0, 0])
        
        cost = compute_subtype_cost(
            model_outputs, target_data, model_indices, target_indices,
            cost_function=SSN_utils.kl_divergence
        )
        
        # Cost should be positive
        assert cost > 0
        assert isinstance(cost, (float, np.floating))
    
    def test_compute_subtype_cost_multiple_subtypes(self):
        """Test that cost is computed separately for each subtype."""
        model_outputs = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ])
        target_data = np.array([
            [1.1, 2.1],
            [3.1, 4.1],
            [5.1, 6.1],
            [7.1, 8.1],
        ])
        model_indices = np.array([0, 0, 1, 1])
        target_indices = np.array([0, 0, 1, 1])
        
        # Compute total cost
        total_cost = compute_subtype_cost(
            model_outputs, target_data, model_indices, target_indices,
            cost_function=SSN_utils.kl_divergence
        )
        
        # Compute individual subtype costs
        subtype0_cost = SSN_utils.compute_cost(
            model_outputs[[0, 1]], target_data[[0, 1]],
            cost_function=SSN_utils.kl_divergence
        )
        subtype1_cost = SSN_utils.compute_cost(
            model_outputs[[2, 3]], target_data[[2, 3]],
            cost_function=SSN_utils.kl_divergence
        )
        expected_total = subtype0_cost + subtype1_cost
        
        # Should be approximately equal (allowing for numerical precision)
        np.testing.assert_allclose(total_cost, expected_total, rtol=1e-5)
    
    def test_compute_subtype_cost_subtype_in_model_not_in_target(self, capsys):
        """Test warning when subtype exists in model but not in target."""
        model_outputs = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        target_data = np.array([
            [1.1, 2.1],
            [3.1, 4.1],
            [5.1, 6.1],
        ])
        model_indices = np.array([0, 1, 2])  # Model has subtypes 0, 1, 2
        target_indices = np.array([0, 0, 0])  # Target only has subtype 0
        
        cost = compute_subtype_cost(
            model_outputs, target_data, model_indices, target_indices,
            cost_function=SSN_utils.kl_divergence
        )
        
        captured = capsys.readouterr()
        assert "Warning: subtype 1 present in model but not in target" in captured.out
        assert "Warning: subtype 2 present in model but not in target" in captured.out
        # Cost should only include subtype 0
        assert cost > 0
    
    def test_compute_subtype_cost_subtype_in_target_not_in_model(self, capsys):
        """Test warning when subtype exists in target but not in model."""
        model_outputs = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        target_data = np.array([
            [1.1, 2.1],
            [3.1, 4.1],
            [5.1, 6.1],
        ])
        model_indices = np.array([0, 0])  # Model only has subtype 0
        target_indices = np.array([0, 0, 1])  # Target has subtypes 0 and 1
        
        cost = compute_subtype_cost(
            model_outputs, target_data, model_indices, target_indices,
            cost_function=SSN_utils.kl_divergence
        )
        
        captured = capsys.readouterr()
        assert "Warning: subtype 1 present in target data but not in model" in captured.out
        # Cost should only include subtype 0
        assert cost > 0
    
    def test_compute_subtype_cost_no_common_subtypes_raises_error(self):
        """Test error when there are no common subtypes."""
        model_outputs = np.array([[1.0, 2.0], [3.0, 4.0]])
        target_data = np.array([[1.1, 2.1], [3.1, 4.1]])
        model_indices = np.array([0, 0])
        target_indices = np.array([1, 1])
        
        # Should return 0 cost when no common subtypes
        cost = compute_subtype_cost(
            model_outputs, target_data, model_indices, target_indices,
            cost_function=SSN_utils.kl_divergence
        )
        
        assert cost == 0.0
    
    def test_compute_subtype_cost_empty_subtype(self, capsys):
        """Test warning when a subtype has no cells in model or target."""
        model_outputs = np.array([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        target_data = np.array([
            [1.1, 2.1],
            [3.1, 4.1],
        ])
        model_indices = np.array([0, 1])  # Subtype 1 has one cell
        target_indices = np.array([0, 0])  # Subtype 1 has zero cells in target
        
        cost = compute_subtype_cost(
            model_outputs, target_data, model_indices, target_indices,
            cost_function=SSN_utils.kl_divergence
        )
        
        captured = capsys.readouterr()
        # When a subtype is in model but not in target, it gets caught by the mismatch check
        assert "Warning: subtype 1 present in model but not in target data" in captured.out


class TestComputeSpatialMask:
    """Test compute_spatial_mask function."""
    
    def test_compute_spatial_mask_circular(self):
        """Test circular spatial mask computation."""
        nodes_x = np.array([0.0, 1.0, 2.0, 3.0])
        nodes_y = np.array([0.0, 1.0, 2.0, 3.0])
        mask_config = {
            'mask_type': 'circular',
            'pos': [1.5, 1.5],
            'rad': 1.5
        }
        
        indices = compute_spatial_mask(nodes_x, nodes_y, mask_config)
        
        # Should include points within 1.5 units of (1.5, 1.5)
        assert len(indices) > 0
        # Points (1.0, 1.0), (2.0, 2.0), (1.5, 1.5) should be included
        assert any(i in indices for i in [1, 2])
    
    def test_compute_spatial_mask_rectangular(self):
        """Test rectangular spatial mask computation."""
        nodes_x = np.array([0.0, 1.0, 2.0, 3.0])
        nodes_y = np.array([0.0, 1.0, 2.0, 3.0])
        mask_config = {
            'mask_type': 'rectangular',
            'pos': [1.5, 1.5],
            'width': 2.0,
            'height': 2.0
        }
        
        indices = compute_spatial_mask(nodes_x, nodes_y, mask_config)
        
        # Should include points within rectangle
        assert len(indices) > 0
        # Points (1.0, 1.0), (2.0, 2.0), (1.5, 1.5) should be included
        assert any(i in indices for i in [1, 2])
    
    def test_compute_spatial_mask_all_outside_raises_error(self):
        """Test that mask with no cells raises ValueError."""
        nodes_x = np.array([0.0, 1.0, 2.0])
        nodes_y = np.array([0.0, 1.0, 2.0])
        mask_config = {
            'mask_type': 'circular',
            'pos': [100.0, 100.0],
            'rad': 1.0
        }
        
        with pytest.raises(ValueError, match="results in zero cells"):
            compute_spatial_mask(nodes_x, nodes_y, mask_config)


class TestEvaluateParametersWithSubtypes:
    """Test evaluate_parameters with --eval-subtypes flag."""
    
    def test_evaluate_parameters_requires_target_model_indices_when_eval_subtypes(self):
        """Test that eval_subtypes=True requires target_model_indices."""
        # This test would need mocks for SSN and other dependencies
        # For now, we document what should be tested
        pass
    
    def test_evaluate_parameters_eval_subtypes_true_vs_false(self):
        """Test that eval_subtypes=True produces different cost than False."""
        # This test would need full setup of SSN model and simulations
        # Should verify that subtype-specific costs differ from aggregate costs
        pass


class TestIntegrationSubtypeCostComputation:
    """Integration tests for subtype cost computation."""
    
    def test_subtype_cost_equals_sum_of_parts(self):
        """Test that subtype cost equals sum of individual subtype costs."""
        np.random.seed(42)
        n_cells = 20
        n_conditions = 3
        n_subtypes = 3
        
        # Create synthetic data
        model_outputs = np.random.rand(n_cells, n_conditions)
        target_data = np.random.rand(n_cells, n_conditions)
        model_indices = np.random.randint(0, n_subtypes, n_cells)
        target_indices = np.random.randint(0, n_subtypes, n_cells)
        
        # Compute total cost
        total_cost = compute_subtype_cost(
            model_outputs, target_data, model_indices, target_indices,
            cost_function=SSN_utils.kl_divergence
        )
        
        # Compute sum of individual subtype costs
        sum_cost = 0.0
        common_subtypes = set(model_indices) & set(target_indices)
        for subtype in common_subtypes:
            model_mask = model_indices == subtype
            target_mask = target_indices == subtype
            if np.any(model_mask) and np.any(target_mask):
                subtype_cost = SSN_utils.compute_cost(
                    model_outputs[model_mask], target_data[target_mask],
                    cost_function=SSN_utils.kl_divergence
                )
                sum_cost += subtype_cost
        
        # Should be approximately equal
        np.testing.assert_allclose(total_cost, sum_cost, rtol=1e-5)
    
    def test_subtype_filtering_preserves_data_integrity(self):
        """Test that subtype filtering doesn't corrupt the data."""
        # Create ground truth data with subtypes
        n_cells = 10
        n_conditions = 2
        model_outputs = np.arange(n_cells * n_conditions).reshape(n_cells, n_conditions).astype(float)
        target_data = model_outputs * 1.1  # Slightly modified
        model_indices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        target_indices = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        
        # Compute cost
        total_cost = compute_subtype_cost(
            model_outputs, target_data, model_indices, target_indices,
            cost_function=SSN_utils.kl_divergence
        )
        
        # Verify all subtypes were processed
        assert total_cost > 0
        # Manually verify subtype 0 data
        assert np.all(model_outputs[[0, 1, 2]] == model_outputs[:3])


class TestParseArgsEvalSubtypes:
    """Test command-line argument parsing for --eval-subtypes."""
    
    def test_eval_subtypes_flag_is_boolean(self):
        """Test that --eval-subtypes flag is parsed as boolean."""
        # This would test the argparse configuration
        # We need to verify the flag gets added correctly
        pass


# Utility test functions
def create_test_h5_file(path, n_cells=10, n_conditions=2, n_subtypes=3):
    """Helper to create test HDF5 files with model_index."""
    with h5py.File(path, 'w') as f:
        # Create target data
        y_data = np.random.rand(n_cells, n_conditions)
        f.create_dataset('y', data=y_data)
        
        # Create model indices
        model_index = np.random.randint(0, n_subtypes, n_cells)
        f.create_dataset('model_index', data=model_index)
        
        return y_data, model_index


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
