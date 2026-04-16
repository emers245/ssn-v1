#!/usr/bin/env python3
"""
Tests for spatial masking functionality in run_optimization.py

Tests cover:
1. Mask computation correctness (circular and rectangular)
2. Mask validation error handling
3. Data flow: ground truth subsampling vs model output masking
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path

try:
    from ..run_optimization import (
        parse_mask_config,
        compute_spatial_mask,
        select_cell_subset,
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from run_optimization import (
        parse_mask_config,
        compute_spatial_mask,
        select_cell_subset,
    )


class TestParsemaskConfig:
    """Test mask configuration parsing and validation."""

    def test_parse_valid_circular_config(self):
        """Valid circular mask config should load without error."""
        config = {
            "mask_type": "circular",
            "pos": [0.0, 0.0],
            "rad": 50.0,
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            result = parse_mask_config(temp_path)
            assert result["mask_type"] == "circular"
            assert result["pos"] == [0.0, 0.0]
            assert result["rad"] == 50.0
        finally:
            Path(temp_path).unlink()

    def test_parse_valid_rectangular_config(self):
        """Valid rectangular mask config should load without error."""
        config = {
            "mask_type": "rectangular",
            "pos": [10.0, 20.0],
            "width": 100.0,
            "height": 80.0,
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            result = parse_mask_config(temp_path)
            assert result["mask_type"] == "rectangular"
            assert result["width"] == 100.0
            assert result["height"] == 80.0
        finally:
            Path(temp_path).unlink()

    def test_parse_none_returns_none(self):
        """parse_mask_config(None) should return None."""
        assert parse_mask_config(None) is None

    def test_missing_mask_type_raises_error(self):
        """Config without mask_type should raise ValueError."""
        config = {"pos": [0.0, 0.0], "rad": 50.0}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="mask_type"):
                parse_mask_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_missing_pos_raises_error(self):
        """Config without pos should raise ValueError."""
        config = {"mask_type": "circular", "rad": 50.0}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="pos"):
                parse_mask_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_invalid_mask_type_raises_error(self):
        """Invalid mask_type should raise ValueError."""
        config = {
            "mask_type": "triangle",  # Invalid
            "pos": [0.0, 0.0],
            "rad": 50.0,
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="must be 'circular' or 'rectangular'"):
                parse_mask_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_circular_missing_radius_raises_error(self):
        """Circular mask without radius should raise ValueError."""
        config = {"mask_type": "circular", "pos": [0.0, 0.0]}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="rad"):
                parse_mask_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_circular_negative_radius_raises_error(self):
        """Circular mask with negative radius should raise ValueError."""
        config = {"mask_type": "circular", "pos": [0.0, 0.0], "rad": -10.0}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="positive"):
                parse_mask_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_rectangular_missing_dimensions_raises_error(self):
        """Rectangular mask without width or height should raise ValueError."""
        config = {"mask_type": "rectangular", "pos": [0.0, 0.0], "width": 100.0}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="width.*height"):
                parse_mask_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_rectangular_negative_dimensions_raises_error(self):
        """Rectangular mask with negative dimensions should raise ValueError."""
        config = {
            "mask_type": "rectangular",
            "pos": [0.0, 0.0],
            "width": -100.0,
            "height": 100.0,
        }
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="positive"):
                parse_mask_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_pos_wrong_length_raises_error(self):
        """pos with wrong number of elements should raise ValueError."""
        config = {"mask_type": "circular", "pos": [0.0, 0.0, 0.0], "rad": 50.0}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="2-element"):
                parse_mask_config(temp_path)
        finally:
            Path(temp_path).unlink()


class TestComputeSpatialMask:
    """Test spatial mask computation."""

    def test_circular_mask_origin_centered(self):
        """Circular mask centered at origin should include nearby cells."""
        # Create grid of nodes
        nodes_x = np.array([-10.0, 0.0, 10.0, -10.0, 0.0, 10.0, -10.0, 0.0, 10.0])
        nodes_y = np.array([-10.0, -10.0, -10.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
        
        config = {"mask_type": "circular", "pos": [0.0, 0.0], "rad": 12.0}
        mask_indices = compute_spatial_mask(nodes_x, nodes_y, config)
        
        # Center cell (index 4) should be included
        assert 4 in mask_indices
        # Corner cells (0, 2, 6, 8) at distance ~14.14 should be excluded
        assert 0 not in mask_indices
        assert 2 not in mask_indices
        assert 6 not in mask_indices
        assert 8 not in mask_indices
        # Edge cells (1, 3, 5, 7) at distance 10 should be included
        assert 1 in mask_indices
        assert 3 in mask_indices
        assert 5 in mask_indices
        assert 7 in mask_indices

    def test_circular_mask_offset_center(self):
        """Circular mask with offset center should work correctly."""
        nodes_x = np.array([0.0, 1.0, 2.0])
        nodes_y = np.array([0.0, 0.0, 0.0])
        
        config = {"mask_type": "circular", "pos": [1.0, 0.0], "rad": 1.5}
        mask_indices = compute_spatial_mask(nodes_x, nodes_y, config)
        
        # Node at (1.0, 0.0) is at distance 0
        assert 1 in mask_indices
        # Nodes at (0.0, 0.0) and (2.0, 0.0) are at distance 1.0
        assert 0 in mask_indices
        assert 2 in mask_indices

    def test_rectangular_mask_origin_centered(self):
        """Rectangular mask centered at origin should select interior cells."""
        nodes_x = np.array([-20.0, 0.0, 20.0, -20.0, 0.0, 20.0, -20.0, 0.0, 20.0])
        nodes_y = np.array([-20.0, -20.0, -20.0, 0.0, 0.0, 0.0, 20.0, 20.0, 20.0])
        
        config = {"mask_type": "rectangular", "pos": [0.0, 0.0], "width": 30.0, "height": 30.0}
        mask_indices = compute_spatial_mask(nodes_x, nodes_y, config)
        
        # Center cell (index 4) should be included
        assert 4 in mask_indices
        # All other cells are at exactly ±20 which is > ±15
        assert len(mask_indices) == 1

    def test_rectangular_mask_includes_boundary(self):
        """Rectangular mask should use strict inequality (< not <=) for boundaries."""
        # Nodes strictly inside the boundaries should be included
        nodes_x = np.array([-14.9, 0.0, 14.9])
        nodes_y = np.array([0.0, 0.0, 0.0])
        
        config = {"mask_type": "rectangular", "pos": [0.0, 0.0], "width": 30.0, "height": 30.0}
        mask_indices = compute_spatial_mask(nodes_x, nodes_y, config)
        
        # All three should be included (strictly inside ±15)
        assert 0 in mask_indices
        assert 1 in mask_indices
        assert 2 in mask_indices

    def test_rectangular_mask_excludes_outside(self):
        """Rectangular mask should exclude cells outside bounds."""
        nodes_x = np.array([-16.0, 0.0, 16.0])
        nodes_y = np.array([0.0, 0.0, 0.0])
        
        config = {"mask_type": "rectangular", "pos": [0.0, 0.0], "width": 30.0, "height": 30.0}
        mask_indices = compute_spatial_mask(nodes_x, nodes_y, config)
        
        # Corner cells at ±16 should be excluded (outside ±15)
        assert 0 not in mask_indices
        assert 2 not in mask_indices
        # Center at 0 should be included
        assert 1 in mask_indices

    def test_zero_nodes_selected_raises_error(self):
        """Mask that selects zero cells should raise ValueError."""
        nodes_x = np.array([100.0, 200.0])
        nodes_y = np.array([100.0, 200.0])
        
        config = {"mask_type": "circular", "pos": [0.0, 0.0], "rad": 10.0}
        with pytest.raises(ValueError, match="results in zero cells"):
            compute_spatial_mask(nodes_x, nodes_y, config)

    def test_all_nodes_selected(self):
        """Mask with large radius should select all nodes."""
        nodes_x = np.array([-10.0, 0.0, 10.0])
        nodes_y = np.array([-10.0, 0.0, 10.0])
        
        config = {"mask_type": "circular", "pos": [0.0, 0.0], "rad": 1000.0}
        mask_indices = compute_spatial_mask(nodes_x, nodes_y, config)
        
        assert len(mask_indices) == 3
        assert set(mask_indices) == {0, 1, 2}

    def test_mask_indices_are_sorted(self):
        """Mask indices should be sorted."""
        nodes_x = np.random.uniform(-50, 50, 20)
        nodes_y = np.random.uniform(-50, 50, 20)
        
        config = {"mask_type": "circular", "pos": [0.0, 0.0], "rad": 30.0}
        mask_indices = compute_spatial_mask(nodes_x, nodes_y, config)
        
        # Verify they're sorted
        assert np.all(mask_indices[:-1] <= mask_indices[1:])


class TestCellSubsetting:
    """Test cell subsetting functionality (for ground truth)."""

    def test_select_full_fraction(self):
        """Selecting fraction=1.0 should return all cells."""
        data = np.random.randn(100, 5)
        indices, subset, _ = select_cell_subset(data, fraction=1.0, seed=42)
        
        assert len(indices) == 100
        assert subset.shape == (100, 5)
        assert np.array_equal(indices, np.arange(100))

    def test_select_half_fraction(self):
        """Selecting fraction=0.5 should return half the cells."""
        data = np.random.randn(100, 5)
        indices, subset, _ = select_cell_subset(data, fraction=0.5, seed=42)
        
        assert len(indices) == 50
        assert subset.shape == (50, 5)

    def test_select_preserves_data(self):
        """Selected subset should contain correct data values."""
        data = np.arange(100).reshape(100, 1)
        indices, subset, _ = select_cell_subset(data, fraction=0.5, seed=42)
        
        # Verify that subset values match data at selected indices
        assert np.array_equal(subset[:, 0], data[indices, 0])

    def test_different_seeds_give_different_subsets(self):
        """Different seeds should produce different random subsets."""
        data = np.random.randn(100, 5)
        indices1, _, _ = select_cell_subset(data, fraction=0.5, seed=42)
        indices2, _, _ = select_cell_subset(data, fraction=0.5, seed=43)
        
        # Very unlikely they'd be identical
        assert not np.array_equal(indices1, indices2)

    def test_same_seed_gives_same_subset(self):
        """Same seed should produce same subset."""
        data = np.random.randn(100, 5)
        indices1, subset1, _ = select_cell_subset(data, fraction=0.5, seed=42)
        indices2, subset2, _ = select_cell_subset(data, fraction=0.5, seed=42)
        
        assert np.array_equal(indices1, indices2)
        assert np.array_equal(subset1, subset2)


class TestDataFlowIntegration:
    """Integration tests for data flow through masking."""

    def test_ground_truth_and_model_cells_can_differ(self):
        """Ground truth cell count (fraction) and model cells (mask) can be different."""
        # Simulate ground truth with 100 cells
        ground_truth = np.random.randn(100, 5)
        
        # Subsample ground truth to 50 cells
        gt_indices, gt_subset, _ = select_cell_subset(ground_truth, fraction=0.5, seed=42)
        assert gt_subset.shape == (50, 5)
        
        # Simulate model with 200 cells
        nodes_x = np.random.uniform(-50, 50, 200)
        nodes_y = np.random.uniform(-50, 50, 200)
        
        # Apply spatial mask to select 80 cells
        mask_config = {"mask_type": "circular", "pos": [0.0, 0.0], "rad": 30.0}
        model_indices = compute_spatial_mask(nodes_x, nodes_y, mask_config)
        model_output = np.random.randn(200, 5)
        model_masked = model_output[model_indices, :]
        
        # Shapes should differ
        assert gt_subset.shape[0] != model_masked.shape[0]
        # But both should have same number of conditions
        assert gt_subset.shape[1] == model_masked.shape[1]

    def test_mask_applied_per_model_instance(self):
        """Different model instances should recompute mask independently."""
        nodes_x1 = np.array([0.0, 5.0, 10.0])
        nodes_y1 = np.array([0.0, 5.0, 10.0])
        
        nodes_x2 = np.array([0.0, 5.0, 10.0]) + np.random.randn(3) * 0.1  # Slightly different
        nodes_y2 = np.array([0.0, 5.0, 10.0]) + np.random.randn(3) * 0.1
        
        config = {"mask_type": "circular", "pos": [0.0, 0.0], "rad": 7.0}
        
        indices1 = compute_spatial_mask(nodes_x1, nodes_y1, config)
        indices2 = compute_spatial_mask(nodes_x2, nodes_y2, config)
        
        # Indices might differ due to position changes
        # Both should be non-empty
        assert len(indices1) > 0
        assert len(indices2) > 0

    def test_mask_produces_correct_shape(self):
        """Masked model outputs should have shape (n_masked_cells, n_conditions)."""
        # Create model with 1000 cells, 10 conditions
        model_output = np.random.randn(1000, 10)
        nodes_x = np.random.uniform(-100, 100, 1000)
        nodes_y = np.random.uniform(-100, 100, 1000)
        
        config = {"mask_type": "circular", "pos": [0.0, 0.0], "rad": 50.0}
        mask_indices = compute_spatial_mask(nodes_x, nodes_y, config)
        
        masked_output = model_output[mask_indices, :]
        
        # Shape should be (n_masked_cells, n_conditions)
        assert masked_output.shape[0] == len(mask_indices)
        assert masked_output.shape[1] == 10
        assert masked_output.shape[0] < 1000  # Should be subset


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
