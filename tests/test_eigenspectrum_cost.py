#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for correlation eigenspectrum utilities in SSN_utils.py."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add SSN directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import SSN_utils


class TestCorrelationEigenspectrum:
    def test_returns_descending_eigenvalues(self):
        rng = np.random.default_rng(1)
        data = rng.normal(size=(80, 6))

        eig = SSN_utils.correlation_eigenspectrum(data, method="pearson", rowvar=False, sort_desc=True)

        assert eig.ndim == 1
        assert eig.shape[0] == 6
        assert np.all(np.diff(eig) <= 1e-12)

    def test_matches_manual_corr_eigendecomposition(self):
        rng = np.random.default_rng(2)
        data = rng.normal(size=(120, 5))

        eig_fn = SSN_utils.correlation_eigenspectrum(data, method="pearson", rowvar=False, sort_desc=True)

        corr = np.corrcoef(data, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 1.0)
        eig_manual = np.linalg.eigvalsh(corr)[::-1]

        np.testing.assert_allclose(eig_fn, eig_manual, rtol=1e-10, atol=1e-12)

    def test_method_compatibility_aliases(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(70, 4))

        eig_pearson = SSN_utils.correlation_eigenspectrum(data, method="pearson")
        eig_fd = SSN_utils.correlation_eigenspectrum(data, method="fd")
        eig_doane = SSN_utils.correlation_eigenspectrum(data, method="doane")
        eig_auto = SSN_utils.correlation_eigenspectrum(data, method="auto")

        np.testing.assert_allclose(eig_pearson, eig_fd, rtol=0, atol=0)
        np.testing.assert_allclose(eig_pearson, eig_doane, rtol=0, atol=0)
        np.testing.assert_allclose(eig_pearson, eig_auto, rtol=0, atol=0)

    def test_drops_zero_variance_columns(self):
        rng = np.random.default_rng(4)
        variable = rng.normal(size=(90, 3))
        constant = np.ones((90, 2)) * 7.0
        data = np.hstack([variable, constant])

        eig = SSN_utils.correlation_eigenspectrum(data, method="pearson", rowvar=False)

        # Constant columns should be excluded before correlation; only 3 variable cols remain.
        assert eig.shape == (3,)
        assert np.all(np.isfinite(eig))

    def test_unsupported_method_raises(self):
        rng = np.random.default_rng(5)
        data = rng.normal(size=(30, 3))

        with pytest.raises(ValueError, match="Unsupported method"):
            SSN_utils.correlation_eigenspectrum(data, method="spearman")


class TestEigenspectrumMSE:
    def test_zero_for_identical_data(self):
        rng = np.random.default_rng(6)
        data = rng.normal(size=(100, 6))

        mse = SSN_utils.eigenspectrum_mse(data, data=data, method="pearson")

        assert np.isclose(mse, 0.0, atol=1e-12)

    def test_precomputed_gt_matches_data_path(self):
        rng = np.random.default_rng(7)
        model = rng.normal(size=(110, 5))
        gt = rng.normal(size=(110, 5))

        gt_eig = SSN_utils.correlation_eigenspectrum(gt, method="pearson")

        mse_data = SSN_utils.eigenspectrum_mse(model, data=gt, method="pearson")
        mse_pre = SSN_utils.eigenspectrum_mse(model, gt_eigvals=gt_eig, method="pearson")

        np.testing.assert_allclose(mse_data, mse_pre, rtol=1e-12, atol=1e-12)

    def test_shape_mismatch_truncates(self):
        """When eigenspectra have different lengths, truncate to the shorter one."""
        rng = np.random.default_rng(8)
        model = rng.normal(size=(100, 5))
        wrong_gt_eig = np.ones(3)

        # Should not raise -- truncates to min length and computes MSE
        mse = SSN_utils.eigenspectrum_mse(model, gt_eigvals=wrong_gt_eig)
        assert isinstance(mse, float)
        assert mse >= 0
