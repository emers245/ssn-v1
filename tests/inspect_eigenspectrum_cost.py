#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot and inspect correlation eigenspectra with eigenspectrum_mse."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

# Add SSN directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import SSN_utils


def _load_array(path: Path) -> np.ndarray:
    """Load a 2D array from .npy or .npz file."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as data:
            if len(data.files) == 0:
                raise ValueError(f"No arrays found in {path}")
            arr = data[data.files[0]]
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .npy or .npz")

    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
    return arr


def _make_synthetic(seed: int, n_cells: int, n_stimuli: int, noise_std: float) -> tuple[np.ndarray, np.ndarray]:
    """Create ground-truth and perturbed model outputs with known structure."""
    rng = np.random.default_rng(seed)

    # Latent-factor structure to induce non-trivial correlations.
    n_factors = min(4, max(2, n_cells // 10))
    factors = rng.normal(size=(n_stimuli, n_factors))
    loadings = rng.normal(size=(n_factors, n_cells))

    gt = factors @ loadings + 0.2 * rng.normal(size=(n_stimuli, n_cells))
    model = gt + noise_std * rng.normal(size=gt.shape)

    return model, gt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot correlation eigenspectra and print eigenspectrum MSE."
    )
    parser.add_argument("--model", type=Path, default=None, help="Path to model array (.npy/.npz)")
    parser.add_argument("--target", type=Path, default=None, help="Path to target array (.npy/.npz)")
    parser.add_argument("--method", type=str, default="pearson", help="Method passed to eigenspectrum functions")
    parser.add_argument("--rowvar", action="store_true", help="Treat rows as variables/cells")
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic mode")
    parser.add_argument("--n-cells", type=int, default=100, help="Synthetic mode: number of cells")
    parser.add_argument("--n-stimuli", type=int, default=40, help="Synthetic mode: number of stimuli/samples")
    parser.add_argument("--noise-std", type=float, default=0.2, help="Synthetic mode: perturbation strength")
    parser.add_argument("--max-components", type=int, default=None, help="Number of top eigenvalues to use in MSE")
    parser.add_argument("--save", type=Path, default=None, help="Optional output path for plot image")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive plot window")

    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Ignoring unrecognized runner arguments: {unknown_args}")

    if (args.model is None) != (args.target is None):
        raise ValueError("Provide both --model and --target, or neither (to use synthetic data).")

    if args.model is None:
        model, target = _make_synthetic(
            seed=args.seed,
            n_cells=args.n_cells,
            n_stimuli=args.n_stimuli,
            noise_std=args.noise_std,
        )
        data_label = "synthetic"
    else:
        model = _load_array(args.model)
        target = _load_array(args.target)
        data_label = f"file inputs: {args.model.name} vs {args.target.name}"

    if model.shape != target.shape:
        raise ValueError(f"Shape mismatch: model {model.shape} vs target {target.shape}")

    eig_model = SSN_utils.correlation_eigenspectrum(
        model, method=args.method, rowvar=args.rowvar, sort_desc=True, max_components=args.max_components
    )
    eig_target = SSN_utils.correlation_eigenspectrum(
        target, method=args.method, rowvar=args.rowvar, sort_desc=True, max_components=args.max_components
    )

    mse = SSN_utils.eigenspectrum_mse(
        model, method=args.method, gt_eigvals=eig_target, rowvar=args.rowvar, max_components=args.max_components
    )

    print("=" * 70)
    print("Eigenspectrum inspection")
    print("=" * 70)
    print(f"Mode: {data_label}")
    print(f"Array shape: {model.shape}")
    print(f"rowvar: {args.rowvar}")
    print(f"method: {args.method}")
    print(f"max_components: {args.max_components if args.max_components else 'all'}")
    print(f"eigenspectrum length: {eig_model.shape[0]}")
    print(f"eigenspectrum_mse: {mse:.8e}")
    print("=" * 70)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, eig_target.shape[0] + 1)
    ax.plot(x, eig_target, marker="o", linewidth=1.8, label="Target eigenspectrum")
    ax.plot(x, eig_model, marker="x", linewidth=1.4, label="Model eigenspectrum")
    ax.set_xlabel("Eigenvalue index (sorted)")
    ax.set_ylabel("Eigenvalue")
    ax.set_title(f"Correlation Eigenspectra (MSE={mse:.3e})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Saved plot to: {args.save}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
