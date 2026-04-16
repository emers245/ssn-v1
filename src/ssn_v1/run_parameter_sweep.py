#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run a single-parameter SSN sweep and save raw model outputs.

This script intentionally does NOT compute any costs. It runs model simulations
for a 1D parameter sweep and stores each simulated model output to disk,
along with sweep metadata for downstream analysis notebooks.

Example:
    python -m ssn_v1.run_parameter_sweep \
        --config config.json \
        --inputs-dir inputs/ \
        --parameter sigmajEE \
        --sweep-min 0.0 \
        --sweep-max 1.0 \
        --n-steps 21 \
        --n-inst 4 \
        --n-cores 8 \
        --output-dir SSN/EI512_trainingTest/param_sweep \
        --output-type end \

Arguments:
    --config: Path to SSN config JSON file.
    --inputs-dir: Directory containing input HDF5 files.
    --input-files: Optional explicit list of input file names relative to --inputs-dir.
    --input-dataset: Input dataset name in each input HDF5 file (default: "stimulus").
    --parameter: Parameter name to sweep (must exist in param map).
    --param-map: Optional JSON file with parameter map override.
    --sweep-min: Minimum sweep value.
    --sweep-max: Maximum sweep value.
    --n-steps: Number of sweep values, inclusive endpoints.
    --n-inst: Number of stochastic instances per sweep value (default: 1).
    --seed: Master random seed (default: 42).
    --fix-seed: Reuse the same seed for all instances (default: False).
    --fix-nodes: Fix spatial node layout across runs (default: False).
    --node-seed: Node map seed when --fix-nodes is set (default: 0).
    --scale: Input scale; one value or one per input condition (default: [1.0]).
    --output-type: Output representation to save via SSN.save_outputs: 'all', 'end', or 'start,end' (time-window mean) (default: "all").
    --output-dir: Directory to store sweep outputs and metadata (default: "param_sweep").
    --append-run-dir: Existing run directory to append into. Requires existing sweep_metadata.json (default: None).
    --on-existing: Collision policy for output files in append mode. Defaults to 'skip' in append mode and 'error' otherwise (choices: "skip", "error", "overwrite").
    --n-cores: Number of CPU cores for parallel execution (default: 1).
"""

from __future__ import annotations

import argparse # for CLI argument parsing
import json
import multiprocessing as mp # for parallel processing
import os
import re # for regex matching of existing output files in append mode
import traceback # for tracing errors
from collections import OrderedDict # for maintaining input condition order in metadata
from datetime import datetime
from pathlib import Path # for filesystem path handling
from time import perf_counter
from typing import Dict, List, Mapping, Sequence # for type annotations

import numpy as np
from tqdm import tqdm

# Add ssn_v1 and functions from run_optimization to path for imports
try:
    from .SSN import SSN
    from .run_optimization import (
        DEFAULT_PARAM_MAP,
        build_network,
        run_simulation,
        resolve_file_list,
        load_input_data,
        broadcast_scale,
    )
# If relative imports fail (e.g. when run as __main__), fall back to absolute imports
except ImportError:
    from ssn_v1.SSN import SSN
    from ssn_v1.run_optimization import (
        DEFAULT_PARAM_MAP,
        build_network,
        run_simulation,
        resolve_file_list,
        load_input_data,
        broadcast_scale,
    )

# Initialize global worker config for multiprocessing workers; will be set by initializer function before any tasks are run
_WORKER_CFG = {}


def _to_builtin(obj):
    """Recursively convert an object to built-in Python types for JSON serialization."""
    if isinstance(obj, dict): # if dict, convert keys to strings and values recursively
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): # if list or tuple, convert each element recursively
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.generic): # if numpy scalar, convert to native Python scalar
        return obj.item()
    if isinstance(obj, np.ndarray): # if numpy array, convert to list
        return obj.tolist()
    return obj


def _load_param_map(path: str | None) -> Mapping[str, Sequence[str]]:
    """Load a parameter map from a JSON file or return the default map."""
    if path is None: # if no param map specified, use default
        return DEFAULT_PARAM_MAP
    with open(path, "r") as f: # load user-specified param map from JSON file
        user_map = json.load(f)
    if not isinstance(user_map, dict): # if loaded JSON is not a dict, raise error
        raise ValueError("--param-map must be a JSON object mapping parameter names to path sequences")
    normalized = {}
    # dictionary must be in correct format
    for key, value in user_map.items():
        if not isinstance(value, (list, tuple)) or len(value) == 0:
            raise ValueError(f"Invalid path for parameter '{key}' in param map")
        normalized[key] = tuple(value)
    return normalized


def _get_nested(dct: Mapping, path: Sequence[str]):
    """Retrieve a nested value from a dictionary given a sequence of keys."""
    cur = dct
    for key in path:
        cur = cur[key]
    return cur


def _extract_base_theta_from_model_parameters(config_file: str, param_map: Mapping[str, Sequence[str]]) -> np.ndarray:
    """Extract the base theta vector from the model parameters in the config file using the param map."""
    ssn = SSN("ei_model", verbose=False)
    ssn.load_config(config_file) # loads parameters into ssn.parameters

    theta = []
    for _, ppath in param_map.items():
        value = _get_nested(ssn.parameters, ppath)
        theta.append(float(value))

    return np.asarray(theta, dtype=float)


def _init_worker(worker_cfg: Dict):
    """Initialize the global worker configuration for multiprocessing workers."""
    global _WORKER_CFG
    _WORKER_CFG = worker_cfg


def _parse_output_type_arg(output_type_raw: str):
    """Parse the output type argument and return a normalized representation."""
    if output_type_raw in {"all", "end"}: # if output type is 'all' or 'end', return as-is
        return output_type_raw
    parts = [p.strip() for p in str(output_type_raw).split(",")] # if start and end specified, parse as floats
    if len(parts) != 2:
        raise ValueError("--output-type must be 'all', 'end', or 'start,end' (two floats)")
    try:
        t_start = float(parts[0])
        t_end = float(parts[1])
    except ValueError as exc:
        raise ValueError("--output-type window must contain two float values: 'start,end'") from exc
    return [t_start, t_end]


def _normalize_output_type(output_type):
    """Normalize the output type for comparison in append mode metadata checks."""
    if isinstance(output_type, str):
        return output_type
    if isinstance(output_type, (list, tuple)) and len(output_type) == 2: # if list or tuple, interpret as start and end floats
        return [float(output_type[0]), float(output_type[1])]
    return output_type


def _load_existing_metadata(metadata_path: Path) -> Dict:
    """Load existing metadata from a JSON file for append mode, with error handling."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Append mode requires existing metadata file: {metadata_path}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid metadata format in {metadata_path}; expected JSON object")
    return metadata


def _write_json_atomic(path: Path, payload: Dict) -> None:
    """Write a JSON file atomically by first writing to a temporary file and then renaming it."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(_to_builtin(payload), f, indent=2)
    os.replace(tmp_path, path)


def _build_append_mismatch_report(existing_metadata: Mapping, current_signature: Mapping) -> List[str]:
    """Compare existing metadata with current run signature and return a list of mismatch descriptions."""
    mismatches = []
    for key, current_value in current_signature.items():
        existing_value = existing_metadata.get(key, None)
        if key == "output_type":
            existing_value = _normalize_output_type(existing_value)
        if existing_value != current_value:
            mismatches.append(f"{key}: existing={existing_value!r}, current={current_value!r}")
    return mismatches


def _next_sweep_index_offset(existing_tasks: Sequence[Mapping], parameter_name: str) -> int:
    """Determine the next sweep index offset based on existing tasks and parameter name."""
    patt = re.compile(rf"{re.escape(parameter_name)}_(\d+)_inst_\d+\.h5$")
    max_index = -1
    for task in existing_tasks:
        out_file = str(task.get("output_file", ""))
        match = patt.search(Path(out_file).name)
        if match:
            max_index = max(max_index, int(match.group(1)))
    return max_index + 1


def _task_outputs_exist(base_output_file: str, condition_names: Sequence[str]) -> bool:
    """Check if the output files for a given task already exist, including condition-specific files."""
    base_path = Path(base_output_file)
    if base_path.exists():
        return True
    for cond_name in condition_names:
        cond_path = Path(base_output_file.replace(".h5", f"__{cond_name}.h5"))
        if cond_path.exists():
            return True
    return False



def _run_single_task(task: Dict) -> Dict:
    """Run a single simulation task with the specified parameter value and return the result."""
    t0 = perf_counter() # initial timestamp for measuring elapsed time of this task
    out_file = task["output_file"]
    sweep_value = float(task["sweep_value"])
    param_index = int(task["param_index"])
    seed = int(task["seed"])
    node_seed = int(task["node_seed"])
    scale = _WORKER_CFG["scale"]
    output_type = _WORKER_CFG["output_type"]

    theta = _WORKER_CFG["base_theta"].copy()
    theta[param_index] = sweep_value

    # try building the network and running the simulation
    try:
        ssn_model = build_network(
            _WORKER_CFG["config_file"],
            theta,
            _WORKER_CFG["param_map"],
            seed=seed,
            node_seed=node_seed,
            verbose=_WORKER_CFG["verbose"],
        )

        saved_files = []
        for cidx, (cond_name, inp) in enumerate(_WORKER_CFG["input_data"].items()):
            ssn_model = run_simulation(ssn_model, inp, scale=scale[cidx], events=None)
            cond_output_file = out_file.replace(".h5", f"__{cond_name}.h5")
            ssn_model.save_outputs(output_path=cond_output_file, local=True, output_type=output_type)
            saved_files.append(cond_output_file)

        return {
            "ok": True,
            "output_file": out_file,
            "saved_files": saved_files,
            "parameter_value": sweep_value,
            "seed": seed,
            "node_seed": node_seed,
            "elapsed_sec": perf_counter() - t0,
            "error": None,
        }
    # if unsuccessful, return the error information
    except Exception:
        return {
            "ok": False,
            "output_file": out_file,
            "saved_files": [],
            "parameter_value": sweep_value,
            "seed": seed,
            "node_seed": node_seed,
            "elapsed_sec": perf_counter() - t0,
            "error": traceback.format_exc(),
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the parameter sweep script."""
    parser = argparse.ArgumentParser(description="Run SSN single-parameter sweep and save raw outputs.")
    parser.add_argument("--config", required=True, help="Path to SSN config JSON file.")
    parser.add_argument("--inputs-dir", required=True, help="Directory containing input HDF5 files.")
    parser.add_argument("--input-files", nargs="*", default=None, help="Optional explicit list of input file names relative to --inputs-dir.")
    parser.add_argument("--input-dataset", default="stimulus", help="Input dataset name in each input HDF5 file.")

    parser.add_argument("--parameter", required=True, help="Parameter name to sweep (must exist in param map).")
    parser.add_argument("--param-map", default=None, help="Optional JSON file with parameter map override.")
    parser.add_argument("--sweep-min", type=float, required=True, help="Minimum sweep value.")
    parser.add_argument("--sweep-max", type=float, required=True, help="Maximum sweep value.")
    parser.add_argument("--n-steps", type=int, required=True, help="Number of sweep values, inclusive endpoints.")

    parser.add_argument("--n-inst", type=int, default=1, help="Number of stochastic instances per sweep value.")
    parser.add_argument("--seed", type=int, default=42, help="Master random seed.")
    parser.add_argument("--fix-seed", action="store_true", help="Reuse the same seed for all instances.")
    parser.add_argument("--fix-nodes", action="store_true", help="Fix spatial node layout across runs.")
    parser.add_argument("--node-seed", type=int, default=0, help="Node map seed when --fix-nodes is set.")

    parser.add_argument("--scale", nargs="*", type=float, default=[1.0], help="Input scale; one value or one per input condition.")
    parser.add_argument("--output-type", default="all", help="Output representation to save via SSN.save_outputs: 'all', 'end', or 'start,end' (time-window mean).")

    parser.add_argument("--output-dir", default="param_sweep", help="Directory to store sweep outputs and metadata.")
    parser.add_argument("--append-run-dir", default=None, help="Existing run directory to append into. Requires existing sweep_metadata.json.")
    parser.add_argument(
        "--on-existing",
        choices=["skip", "error", "overwrite"],
        default=None,
        help="Collision policy for output files in append mode. Defaults to 'skip' in append mode and 'error' otherwise.",
    )
    parser.add_argument("--force-append", action="store_true", help="Allow append even if metadata compatibility checks fail.")
    parser.add_argument("--tag", default=None, help="Optional run tag to include in output folder name.")
    parser.add_argument("--n-cores", type=int, default=1, help="Number of CPU cores for parallel execution.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose SSN simulation output.")

    return parser.parse_args()


def main() -> None:
    """ Main script """
    args = parse_args() # get CLI arguments

    if args.n_steps < 2: #must have at least 2 steps in sweep
        raise ValueError("--n-steps must be >= 2")
    if args.n_inst < 1: # must have at least one model instance per parameter value
        raise ValueError("--n-inst must be >= 1")
    if args.n_cores < 1: # must use at last one CPU core
        raise ValueError("--n-cores must be >= 1")

    config_path = Path(args.config).resolve() # turn relative paths into absolute path
    inputs_dir = Path(args.inputs_dir).resolve()
    out_root = Path(args.output_dir).resolve()

    # Error handling for missing files
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not inputs_dir.exists():
        raise FileNotFoundError(f"Inputs directory not found: {inputs_dir}")

    # Load parameter map
    param_map = _load_param_map(args.param_map)
    if args.parameter not in param_map:
        available = ", ".join(sorted(param_map.keys()))
        raise KeyError(f"Parameter '{args.parameter}' not in param map. Available: {available}")

    # Get input paths
    input_paths = resolve_file_list(str(inputs_dir), args.input_files)
    if len(input_paths) == 0:
        raise ValueError(f"No input files found in {inputs_dir}")

    # Load input data
    input_data, input_labels = load_input_data(input_paths, dataset=args.input_dataset)
    if not isinstance(input_data, OrderedDict):
        input_data = OrderedDict(input_data)

    scale = broadcast_scale(args.scale, len(input_data)) # set scale for inputs
    output_type = _parse_output_type_arg(args.output_type) # get output type
    condition_names = list(input_data.keys()) # get condition names for each input

    param_names = list(param_map.keys()) # get parameter names
    param_index = param_names.index(args.parameter) # get index of parameter to sweep in theta vector
    base_theta = _extract_base_theta_from_model_parameters(str(config_path), param_map) # get the base parameter values

    # Construct parameter sweep values
    sweep_values = np.linspace(args.sweep_min, args.sweep_max, args.n_steps, dtype=float)

    # Start sweep setup and execution
    append_mode = args.append_run_dir is not None # check for append mode
    on_existing = args.on_existing or ("skip" if append_mode else "error") # set collision policy for existing outputs

    # initialize variables for metadata handling in append mode
    existing_metadata = None
    existing_tasks = []
    metadata_path = None

    # set up for apppend mode
    if append_mode:
        run_dir = Path(args.append_run_dir).resolve() # location of existing run to append to
        outputs_dir = run_dir / "outputs" # location of HDF5 output files
        outputs_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = run_dir / "sweep_metadata.json" # path to existing metadata
        existing_metadata = _load_existing_metadata(metadata_path) # load metadata

        # Construct dictionary based on current parameters to compare to existing metadata
        current_signature = {
            "config": str(config_path),
            "inputs_dir": str(inputs_dir),
            "input_files": [str(p) for p in input_paths],
            "input_labels": input_labels,
            "input_dataset": args.input_dataset,
            "param_map": _to_builtin(param_map),
            "scale": [float(v) for v in scale],
            "output_type": _normalize_output_type(output_type),
            "fix_nodes": bool(args.fix_nodes),
        }
        if args.fix_nodes:
            current_signature["node_seed"] = int(args.node_seed)
        # Find mismatches between current command and metadata
        mismatches = _build_append_mismatch_report(existing_metadata, current_signature)
        if mismatches and not args.force_append:
            mismatch_msg = "\n  - " + "\n  - ".join(mismatches) # Report mismatches if foun
            raise ValueError(
                "Append compatibility checks failed. Use --force-append to override mismatches:" + mismatch_msg
            )
        if mismatches and args.force_append:
            print("WARNING: Forcing append despite metadata mismatches:") # If force_append, print mismatches but do not raise error
            for mm in mismatches:
                print(f"  - {mm}")

        existing_tasks = list(existing_metadata.get("tasks", [])) # load existing tasks
        sweep_index_offset = _next_sweep_index_offset(existing_tasks, args.parameter) #determine the next index based on the existing tasks and the new parameters
    # Set up in normal mode
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = args.tag or args.parameter
        run_dir = out_root / f"{tag}_{timestamp}"
        outputs_dir = run_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        sweep_index_offset = 0
        metadata_path = run_dir / "sweep_metadata.json"

    rng = np.random.default_rng(args.seed) # set rng with seed
    tasks = [] # list to store tasks
    skipped_existing = 0 # counter for skipped existing tasks
    for sval_idx, sweep_value in enumerate(sweep_values):
        sweep_idx = int(sweep_index_offset + sval_idx)
        for inst_idx in range(args.n_inst):
            if args.fix_seed:
                seed = int(args.seed)
            else:
                seed = int(rng.integers(0, 2_147_483_647))
            if args.fix_nodes:
                node_seed = int(args.node_seed)
            else:
                node_seed = int(rng.integers(0, 2_147_483_647))
            model_name = f"{args.parameter}_{sweep_idx:04d}_inst_{inst_idx:04d}.h5"
            output_file = str(outputs_dir / model_name)
            
            # Handle existing outputs conflicts
            output_exists = _task_outputs_exist(output_file, condition_names)
            if output_exists and on_existing == "error":
                raise FileExistsError(
                    f"Output collision for task {model_name}. Use --on-existing skip or overwrite."
                )
            if output_exists and on_existing == "skip":
                skipped_existing += 1
                continue

            tasks.append(
                {
                    "sweep_index": int(sweep_idx),
                    "instance_index": int(inst_idx),
                    "parameter": str(args.parameter),
                    "sweep_value": float(sweep_value),
                    "param_index": int(param_index),
                    "seed": seed,
                    "node_seed": node_seed,
                    "output_file": output_file,
                }
            )

    # Set worker configuration
    worker_cfg = {
        "config_file": str(config_path),
        "param_map": param_map,
        "param_names": param_names,
        "parameter_name": args.parameter,
        "base_theta": base_theta,
        "input_data": input_data,
        "scale": scale,
        "output_type": output_type,
        "verbose": bool(args.verbose),
    }

    n_workers = min(args.n_cores, len(tasks), os.cpu_count() or 1) if tasks else 1
    t0 = perf_counter()
    pbar_desc = f"Sweep {args.parameter}"

    # if no tasks to be run, return empty list
    if len(tasks) == 0:
        results = []
    # Run on one CPU without multiprocessing
    elif n_workers <= 1:
        _init_worker(worker_cfg)
        results = []
        with tqdm(total=len(tasks), desc=pbar_desc, unit="run") as pbar:
            for task in tasks:
                result = _run_single_task(task)
                results.append(result)
                pbar.set_postfix_str(f"{args.parameter}={result['parameter_value']:.6g}")
                pbar.update(1)
    # Run parallel processing
    else:
        with mp.Pool(processes=n_workers, initializer=_init_worker, initargs=(worker_cfg,)) as pool:
            results = []
            with tqdm(total=len(tasks), desc=pbar_desc, unit="run") as pbar:
                for result in pool.imap_unordered(_run_single_task, tasks):
                    results.append(result)
                    pbar.set_postfix_str(f"{args.parameter}={result['parameter_value']:.6g}")
                    pbar.update(1)

    elapsed = perf_counter() - t0

    n_ok = int(sum(r["ok"] for r in results))
    n_fail = int(len(results) - n_ok)

    # Record appended metadata
    append_record = {
        "timestamp": datetime.now().isoformat(),
        "append_mode": bool(append_mode),
        "parameter": str(args.parameter),
        "sweep_min": float(args.sweep_min),
        "sweep_max": float(args.sweep_max),
        "n_steps": int(args.n_steps),
        "n_inst": int(args.n_inst),
        "seed": int(args.seed),
        "fix_seed": bool(args.fix_seed),
        "on_existing": str(on_existing),
        "n_requested_tasks": int(args.n_steps * args.n_inst),
        "n_skipped_existing": int(skipped_existing),
        "n_executed_tasks": int(len(tasks)),
        "n_success": int(n_ok),
        "n_failed": int(n_fail),
        "elapsed_sec": float(elapsed),
        "force_append": bool(args.force_append),
    }

    # If in append mode, add append history to the metadata and merge tasks
    if append_mode:
        merged_tasks = existing_tasks + results
        metadata = dict(existing_metadata)
        history = list(metadata.get("history", []))
        history.append(append_record)
        metadata["history"] = history
        metadata["updated_at"] = datetime.now().isoformat()
        metadata["last_parameter"] = str(args.parameter)
        metadata["last_n_cores"] = int(n_workers)
        metadata["last_elapsed_sec"] = float(elapsed)
        metadata["tasks"] = merged_tasks
        metadata["n_tasks"] = int(len(merged_tasks))
        metadata["n_success"] = int(sum(1 for r in merged_tasks if r.get("ok")))
        metadata["n_failed"] = int(sum(1 for r in merged_tasks if not r.get("ok")))
        metadata["elapsed_sec"] = float(metadata.get("elapsed_sec", 0.0)) + float(elapsed)

        backup_path = run_dir / "sweep_metadata.prev.json"
        with open(backup_path, "w") as f:
            json.dump(_to_builtin(existing_metadata), f, indent=2)
    # Otherwise, write new matadata from scratch
    else:
        metadata = {
            "script": "run_parameter_sweep.py",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "config": str(config_path),
            "inputs_dir": str(inputs_dir),
            "input_files": [str(p) for p in input_paths],
            "input_labels": input_labels,
            "input_dataset": args.input_dataset,
            "output_dir": str(run_dir),
            "outputs_dir": str(outputs_dir),
            "parameter": args.parameter,
            "param_map": _to_builtin(param_map),
            "base_params": {name: float(base_theta[i]) for i, name in enumerate(param_names)},
            "sweep_min": float(args.sweep_min),
            "sweep_max": float(args.sweep_max),
            "n_steps": int(args.n_steps),
            "sweep_values": sweep_values.tolist(),
            "n_inst": int(args.n_inst),
            "seed": int(args.seed),
            "fix_seed": bool(args.fix_seed),
            "fix_nodes": bool(args.fix_nodes),
            "node_seed": int(args.node_seed),
            "scale": [float(v) for v in scale],
            "output_type": output_type,
            "n_cores": int(n_workers),
            "elapsed_sec": float(elapsed),
            "n_tasks": int(len(results)),
            "n_success": n_ok,
            "n_failed": n_fail,
            "tasks": results,
            "history": [append_record],
        }

    _write_json_atomic(metadata_path, metadata)

    # Track failed tasks separately
    all_failed = [r for r in metadata.get("tasks", []) if not r.get("ok")]
    if len(all_failed) > 0:
        failed_path = run_dir / "failed_tasks.json"
        _write_json_atomic(failed_path, all_failed)

    print(
        f"Sweep complete: {n_ok}/{len(tasks)} successful, {n_fail} failed"
        f" ({skipped_existing} skipped existing)"
    )
    print(f"Outputs: {outputs_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
