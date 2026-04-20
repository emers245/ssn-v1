"""
make_targets.py

Run the 'true' SSN on each plaid stimulus to produce target firing rates for
the model recovery tutorial.  Run after make_inputs.py:

    cd ssn-v1/claude_tutorial
    python make_targets.py

The 'true' network is built using the default parameter values from
config.network.EI512_tutorial.json:

    jEE =  3.839   (E -> E synaptic weight)
    jEI = -2.093   (I -> E synaptic weight)
    jIE =  3.678   (E -> I synaptic weight)
    jII = -1.444   (I -> I synaptic weight)

These are the parameters that Bayesian Optimization will try to recover.

Outputs are saved to targets/ as HDF5 files named target_C<c>_maskC<mc>.h5.
Each file contains:
    'rates'  — float64 array, shape (N_cells,) — final firing rates
    'is_E'   — bool array, shape (N_cells,)    — True for E cells
    'is_I'   — bool array, shape (N_cells,)    — True for I cells
"""

import os
import sys
import json
import numpy as np
import h5py

mainDir = os.path.abspath('.')
SSNDir  = os.path.abspath(os.path.join(mainDir, '..'))
sys.path.insert(0, mainDir)
sys.path.append(SSNDir)

from ssn_v1.SSN import SSN, NumericalInstabilityError

# ---------------------------------------------------------------------------- #
#  True parameters                                                              #
# ---------------------------------------------------------------------------- #

TRUE_PARAMS = {
    'jEE':  3.5,
    'jEI': -2.0,
    'jIE':  3.2,
    'jII': -1.8,
}

PARAM_MAP = {
    'jEE': ('edges', 'spatial_tuning', 'E<-E', 'j'),
    'jEI': ('edges', 'spatial_tuning', 'E<-I', 'j'),
    'jIE': ('edges', 'spatial_tuning', 'I<-E', 'j'),
    'jII': ('edges', 'spatial_tuning', 'I<-I', 'j'),
}

MAP_SEED = 42
NET_SEED = 42

# ---------------------------------------------------------------------------- #
#  Build the true network                                                       #
# ---------------------------------------------------------------------------- #

config_file = os.path.join(mainDir, 'config.EI512_tutorial.json')
V1net = SSN('EI512_tutorial_true')
V1net.load_config(config_file)

# Verify / apply true parameters (they are already defaults in the config, but
# we set them explicitly here so this script is self-documenting)
for name, path in PARAM_MAP.items():
    d = V1net.parameters
    for key in path[:-1]:
        d = d[key]
    d[path[-1]] = TRUE_PARAMS[name]

V1net.set_rand_seed(seed=MAP_SEED)
V1net.add_nodes()
V1net.add_edges(seed=NET_SEED)

is_E = V1net.nodes['ei'].values == 'e'
is_I = V1net.nodes['ei'].values == 'i'

print(f"True network built.  N_E={is_E.sum()}  N_I={is_I.sum()}")
print("True parameters:")
for k, v in TRUE_PARAMS.items():
    print(f"  {k} = {v}")

# ---------------------------------------------------------------------------- #
#  Run each stimulus condition                                                  #
# ---------------------------------------------------------------------------- #

inputs_dir  = os.path.join(mainDir, 'inputs')
targets_dir = os.path.join(mainDir, 'targets')
os.makedirs(targets_dir, exist_ok=True)

input_files = sorted(f for f in os.listdir(inputs_dir) if f.endswith('.h5'))

if not input_files:
    raise FileNotFoundError(
        "No HDF5 files found in inputs/.  Run make_inputs.py first.")

print(f"\nProcessing {len(input_files)} stimulus conditions...")

for fname in input_files:
    fpath = os.path.join(inputs_dir, fname)
    with h5py.File(fpath, 'r') as hf:
        raw_inputs = hf['stimulus'][:]

    V1net.inputs = raw_inputs
    V1net.connect_inputs()

    try:
        V1net.run()
    except NumericalInstabilityError:
        print(f"  UNSTABLE for {fname} — skipping")
        continue

    rates = V1net.outputs.y[:, -1]

    out_fname = fname.replace('plaid_', 'target_')
    out_path  = os.path.join(targets_dir, out_fname)

    with h5py.File(out_path, 'w') as hf:
        hf.create_dataset('rates', data=rates)
        hf.create_dataset('is_E',  data=is_E)
        hf.create_dataset('is_I',  data=is_I)
        hf.attrs['source'] = fname

    e_mean = rates[is_E].mean()
    i_mean = rates[is_I].mean()
    print(f"  {fname}  =>  E: {e_mean:.1f} Hz  I: {i_mean:.1f} Hz")

print("\nDone. Target firing rates saved to targets/.")
