"""
make_inputs.py

Generate plaid stimulus inputs for the SSN model recovery tutorial.

Run this script once from the claude_tutorial/ directory before opening
the tutorial notebook:

    cd ssn-v1/claude_tutorial
    python make_inputs.py

Six plaid conditions are created (varying target and mask grating contrasts).
Each stimulus is a full-field plaid of two orthogonal gratings (0 deg and 90 deg).

Outputs are saved to inputs/ as HDF5 files named plaid_C<c>_maskC<mc>.h5.
Each file contains:
    'stimulus' — float64 array, shape (NX, NY, NT, NOri)

NX = NY = 16   (spatial grid derived from network config)
NT = 60        (3000 ms at 50 ms per step)
NOri = 16      (orientation channels, 0–168.75 deg)
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

from ssn_v1 import designStim, SSN_utils

# ---------------------------------------------------------------------------- #
#  Network config (needed for spatial geometry and tuning params)              #
# ---------------------------------------------------------------------------- #

network_config_path = os.path.join(mainDir, 'network', 'config.network.EI512_tutorial.json')
with open(network_config_path, 'r') as f:
    network_config = json.load(f)

scaleXY = network_config['network']['nodes']['spatial_config']['scaleXY']
deltaXY = network_config['network']['nodes']['spatial_config']['deltaXY']
NX = int(scaleXY[0] / deltaXY)
NY = int(scaleXY[1] / deltaXY)
print(f"Spatial grid: NX={NX}, NY={NY}")

# ---------------------------------------------------------------------------- #
#  Stimulus conditions                                                          #
# ---------------------------------------------------------------------------- #

# Each tuple: (target_contrast, mask_contrast)
# Target grating: 0 degrees.  Mask grating: 90 degrees.
CONDITIONS = [
    (0.00, 0.05),   # near-zero target, faint mask
    (0.05, 0.00),   # faint target, no mask
    (1.00, 0.00),   # full-contrast target, no mask
    (0.00, 1.00),   # no target, full-contrast mask
    (0.25, 0.75),   # mixed contrasts
    (0.50, 0.50),   # equal contrasts
]

NORI        = 16         # orientation channels
INPUT_SCALE = 100.0      # scale factor so peak input ~ 100 spk/s

# Temporal envelope: difference of Gaussians giving onset transient + sustained
TEMPORAL_PARAMS = {
    'T': 3000, 't_steps': 50,
    'm1': 170, 's1': 100, 'A1': 1,
    'm2': -200, 's2': 200, 'A2': 1.6,
    'C': 1.0, 't_delay': 100,
}

# ---------------------------------------------------------------------------- #
#  Generate and save                                                            #
# ---------------------------------------------------------------------------- #

os.makedirs(os.path.join(mainDir, 'inputs'), exist_ok=True)

for (tc, mc) in CONDITIONS:
    input_config = {
        'orientation_1': 0.0,
        'orientation_2': 90.0,
        'contrast_1': tc,
        'contrast_2': mc,
        'spatial_frequency': 0.1,
        'center_x': 0,
        'center_y': 0,
        'orientation_channels': np.linspace(0, 180, NORI, endpoint=False).tolist(),
        'mask': {'type': 'none'},
        'temporal': {'type': 'transient_sustained', 'params': TEMPORAL_PARAMS},
    }

    inputs = designStim.generate_plaid_stimulus(
        input_config,
        network_config['network'],
        seed=42,
        tuning_func=SSN_utils.von_mises,
    )
    inputs = inputs * INPUT_SCALE

    fname = f'plaid_C{tc:.2f}_maskC{mc:.2f}.h5'
    fpath = os.path.join(mainDir, 'inputs', fname)

    with h5py.File(fpath, 'w') as hf:
        hf.create_dataset('stimulus', data=inputs)
        hf.attrs['target_contrast'] = tc
        hf.attrs['mask_contrast']   = mc

    print(f"  saved {fname}  shape={inputs.shape}  "
          f"min={inputs.min():.2f}  max={inputs.max():.2f}")

print("\nDone. Stimuli saved to inputs/.")
