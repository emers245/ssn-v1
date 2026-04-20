""" Make Model Inputs """

#%% Add paths and imports
# Assumes location of this file in working directory
import os, sys
#os.chdir("model_sandbox/SSN/EI512_trainingTest")
mainDir = os.path.abspath('.')
SSNDir = os.path.abspath(os.path.join(mainDir, '..'))
sys.path.insert(1, mainDir)
sys.path.append(SSNDir)
# Imports
import numpy as np
import json
import pandas as pd
from ssn_v1 import SSN_utils
from ssn_v1 import SSN
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import h5py
import paramiko
import matplotlib.pyplot as plt
import tqdm
from mpl_toolkits.mplot3d import Axes3D
import errno
import tempfile
import getpass
from ssn_v1 import designStim
# Check pwd
print(f"Working directory: {os.system('pwd')}")

#%% Helper Functions

def push_remote_file(local_path, remote_path, username, password, server):
    """Push a local file to a remote server using SFTP."""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(server, username=username, password=password)
        sftp = ssh.open_sftp()
        sftp.put(local_path, remote_path)
        sftp.close()
        ssh.close()
        print(f"File pushed from {local_path} to {server}:{remote_path}")
    except Exception as e:
        print(f"Failed to push file to remote server: {e}")
        raise

def make_temp_inputs(inputs, input_config):
    """Create a temporary HDF5 file with the given inputs."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        temp_path = tmp_file.name
        with h5py.File(temp_path, 'w') as hf:
            hf.create_dataset('stimulus', data=inputs)
    return temp_path

def make_temp_inputConfig(input_config):
    """Create a temporary JSON file with the given input configuration."""
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as tmp_file:
        temp_path = tmp_file.name
        json.dump(input_config, tmp_file)
    return temp_path

def format_value(val, decimals=2):
    if decimals > 0:
        scaled = int(round((val % 1) * (10 ** decimals)))
        return f"{int(val):.0f}p{scaled:0{decimals}d}"
    elif decimals == 0:
        return f"{int(round(val)):.0f}"
    else:
        raise ValueError("decimals must be a non-negative integer.")

#%% Create Inputs

# Save Remotely?
save_remotely = False

# Load Network Config
network_config_path = os.path.join(mainDir, 'network', 'config.network.EI512_trainingTest3.json')
with open(network_config_path, 'r') as f:
    network_config = json.load(f)

# Set up spatial geometry
scaleXY = network_config['network']['nodes']['spatial_config']["scaleXY"]
scaleX = scaleXY[0]
scaleY = scaleXY[1]
degrees_per_pixel = network_config['network']['nodes']['spatial_config']["deltaXY"]
NX, NY = (int(scaleX // degrees_per_pixel), int(scaleY // degrees_per_pixel))

#%% Set username and password for server

username = 'None' #input("Enter username for server: ")
password = 'None' #getpass.getpass("Enter password for server: ")

#%% Load network so we can generate currents files

V1net = SSN("EI512_trainingTest3_mapSeed42_netSeed42")
V1net.load_config(os.path.join(mainDir, 'config.EI512_trainingTest3_mapSeed42_netSeed42.training_sample_3_FullFieldPlaid_ori0p0_maskori90p0_C0p00_maskC0p05_dur3000.json'), username=username, password=password)
V1net.add_nodes()
V1net.add_edges()

# Load Network Config
network_config_path = os.path.join(mainDir, 'network', 'config.network.EI512_trainingTest3.json')
with open(network_config_path, 'r') as f:
    network_config = json.load(f)

# Set up spatial geometry
scaleXY = network_config['network']['nodes']['spatial_config']["scaleXY"]
scaleX = scaleXY[0]
scaleY = scaleXY[1]
degrees_per_pixel = network_config['network']['nodes']['spatial_config']["deltaXY"]
NX, NY = (int(scaleX // degrees_per_pixel), int(scaleY // degrees_per_pixel))

V1net = SSN("EI512_trainingTest3_mapSeed42_netSeed42")
V1net.load_config(os.path.join(mainDir, 'config.EI512_trainingTest3_mapSeed42_netSeed42.training_sample_3_FullFieldPlaid_ori0p0_maskori90p0_C0p00_maskC0p05_dur3000.json'), username=username, password=password)
V1net.add_nodes()
V1net.add_edges()

# Make Plaids
target_contrast = np.array([0.0, 0.05, 1.0, 0.0, 0.25, 0.5]).tolist()
mask_contrast = np.array([0.05, 0.0, 0.0, 1.0, 0.75, 0.5]).tolist()
target_orientations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] #np.linspace(0, 180, 8, endpoint=False).tolist()
mask_orientations = [90.0, 90.0, 90.0, 90.0, 90.0, 90.0] #np.linspace(0, 180, 8, endpoint=False).tolist()
input_scale = 100

# Input Config

input_config = {
    "file_name": "FullFieldPlaid_ori0p0_maskori90p0_C0p50_maskC0p50_dur3000.h5",
    "stimulus": {
        "orientation_1": 0.0,  # degrees for first grating
        "orientation_2": 90.0,  # degrees for second grating
        "contrast_1": 0.5,  # contrast for first grating
        "contrast_2": 0.5,  # contrast for second grating
        "spatial_frequency": 0.1,  # cycles per degree (currently unused)
        "center_x": 0,  # stimulus center in degrees
        "center_y": 0,
        "orientation_channels": np.linspace(0, 180, 16, endpoint=False).tolist(),  # orientation channels in the network
        "mask": {
            "type": "none"
        },
        "stim_type": "current_field",
        "temporal": {
            "type": "transient_sustained",  # or "rectangular"
            "params": {
                "T": 3000, "t_steps": 50, "m1": 170, "s1": 100,
                "A1": 1, "m2": -200, "s2": 200, "A2": 1.6, "C": 1.0, "t_delay": 100
            }
        }
    },
    "model": {
        "model_type": "designStim",
        "seed": 42
    }
}

# Main config template
template_path = os.path.join(mainDir, 'config.EI512_trainingTest3_mapSeed42_netSeed42.training_sample_3_FullFieldPlaid_ori0p0_maskori90p0_C1p00_maskC0p00_dur3000.json')
V1net.load_config(template_path, username=username, password=password)
with open(template_path, 'r') as f:
    main_config_template = json.load(f)

total_iters = len(target_contrast)
with tqdm.tqdm(total=total_iters, desc="Stimuli") as progress:
    for plaid_i in range(len(target_contrast)):
        print(f"Generating stimulus with contrast {target_contrast[plaid_i]}, mask contrast {mask_contrast[plaid_i]}, and orientation {target_orientations[plaid_i]}, mask orientation {mask_orientations[plaid_i]}")
        input_config['stimulus']['contrast_1'] = target_contrast[plaid_i]
        input_config['stimulus']['contrast_2'] = mask_contrast[plaid_i]
        input_config['stimulus']['orientation_1'] = target_orientations[plaid_i]
        input_config['stimulus']['orientation_2'] = mask_orientations[plaid_i]
        input_config['file_name'] = (
            f"FullFieldPlaid"
            f"_ori{format_value(target_orientations[plaid_i], decimals=1)}"
            f"_maskori{format_value(mask_orientations[plaid_i], decimals=1)}"
            f"_C{format_value(target_contrast[plaid_i], decimals=2)}"
            f"_maskC{format_value(mask_contrast[plaid_i], decimals=2)}"
            f"_dur{format_value(input_config['stimulus']['temporal']['params']['T'], decimals=0)}.h5"
        )
        inputs = designStim.generate_plaid_stimulus(input_config['stimulus'], network_config['network'], seed=input_config['model']['seed'], tuning_func=SSN_utils.von_mises)
        print(f"Generated stimulus with shape: {inputs.shape}")

        # Applying appropriate scale
        inputs = inputs * input_scale

        print("Min and Max of Scaled Inputs:", np.min(inputs), np.max(inputs))

        # Save Current Field to HDF5 file

        # Get file paths
        base_dir = V1net.manifest['$BASE_DIR']
        input_dir = V1net.manifest['$INPUTS_DIR'].replace('$BASE_DIR', base_dir)

        # Save currents
        
        # Make currents
        currents_input_config = input_config.copy()
        V1net.stim_params['stim_type'] = 'current_field'
        currents_input_config['file_name'] = f"{currents_input_config['file_name']}"
        V1net.inputs = inputs #set inputs directly
        V1net.connect_inputs() #connect inputs to generate currents
        currents_input_config['stimulus']['stim_type'] = 'currents' #now change type to currents for saving

        # Get file paths
        currents_path = os.path.join(mainDir, 'inputs', 'training_sample_3', f"{currents_input_config['file_name']}")
        currents_config_path = os.path.join(mainDir, 'inputs', 'training_sample_3', f"config.inputs.{currents_input_config['file_name'].replace('.h5', '.json')}")

        # Save locally
        with h5py.File(currents_path, 'w') as hf:
            hf.create_dataset('stimulus', data=V1net.h)
        with open(currents_config_path, 'w') as f:
            json.dump(currents_input_config, f)

        # Make local main config file for this stimulus

        # Get path
        main_config_path = os.path.join(base_dir, f"config.{V1net.name}.training_sample_3_{input_config['file_name'].replace('.h5', '.json')}")

        # Make changes to main config template
        main_config = main_config_template['inputs']['file_name'] = f"$INPUTS_DIR/{input_config['file_name']}"
        main_config_template['inputs']['inputs_config'] = f"$INPUTS_DIR/config.inputs.{input_config['file_name'].replace('.h5', '.json')}"
        main_config_template['manifest']['$INPUTS_DIR'] = "$BASE_DIR/inputs/training_sample_3"
        main_config_template['outputs']['rates_h5'] = f"rates.{V1net.name}.{input_config['file_name'].replace('.h5', '.h5')}"
        main_config_template['outputs']['outputs_dir'] = os.path.join("$BASE_DIR", 'outputs', 'training_sample_3')

        # Save locally
        with open(main_config_path, 'w') as f:
            json.dump(main_config_template, f)

        progress.update(1)

# %% Visualize

# Plot histogram of input strengths
plt.figure()
plt.hist(V1net.h[:,-1], bins=50)
plt.xlabel('Input Strength')
plt.ylabel('Number of Cells')
plt.title('Histogram of Input Strengths with Uniform Tuning Functions')
plt.show()

# Plot input strengths vs orientation preference
plt.figure()
plt.scatter(V1net.nodes['orientation'], V1net.h[:,-1], alpha=0.5)
plt.xlabel('Orientation Preference (radians)')
plt.ylabel('Input Strength')
plt.title('Input Strength vs Orientation Preference')
plt.show()

