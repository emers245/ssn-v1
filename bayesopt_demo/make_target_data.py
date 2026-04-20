""" Make target data """

#%% Set up

mainDir = '.'
SSNDir = '../' #path to directory with SSN code

import sys
sys.path.insert(1, mainDir)
sys.path.append(SSNDir)

import numpy as np
import json
import pandas as pd
import os
from ssn_v1 import SSN_utils
from ssn_v1 import SSN
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import h5py
import paramiko
import matplotlib.pyplot as plt
import tqdm
from mpl_toolkits.mplot3d import Axes3D
import h5py
import errno
import paramiko
import tempfile
import glob

#%% Set username and password for remote server

import getpass

outputs_local = False #if false save to server
if outputs_local == False:
    server = 'None'
    username = 'None'
    password = 'None'

# %% training sample 3 : plaid stimuli New network

experiment_path = os.path.join(mainDir,'inputs','training_sample_3')
target_data_path = os.path.join(mainDir,'outputs','training_sample_3')
if not os.path.exists(target_data_path):
    os.makedirs(target_data_path)
# Get all HDF5 file names in the experiment path
file_names = glob.glob(os.path.join(experiment_path, '*.h5'))
stimulus_names = [os.path.splitext(os.path.basename(f))[0] for f in file_names]
model_name = 'EI512_trainingTest3_mapSeed42_netSeed42'
config_dir = f'.'

eigspectra = {}
W_cached = None

# User progress bar to track stimulus processing
for stimulus_name in tqdm.tqdm(stimulus_names):

    # Initialize Network
    config = os.path.join(config_dir, 'config.'+model_name+'.'+'training_sample_3_'+stimulus_name+'.json')
    overwrite_outputs = False
    V1net = SSN("ei_model")
    V1net.load_config(os.path.join(mainDir,config))
    print(f"Loading {config}")

    # Load or Build Network
    model_file = os.path.join(mainDir,model_name+'.joblib')
    if os.path.exists(model_file):
        V1net = V1net.load(model_file)
    else:
        print(f"{model_file} not found. Constructing network from config file.")
        
        # Add Nodes
        V1net.add_nodes()
        
        # Add edges
        V1net.add_edges()

    # Load Stimulus
    V1net.load_inputs()
    V1net.connect_inputs()

    # Run Simulation
    V1net.run()

    # Save Target Data
    rates_h5 = V1net.outputs_config['rates_h5']
    V1net.save_outputs(os.path.join(target_data_path, rates_h5))

    print(f"Saved target data for {stimulus_name} to {target_data_path}")

# %%
