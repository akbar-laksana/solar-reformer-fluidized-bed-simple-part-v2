#%% solarReact_RunFiles
#   This sets up the input or restart files to run a solar reformer model
#   by Akbar Laksana, 2025

import numpy as np
import os
import glob
from datetime import datetime

from solarReact_Runner import solarReact_Runner

#%% Find all json files in Input Files folder
def find_json_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".json"):
                filepath = os.path.join(root, file)
                print(filepath)

#%% Define file naming convention and folders
filedate = datetime.now().strftime('%Y_%m_%d')
project  = 'SETO_Reform';

input_path = os.path.join(".",'Data_Files', 'Input_Files')
output_path = os.path.join(".",'Data_Files', 'Output_Files')
infiledate_path = os.path.join(input_path, filedate + '_' + project)
outfiledate_path = os.path.join(output_path, filedate + '_' + project)

#infiledatevar_path = os.path.join(infiledate_path, 'm_g_flux_1.2_q_sol_ap_240-300')
infiledatevar_path = os.path.join(input_path, filedate + '_' + project)

#%% Locate how many files exist to prepare loop for running the solution
files = glob.glob(os.path.join(infiledatevar_path, '*.json'))  # Locate all '.json' files
filenames = [os.path.basename(file) for file in files]  # Extract filenames
n_files = len(filenames)  # Number of files

#%% Set flag for restart from previous solution or overwrite of previous solutions
restart = False * np.ones(n_files)
overwrite   = True         # True: Rerun and overwrite solutions for input file with existing solution file
#overwrite = False         # False: Only generate solutions for input files without existing solution file

#%%  Call file for using non-linear solver to find solution for all of the input file conditions
for i_file in range(n_files):
    infile_path = os.path.join(infiledatevar_path, filenames[i_file])
    solfile_path = os.path.join(outfiledate_path, filenames[i_file].replace(".json",""))

    isExist = os.path.exists(solfile_path)
    if not isExist:
        os.makedirs(solfile_path)

    # Do not generate solution for input file if overwrite is false and solution file for specified input file already exists %
    if overwrite is False and os.path.exist(solfile_path):
        pass
    # Run model with current input file 
    else:
        [gas, gas_surf, part, wall, surf, env, GasParams, PartParams, BedParams, WallParams, FinParams, EnvParams, ind, scale, Result] = solarReact_Runner(infile_path, solfile_path, filenames[i_file], restart[i_file])