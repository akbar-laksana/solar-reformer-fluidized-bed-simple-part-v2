#%%
# Version of the Receiver-Reactor Model
# no FreeBoard Zone
# Detailed Surface Chemistry
# for Solar Methane Reforming

#%% solarReact_Generate_Input_json
#   This generate input file(s) needed to run a solar reformer model in .json format
#   Set values for parameters before the i_file for loop
#   The i_file loop will handle creation of each files based on values and n_files settings
#   by Akbar Laksana, 2025

import numpy as np
import math 
import cantera as ct
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Name file with current date and corresponding project
filedate = datetime.now().strftime('%Y_%m_%d')
project  = 'SETO_Reform'

input_path = os.path.join(".",'Data_Files', 'Input_Files')
output_path = os.path.join(".",'Data_Files', 'Output_Files')
infiledate_path = os.path.join(input_path, filedate + '_' + project)
outfiledate_path = os.path.join(output_path, filedate + '_' + project)

# Check whether the specified path exists or not
isExist = os.path.exists('Data_Files\Input_Files')
isExist2 = os.path.exists('Data_Files\Output_Files')
isExist3 = os.path.exists(infiledate_path)
isExist4 = os.path.exists(outfiledate_path)

if not isExist:
   os.makedirs('Data_Files\Input_Files')
   
if not isExist2:
    os.makedirs('Data_Files\Output_Files')

if not isExist3:
    os.makedirs(infiledate_path)
    
if not isExist4:
    os.makedirs(outfiledate_path)

#%% Define dictionaries
# Params dictionaries
GasParams       = {}
PartParams      = {}
BedParams       = {}
WallParams      = {}
FinParams       = {}
SurfParams      = {}
EnvParams       = {}    

# Cantera object definition
gas  = {}
gas_surf  = {}
part = {}
wall = {}
surf = {}
env  = {}

#%% Set number of files to be generated
n_files         = 1

#%% Set Cantera object yaml files
file = 2
if file == 1:
    GasParams['filename'] = 'Mechanism/sm_Ni_Rakhi.yaml'
elif file == 2:
    GasParams['filename'] = 'Mechanism/sm_Ni_Delgado.yaml' # Best one for mixed methane reforming
elif file == 3:
    GasParams['filename'] = 'Mechanism/sm_Ni_Maier.yaml'

#%% How to handle chemistry, can choose between detailed surface chemistry or simple global mechanism
BedParams['kinetics'] = 1 # 1: Global mechanism (4 reactions, 5 gas), 2: Detailed surface chemistry (52 reaction, 5 gas, 14 surface)

gas['jreac']    = 4
gas['jDMR']     = 0 
gas['jSMR1']    = 1
gas['jSMR2']    = 2
gas['jWGS']     = 3

# If using detailed surface chemistry:
SurfParams['integrate'] = 1 # 1: integrate, 2: do not integrate
SurfParams['delta_t'] = 1.5e0 # [s]

#%% Set the file and id for each of the Cantera object
gas['file']     = GasParams['filename']
if BedParams['kinetics'] == 1:
    gas['id']   = 'reformate-part-nosurf'
elif BedParams['kinetics'] == 2:
    #gas['id']   = 'reformate-part'
    gas['id']   = 'reformate-part-nosurf'
    
gas_surf['id']  = 'reformate-part'

surf['file']    = GasParams['filename']
surf['id']      = 'surf'

wall['file']    = 'Mechanism/CARBO_air.yaml'
wall['id']      = 'Inco470H' #'silicon-carbide'

part['file']    = 'Mechanism/CARBO_air.yaml'
part['id']      = 'Alumina'

env['file']     = 'Mechanism/CARBO_air.yaml'
env['id']       = 'air'

# Initialize Cantera object
gas['obj']      = ct.Solution(GasParams['filename'], gas['id'])
gas_surf['obj'] = ct.Solution(GasParams['filename'], gas_surf['id'])
part['obj']     = ct.Solution(part['file'], part['id'])
wall['obj']     = ct.Solution(wall['file'], wall['id'])
surf['obj']     = ct.Interface(GasParams['filename'], surf['id'], [gas_surf['obj']])
env['obj']      = ct.Solution(env['file'], env['id'])

#%% Set geometry
geo_b           = 2         # Bed geometry  1 = rectangular, 2 = cylindrical (annular or tubular)
arr_tube        = 2         # Tube arrangement 1 = lab scale (uniform heating), 2 = solar tower

#%% Custom naming
BedParams['custom_name'] = '_no_spec_therm_disp'

#%% Solve chem
SurfParams['chem'] = 1      # 1 = solve chem, 2 = do not solve chem

#%% Solve energy balance
BedParams['energy'] = 1     # 1 = solve energy, 2 = do not solve energy

#%% Isothermal bed
BedParams['isothermal'] = 2 # 1 = isothermal case, 2 = spatially dependent temperature profile

#%% Set dispersion model
BedParams['species_dispersion'] = 2 # 1 = with dispersion, 2 = without dispersion
BedParams['thermal_dispersion'] = 2 # 1 = with dispersion, 2 = without dispersion

#%% Solve interface species of not
PartParams['interface'] = 1 # 1 = solve interface species, 2 = do not solve interface species

#%% Solve Momentum equations
BedParams['gas_momentum'] = 1 # 1 = solve momentum eqn, 2 = do not solve momentum eqn
BedParams['solid_momentum'] = 1 # 1 = solve momentum eqn, 2 = do not solve momentum eqn

#%% Sherwood number correlation
PartParams['Sherwood'] = 'constant' # Gunn, Frossling, Zhang, constant

#%% Nusselt number correlation
BedParams['Nusselt'] = 'Gunn' # Gunn, other, Zhang

#%% Assume pressude drop in kPa/m along the bed height
BedParams['P_drop'] = 9.45e3 # [Pa/m]

#%% Uniform flux profile or custom (applied in Init and WallEqs, set in the below EnvParams section)
EnvParams['flux_profile'] = 1 # 1 = uniform, 2 = custom
flux_profile_shape = 3 # 1 = platikurtic, 2 = 1/x, 3 = conical over mid height

#%% Fin parameters that uses refined correlation based on the area increase
BedParams['fin'] = 1 # 1 = with fin, 2 = without fin

if BedParams['fin'] == 1:
    fin_mat = 'SS_bulk' # fin material for conductivity: choices 'SS_bulk', 'Haynes_230', 'Inco470H'
    fin_mt  =  0.50e-3  # fin wall thickness (0.05-0.5 mm)  [m]
    fin_dz  =  5.00e-3  # fin depth extension (into bed depth), from base to outside of crest [m]
    fin_dx  =  0.0254/7.6162 # width of a single fin unit (for strip fins, half the width of a U-shape + half the trough) [1/m]
    fin_dx_crest = 1.5e-3 # outer crest width (equal to r_crest_out for U-shaped fin) [m]
    fin_r_crest_out = 1.50e-3 # outer bend radius on crest (1 thickness minimum) [m]
    fin_r_trough_in = 0.75e-3 # inner bend radius on trough (1 thickness minimum) [m]
    fin_dy  =  16.0e-3  # fin segment height (along channel height) [m]
    fin_dy_offset  =  3.0e-3 # fin segment offset height between fins (along channel height) [m]

#%% Number of particle cells, if it is a multicell particle
PartParams['multi_part'] = 2 # if multi_part = 1, then n_p has to be > 1
n_p = 5

# Discretization of the particle
SurfParams['geo'] = 1 # 1 = evenly spaced, 2 = dense outer

# if multi_part is 1, then simple part can not be 1 (deactivated)
if PartParams['multi_part'] == 1:
    PartParams['simple_part'] = 2 
else:
    PartParams['simple_part'] = 1   
    
#%% Set gas flow conditions
mdot_g_flux     = [1.0] * n_files              # Fluidizing gas inlet mass flux [kg/m^2-s]
Tg_in           = [(700 + 273.15)] * n_files   # Fluidizing gas inlet temperature [K]
Pg_in           = [10.0E5] * n_files           # Fluidizing gas outlet pressure [Pa]

# Gas information
gas['R']                = ct.gas_constant
gas['kspec']            = gas['obj'].n_species
gas['species_names']    = gas['obj'].species_names

# Gas phase species names and inlet mole fractions
gas['kH2']  = gas['obj'].species_index('H2')
gas['kCH4'] = gas['obj'].species_index('CH4')
gas['kCO']  = gas['obj'].species_index('CO')
gas['kCO2'] = gas['obj'].species_index('CO2')
gas['kH2O'] = gas['obj'].species_index('H2O')
if gas['kspec'] == 6:
    gas['kO2']  = gas['obj'].species_index('O2')
if gas['kspec'] == 7:
    gas['kN2']  = gas['obj'].species_index('N2')

# Mole ratio of inlet CO2 and H2O with CH4   
rat_CO2 = 0.0
rat_H2O = 2.0

# Percent [%] Mole(volume) of inert gas N2 to calculate mole ratio of N2 with CH4
vol_N2 = 0/100
rat_N2 = (vol_N2 * (rat_CO2 + rat_H2O + 1)) / (1 - vol_N2)

X_CH4 = 1/(rat_CO2 + rat_H2O + 1 + rat_N2) 

Xk_bg_in = np.zeros((gas['kspec']))
Xk_bg_in[gas['kH2']] = 1e-6
Xk_bg_in[gas['kCH4']] = X_CH4
Xk_bg_in[gas['kH2O']] = 1 - np.sum(Xk_bg_in) #rat_H2O * X_CH4
Xk_bg_in[gas['kCO']] = 1e-6
Xk_bg_in[gas['kCO2']] = 1e-6 #rat_CO2 * X_CH4 #1e-6
if gas['kspec'] == 6:
   Xk_bg_in[gas['kO2']] = 0
if gas['kspec'] == 7:
   Xk_bg_in[gas['kN2']] = 1 - np.sum(Xk_bg_in) #rat_N2 * X_CH4 #

Xk_bg_in = Xk_bg_in /np.sum(Xk_bg_in)

#%% Set partcle flow conditions
mdot_p_flux     = [0] * n_files                     # Particle inlet mass flux [kg/m^2-s]

#%% Set particle/surface parameters
dp              = [600e-6] * n_files       # Particle diameter [m]
tau_p           = [3.0] * n_files          # [-], tortuosity of porous particle
phi_p           = [0.5] * n_files          # [-], porosity of porous particle    
r_pore_p        = [300e-9] * n_files       # [-], average pore radius 

cat_loading     = [0.01] * n_files
active_radius   = [1.00] * n_files

#%% Define species indices for the surface phase
surf['kHCO']    = surf['obj'].species_index('HCO(s)')
surf['kCO2']    = surf['obj'].species_index('CO2(s)')
surf['kO']      = surf['obj'].species_index('O(s)')
surf['kCH4']    = surf['obj'].species_index('CH4(s)')
surf['kCH']     = surf['obj'].species_index('CH(s)')
surf['kCH2']    = surf['obj'].species_index('CH2(s)')
surf['kCH3']    = surf['obj'].species_index('CH3(s)')
surf['kC']      = surf['obj'].species_index('C(s)')
surf['kCO']     = surf['obj'].species_index('CO(s)')
surf['kOH']     = surf['obj'].species_index('OH(s)')
if file != 3:
    surf['kCOOH']   = surf['obj'].species_index('COOH(s)')
surf['kH']      = surf['obj'].species_index('H(s)')
surf['kH2O']    = surf['obj'].species_index('H2O(s)')
surf['kNi']     = surf['obj'].species_index('Ni(s)')

surf['species_names']   = surf['obj'].species_names
surf['kspec']           = surf['obj'].n_species

# Set initial surface composition
uniform_cov = 0
SurfParams['Zk_p_0'] = np.zeros(surf['kspec'])
SurfParams['Zk_p_0'][surf['kHCO']] = uniform_cov
SurfParams['Zk_p_0'][surf['kCO2']] = uniform_cov
SurfParams['Zk_p_0'][surf['kO']] = uniform_cov
SurfParams['Zk_p_0'][surf['kCH4']] = uniform_cov
SurfParams['Zk_p_0'][surf['kCH']] = uniform_cov
SurfParams['Zk_p_0'][surf['kCH2']] = uniform_cov
SurfParams['Zk_p_0'][surf['kCH3']] = uniform_cov
SurfParams['Zk_p_0'][surf['kC']] = uniform_cov
SurfParams['Zk_p_0'][surf['kCO']] = 0
SurfParams['Zk_p_0'][surf['kOH']] = uniform_cov
if file != 3:
    SurfParams['Zk_p_0'][surf['kCOOH']] = uniform_cov
SurfParams['Zk_p_0'][surf['kH']] = 0
SurfParams['Zk_p_0'][surf['kH2O']] = uniform_cov
SurfParams['Zk_p_0'][surf['kNi']] = 1 - np.sum(SurfParams['Zk_p_0'])

# Convert to list to make it json serializable
SurfParams['Zk_p_0']    = SurfParams['Zk_p_0'].tolist()

#%% Set solar flux parameters on external wall depending on geometry
if geo_b == 1:
    q_sol_ap    = [350E3] * n_files             # Concentrated solar flux at aperture (before flux spreading) [W/m^2]
    angle_w     = 180/np.pi*math.asin(0.2)      #  Wall angle or cylindricl geometry of  receiver cavity to spread flux [deg.]
    q_sol_w     = [x * math.sin(math.radians(angle_w)) for x in q_sol_ap] # Concentrated solar flux on rectangular cavity wall (after flux spreading) [W/m^2]
    n_w_rad     = [2] * n_files                 # Number of irradiated walls for heat transfer [--]
    n_w         = [2] * n_files                 # Number of walls for particle drag [--]
elif geo_b == 2:
    q_sol_ap    = [150E3] * n_files             # Concentrated solar flux at aperture (before flux spreading) [W/m^2]
    if arr_tube == 1:
        sol_frac_w  = 1                         # Fraction of tube that sees the solar flux (which is assumed to uniformly spread around the tube [--]
    elif arr_tube ==2:
        sol_frac_w  = 1/np.pi
    q_sol_w     = [x * sol_frac_w for x in q_sol_ap] # Mean concentrated solar flux at on cir  (after flux spreading) [W/m^2]
    n_w_rad     = [1] * n_files                 # Number of irradiated walls for heat transfer [--]
    n_w         = [1] * n_files                 # Number of walls for particle drag [--]

#%% Set ambient parameters
P_amb = [1.01325e5] * n_files               #  ambient pressure for external heat transfer [Pa]
T_amb = [(20 + 273.15)] * n_files           #  ambient temperature for external heat transfer [K]

# Environment (air) information
env['R']        = ct.gas_constant
env['kspec']    = env['obj'].n_species 
env['species_names'] = env['obj'].species_names

#%%  Set up vertical mesh for reactor geometry
y_b         = [2.0] * n_files          # Bed channel height for heat transfer [m]
n_y_b       = [51] * n_files           # Number of mesh points in the vertical direction of the channel flow
dy_cond_0   = [1e-1] * n_files         # Vertical conduction length at the top of the wall [m]
dy_cond_L   = [1e-1] * n_files         # Vertical conduction length at the bottom of the wall [m]

#%%  Set up bed parameters
# Rectangular
if geo_b == 1:
    dx_b        = [1.0] * n_files       # Fluidized bed channel width [m]
    dz_b        = [0.1] * n_files       # Fluidized bed channel depth between heat transfer walls [m]
    dx_w_b      = [1.0] * n_files       # Fluidized bed wall channel width [m]
    dz_w_b      = [1E-2] * n_files      # Fluidized bed depth of receiver wall [m]

elif geo_b == 2:
    d_out_b     = [0.1] * n_files  # Bed outer diameter [m] # 1.5 inch = 31.75e-3 m, 5in = 0.127
    #d_out_b     = [31.75e-3, 50.8e-3]   # Bed outer diameter [m]
    wt          = [2.0e-3] * n_files  # 1.5in OD = 38.1mm OD, 1/8in w.t. = 3.175 mm w.t., 1.25in ID = 31.75mm ID
    d_in_b      = [0E-03] * n_files     # Bed  inner diameter [m]
    d_w_out_b   = [x + 2*y for x, y in zip(d_out_b, wt)] # Bed tube outer diameter [m]

#%% Generate input for each file case        
for i_file in range(n_files):
    # Re-initialize Cantera obj since it is deleted when creating files
    gas['obj']      = ct.Solution(GasParams['filename'], gas['id'])
    gas_surf['obj'] = ct.Solution(GasParams['filename'], gas_surf['id'])
    part['obj']     = ct.Solution(part['file'], part['id'])
    wall['obj']     = ct.Solution(wall['file'], wall['id'])
    surf['obj']     = ct.Interface(GasParams['filename'], surf['id'], [gas_surf['obj']])
    env['obj']      = ct.Solution(env['file'], env['id'])
    
    #%% SECTION: Bed and Wall - Bed/Wall channel geometric grid/mesh information
    BedParams['n_y_b']          = n_y_b[i_file]    
    BedParams['y_0']            = 0.0001                         # [m], Position of the first cell at the fluidizing gas injection point
    BedParams['y_b']            = np.linspace(BedParams['y_0'], y_b[i_file], BedParams['n_y_b'])
    
    BedParams['n_y']            = BedParams['n_y_b']
    BedParams['y']              = BedParams['y_b']
    
    BedParams['dy_b'] = np.concatenate(( [0.5*(BedParams['y_b'][1] - BedParams['y_b'][0])], \
            0.5*(BedParams['y_b'][2:] - BedParams['y_b'][:-2]), \
            [0.5*(BedParams['y_b'][-1] - BedParams['y_b'][-2])] ))              # Cell mesh heights in bed channel [m]
    
    BedParams['dy'] = BedParams['dy_b']  # All cell mesh heights in bed channel and freeboard zone [m]
    
    BedParams['y_bnd'] = np.zeros(BedParams['n_y'] + 1)
    BedParams['y_bnd'][0] = BedParams['y_0']
    for i_y in range(1, len(BedParams['y_bnd'])):
            BedParams['y_bnd'][i_y] = BedParams['y_bnd'][i_y-1] + BedParams['dy'][i_y-1]
    
    WallParams['n_y_b']     = BedParams['n_y_b']
    WallParams['n_y']       = BedParams['n_y']
    WallParams['y_b']       = BedParams['y_b']
    WallParams['y']         = BedParams['y']            # All heights of wall in bed channel and freeboard zone [m]
    WallParams['dy_b']      = BedParams['dy_b']         # Heights of wall in bed channel [m]
    WallParams['dy']        = BedParams['dy']           # All heights of bed channel and freeboard zone [m]
    
    BedParams['dy_cond_0']  = dy_cond_0[i_file]         # [m], Conduction loss length scale at bottom of bed  (fluidizing gas inlet)
    BedParams['dy_cond_L']  = dy_cond_L[i_file]         # [m], Conduction loss length scale at top of bed 
    BedParams['dy_cond']    = np.concatenate(( [BedParams['dy_cond_0']], BedParams['y'][1:] - BedParams['y'][:-1], [BedParams['dy_cond_L']] ))
    
    WallParams['dy_cond_0'] = BedParams['dy_cond_0']    # Conduction length scale at bottom of wall
    WallParams['dy_cond_L'] = BedParams['dy_cond_L']    # Conduction length scale at top of wall
    WallParams['dy_cond']   = np.concatenate(( [WallParams['dy_cond_0']], WallParams['y'][1:] - WallParams['y'][:-1], [WallParams['dy_cond_L']] ))
    
    BedParams['n_w'] = n_w[i_file]                      # Number of walls for particle drag
    BedParams['n_w_rad'] = n_w_rad[i_file]              # Number of irradiated walls
    WallParams['n_w'] = n_w[i_file]                     # Number of walls for particle drag
    WallParams['n_w_rad'] = n_w_rad[i_file]             # Number of irradiated walls
    
    BedParams['geo'] = geo_b
    if BedParams['geo'] == 1:
        BedParams['geo_name']   = 'planar'
        BedParams['geo_size']   = dx_b[i_file]
        BedParams['dz_b']       = dz_b[i_file]*np.ones(BedParams['n_y_b'])      # Fluidized bed channel depth between heat transfer walls [m]
        BedParams['dx_b']       = dx_b[i_file]*np.ones(BedParams['n_y_b'])      # Fluidized bed channel width [m]
        BedParams['dz']         = np.concatenate(( BedParams['dz_b'] ))  # Fluidized bed and reeboard zone depths [m]
        
        WallParams['dz_b']      = dz_w_b[i_file]*np.ones(BedParams['n_y_b'])    # Walll depth for fluidized bed [m]
        WallParams['dx_b']      = dx_w_b[i_file]*np.ones(BedParams['n_y_b'])    # Walll widt for fluidized bed [m]
        WallParams['dz']        = np.concatenate(( WallParams['dz_b'] )) # Wall depthss infFluidized bed and freeboard zone [m]
    
    elif BedParams['geo'] == 2:
        BedParams['geo_name']   = 'tubular'
        BedParams['geo_size']   = d_out_b[i_file]
        BedParams['d_in_b']     = d_in_b[i_file]*np.ones(BedParams['n_y_b'])    # Inner diameter of tubular fluidized bed [m]
        BedParams['d_out_b']    = d_out_b[i_file]*np.ones(BedParams['n_y_b'])   # Outer diameter of tubular fluidized bed [m]
        BedParams['d_in']       = BedParams['d_in_b']   # Inner diameters of tubular fluidized bed and freeboard zone [m]
        BedParams['d_out']      = BedParams['d_out_b']  # Outer diameters of tubular fluidized bed and freeboard zone [m]
        
        WallParams['d_in_b']    = d_out_b[i_file]*np.ones(BedParams['n_y_b'])   # Outer wall dimaeter of tubular bed [m]
        WallParams['d_out_b']   = d_w_out_b[i_file]*np.ones(BedParams['n_y_b']) # Outer wall dimaeter of tubular bed [m]
        WallParams['d_in']      = WallParams['d_in_b']   # Outer diameters of walls for tubular fluidized bed and freeboard zone [m]
        WallParams['d_out']     = WallParams['d_out_b']  # Outer diameters of walls for tubular fluidized bed and freeboard zone [m]
    
    # Indexes for fluidized bed and bed zone
    BedParams['index_b']        = np.array( np.where(BedParams['y'] <= BedParams['y_b'][-1]) ).reshape(BedParams['n_y_b']) # finds indexes for the bed zone

    if BedParams['geo'] == 1:
        BedParams['Ay_b']       = BedParams['dx_b']*BedParams['dz_b']           # Horizontal cross-sectional areas in fluidized bed [m^2]
        BedParams['Ay']         = np.concatenate(( BedParams['Ay_b'] )) # All horizontal cross-sectional areas in bed [m^2]
        BedParams['Ay_cond']    = np.concatenate(( [BedParams['Ay_b'][0]], BedParams['Ay'] )) # All horizontal cross-sectional areas for conduction in bed [m^2]
        
        BedParams['Az_in_b']    = 0.0*BedParams['dy_b']*BedParams['dx_b']       # Vertical areas of internal wall in fluidized bed [m^2]
        BedParams['Az_out_b']   = BedParams['n_w']*BedParams['dy_b']*BedParams['dx_b'] # Vertical areas of external wall in fluidized bed [m^2]
        BedParams['Az_in']      = np.concatenate(( BedParams['Az_in_b'] ))   # Vertical areas of internal wall in fluidized bed and freeboard zone [m^2]
        BedParams['Az_out']     = np.concatenate(( BedParams['Az_out_b'] )) # Vertical areas of external wall in fluidized bed and freeboard zone [m^2]
        BedParams['Az_out_tot_b'] = BedParams['n_w']*(BedParams['y_b'][-1] - BedParams['y_b'][0])*BedParams['dx_b'] # Total internal wall area parallel to flow direction [m^2]
        
        BedParams['dVol_b']     = BedParams['Ay_b']*BedParams['dy_b']               # Volumes of the bed for the  fluidized bed [m^3]
        BedParams['dVol']       = np.concatenate(( BedParams['dVol_b'] )) # Volumes of the bed for the fluidized bed and the freeboard zone [m^3]
        BedParams['D_hyd_b']    = 2*BedParams['Ay_b']/(BedParams['dz_b'] + BedParams['dx_b'])   # Hydraulic diameters of fluidized bed [m]
        BedParams['D_hyd']      = np.concatenate(( BedParams['D_hyd_b'] )) # Hydraulic diameters for both fluidized bed chnnel and freeboard zone [m]
        
        WallParams['Ay_b']      = WallParams['n_w']*WallParams['dz_b']*WallParams['dx_b']   # Horizontal cross-sectional areas of wall for fluidiized bed [m^2]
        WallParams['Ay']        = np.concatenate(( WallParams['Ay_b'] )) # All horizontal cross-sectional areas of wall [m^2]
        WallParams['Ay_cond']   = np.concatenate(( [WallParams['Ay'][0]], WallParams['Ay'] )) # All horizontal cross-sectional areas for conduction in wall [m^2]
        
        WallParams['Az_in_b']   = WallParams['n_w']*WallParams['dy_b']*WallParams['dx_b']   # Vertical areas of internal wall in contact with the bed for the fludizeed bed [m^2]
        WallParams['Az_out_b']  = WallParams['n_w']*WallParams['dy_b']*WallParams['dx_b']   # Vertical areas of external wall in contact with the ambient for the fludizeed bed  [m^2]
        WallParams['Az']        = np.concatenate(( WallParams['Az_in_b'] )) # All vertical cross-sectional areas of wall [m^2]
        
        WallParams['dVol_b']    = WallParams['Ay_b']*WallParams['dy_b']         # Volumes of the wall for the fluidized bed [m^3]
        WallParams['dVol']      = np.concatenate(( WallParams['dVol_b'] )) # Volumes of the wall for the fluidized bed and the freeboard zone [m^3]
    
    elif BedParams['geo'] == 2:
        BedParams['Ay_b']       = 0.25*np.pi*(BedParams['d_out_b']**2 - BedParams['d_in_b']**2) # Horizontal cross-sectional area perpendicular to flow directionn in fluidized bed [m^2]
        BedParams['Ay']         = BedParams['Ay_b']  # All horizontal cross-sectional areas in bed [m^2]
        BedParams['Ay_cond']    = 0.5*( np.concatenate(( [BedParams['Ay'][0]], BedParams['Ay'] )) + \
                                       np.concatenate(( BedParams['Ay'], [BedParams['Ay'][-1]] )) )  # All horizontal cross-sectional areas for conduction in bed [m^2]
        
        BedParams['Ar_in_b']    = np.pi*BedParams['d_in_b']*BedParams['dy_b']   # Vertical areas of internal wall in fluidized bed [m^2]
        BedParams['Ar_out_b']   = np.pi*BedParams['d_out_b']*BedParams['dy_b']  # Vertical areas of exernal wall in fluidized bed [m^2]
        BedParams['Ar_in']      = BedParams['Ar_in_b']  # Vertical areas of internal wall in fluidized bed and freeboard zone [m^2]
        BedParams['Ar_out']     = BedParams['Ar_out_b'] # Vertical areas of external wall in fluidized bed and freeboard zone [m^2]
        BedParams['Ar_out_tot_b'] = np.pi*BedParams['d_out_b']*(BedParams['y_b'][-1] - BedParams['y_b'][0]) # Total interal wall area parallel to flow direction [m^2]
        
        BedParams['dVol_b']     = BedParams['Ay_b']*BedParams['dy_b']           # Volumes of the bed for the fluidized bed [m^3]
        BedParams['dVol']       = BedParams['dVol_b']  # Volumes of the bed for the fluidized bed and the freeboard zone [m^3]
        
        BedParams['D_hyd_b']    = 4*BedParams['Ay_b']/(np.pi*BedParams['d_out_b']) # Hydraulic diameter of fluidized bed [m]
        BedParams['D_hyd']      = BedParams['D_hyd_b']  # Hydraulic diameters for both fluidized bed chnnel and freeboard zone [m]
        
        WallParams['Ay_b']      = WallParams['n_w']*0.25*np.pi*(WallParams['d_out_b']**2 - WallParams['d_in_b']**2) # Horizontal cross-sectional areas of wall for fluidized bed [m^2]
        WallParams['Ay']        = WallParams['Ay_b'] # All horizontal cross-sectional areas of wall [m^2]
        WallParams['Ay_cond']   = 0.5*( np.concatenate(( [WallParams['Ay'][0]], WallParams['Ay'] )) + \
                                       np.concatenate(( WallParams['Ay'], [WallParams['Ay'][-1]] )) ) # All horizontal cross-sectional areas for conduction in wall [m^2]
        
        WallParams['Ar_in_b']   = WallParams['n_w']*np.pi*WallParams['d_in_b']*WallParams['dy_b'] # Vertical areas of internal wall in contact with the bed for the fludizeed bed [m^2]
        WallParams['Ar_out_b']  = WallParams['n_w']*np.pi*WallParams['d_out_b']*WallParams['dy_b'] # Vertical areas of internal wall in contact with the ameinbe for the fludizeed bed [m^2]
        
        WallParams['dVol_b']    = WallParams['Ay_b']*WallParams['dy_b']         # Volume of the wall for the fluidized bed  [m^3]
        WallParams['dVol']      = WallParams['dVol_b']  # Volumes of the wall for the fluidized bed and the freeboard zone [m^3]
        
    BedParams['Vol_tot'] = np.sum(BedParams['dVol'])
    
    #%% Bed and Wall Temperature Parameters
    # Bed temperature parameters at top and bottom boundaries
    BedParams['T_y_0']          = Tg_in[i_file]      # [K], Wall temperature at bottom of bed (fluidizing gas inlet)
    BedParams['T_y_L']          = Tg_in[i_file]      # [K], Wall temperature at top of bed (particle inlet)
    
    # Wall temperature parameters at top and bottom boundaries
    WallParams['T_y_0']         = BedParams['T_y_0']        # Wall temperature at bottom of bed
    WallParams['T_y_L']         = BedParams['T_y_L']        # Wall temperature at top of bed

    #%% SECTION: Wall - Receiver-Reactor wall geometry and properties
    WallParams['id']    = str(wall['id'])                           # Identification/name of gas
    
    # Set generic state of wall and obtain its density
    wall['obj'].TP      = WallParams['T_y_0'] , 1.01325e5     # Set wall temperature and pressure
    WallParams['rho']   = wall['obj'].density           # Wall density in kg/m^3
    
    if WallParams['id'] == 'SS_bulk':
        WallParams['emis'] = 0.78   # Wall emissivity [-] (source?) (should be for oxidized SS at thermal temps (~900 C) [--]
        WallParams['abs'] = 0.95    # Wall absorptivity at solar temperatures (~5800K) for oxidized SS (CHECK value), assuming better absorptivity due to Pyromark
    elif WallParams['id'] == 'Inco470H':
        WallParams['emis'] = 0.78
        WallParams['abs'] = 0.95
    elif WallParams['id'] == 'silicon-carbide':
        WallParams['emis'] = 0.8
        WallParams['abs'] = 0.95
    
    #%% Section: Fin parameters
    if BedParams['fin'] == 1:
        FinParams['mat']        = fin_mat # fin material for conductivity: choices 'SS_bulk', 'Haynes_230', 'Inco470H'
        FinParams['mt']         = fin_mt # fin wall thickness (0.05-0.5 mm)  [m]
        FinParams['dz']         = fin_dz # fin depth extension (into bed depth), from base to outside of crest [m]
        FinParams['dx']         = fin_dx # width of a single fin unit (for strip fins, half the width of a U-shape + half the trough) [1/m]
        FinParams['dx_crest']   = fin_dx_crest # outer crest width (equal to r_crest_out for U-shaped fin) [m]
        FinParams['r_crest_out'] = fin_r_crest_out # outer bend radius on crest (1 thickness minimum) [m]
        FinParams['r_trough_in'] = fin_r_trough_in # inner bend radius on trough (1 thickness minimum) [m]
        FinParams['dy']          = fin_dy # fin segment height (along channel height) [m]
        FinParams['dy_offset']   = fin_dy_offset
        
        # This function calculates fin geometric properties needed 
        # Calculate important lengths and radii that are needed for getting conduction and areas for fin
        FinParams['dx_trough']      = FinParams['dx'] - FinParams['dx_crest'] # half trough width (perpendicular to channel length) [m]

        FinParams['r_trough_out']   = FinParams['r_trough_in'] + FinParams['mt'] # outer bend radius on trough where the U-shaped fin connects to the base [m]
        FinParams['r_trough_mean']  = 0.5 *(FinParams['r_trough_in'] + FinParams['r_trough_out']) # mean bend radius on trough where the U-shaped fin connects to the base [m]

        FinParams['dx_crest_flat']  = FinParams['dx_crest'] - FinParams['r_crest_out'] # half flat portion at top of crest (= 0 for U-shaped fin) [m]
        FinParams['dx_trough_flat'] = FinParams['dx_trough'] - FinParams['r_trough_in'] # flat portion of trough per fin [m]
        FinParams['dz_crest_flat']  = FinParams['dz'] - FinParams['r_trough_out'] - FinParams['r_crest_out'] # flat portion of crest depth /sidewall [m]
        
        # Determine the inner, outer, and mean (cnduction) lengths of the fin unit away from the wall
        FinParams['l_in']           = np.pi/2 * FinParams['r_trough_in'] + FinParams['dz_crest_flat'] + np.pi/2 * FinParams['r_crest_out'] + FinParams['dx_crest_flat'] # fin length on outer surface [m]
        FinParams['l_out']          = np.pi/2 * FinParams['r_trough_out'] + FinParams['dz_crest_flat']  + np.pi/2 *(FinParams['r_crest_out'] - FinParams['mt']) + FinParams['dx_crest_flat'] # fin length on inner surface [m]
        FinParams['l_cond']         = 0.5 * (FinParams['l_in'] + FinParams['l_out']) # fin length used for conduction [m]
        
        # Calculate areas for fin calculations
        FinParams['Perim']      = 2*(FinParams['dy'] + FinParams['mt']) # perimeter of single fin at a given cross section [m]
        FinParams['Az_cond']    = FinParams['dy'] * FinParams['mt'] # area of conduction in z-directon of single fin [m^2]
        FinParams['A_base']     = FinParams['dx'] * (FinParams['dy'] + FinParams['dy_offset']) # base wall area for a single fin unit including base [m^2]
        FinParams['A_surf']     = ( FinParams['l_in'] + FinParams['l_out'] ) * FinParams['dy'] # surface area for a single fin unit [m^2]
        FinParams['A_tot']      =  FinParams['A_surf'] + FinParams['A_base'] - FinParams['Az_cond'] # total area for a single fin unit [m^2]
        FinParams['A_fin_o_A_tot']  = FinParams['A_surf'] / FinParams['A_tot'] # ratio of fin area over total area [--]
        FinParams['A_tot_o_A_base'] = FinParams['A_tot'] / FinParams['A_base'] # ratio of finned wall total area over base wall area [--]
    
    #%% SECTION: Particle
    PartParams['id'] = str(part['id'])      # Identification/name of particle

    # Define physical and radiative properties of particles based on particle ID
    if PartParams['id'] == 'Alumina':
        PartParams['phi_bs_max'] = 0.59  # [-], static bed solid volume fraction
        PartParams['emis'] = 0.8         # [-], particle emissivity
    elif PartParams['id'] == 'CARBO_HSP':
        PartParams['phi_bs_max'] = 0.585 # [-], static bed solid volume fraction
        PartParams['emis'] = 0.78        # [-], particle emissivity from Siegel et al (2015)
    elif PartParams['id'] == 'Silica':
        PartParams['phi_bs_max'] = 0.58  # [-], static bed solid volume fraction
        PartParams['emis'] = 0.715       # [-] from ho paper
    elif PartParams['id'] == 'SiO2':
        PartParams['phi_bs_max'] = 0.59  # [-], static bed solid volume fraction (Brayton)
        PartParams['emis'] = 0.8         # [-], particle emissivity (Brayton)
    elif PartParams['id'] == 'Regolith':
        PartParams['phi_bs_max'] = 0.5   # [-] static bed solid volume fraction
        PartParams['emis'] = 0.8         # [-], particle emissivity (estimated)
    
    PartParams['phi_bg_min']    = 1 - PartParams['phi_bs_max']
    
    PartParams['dp']            = dp[i_file]     # [m], Mean sauter diameter of particle
    PartParams['phi']           = phi_p[i_file]                          # [-] porosity of the particle 
    
    PartParams['e']             = 0.96           # [-], coefficient of restitution, Lv 2018
    PartParams['g_o']           = (1+(1-PartParams['phi_bs_max'])**(1/3))**(-1)   # [-] radial distribution function Ding and Gidaspow 1990
    
    #  Compaction stress calculation terms
    PartParams['Go']            = 1     # [Pa]
    PartParams['c']             = 500   # [-], compaction modulus
    
    # Drag model and correction factor
    PartParams['drag_model']        = 'Gidaspow'
    #PartParams['drag_model']        = 'Syamlal-OBrien'
    PartParams['drag_corrector']    = 'phi_star'
    #PartParams['drag_corrector']    = 'none'
    
    PartParams['T_in']          = Tg_in[i_file]
    
    part['obj'].TP              = PartParams['T_in'], 1.01325e5                 # Define inlet state of particles
    PartParams['rho_in']        = (1 - PartParams['phi']) * part['obj'].density # [kg/m^3], inlet density of particles
    PartParams['cp_in']         = part['obj'].cp_mass                           # [J/kg-K], inlet heat capacity of particles
    PartParams['h_in']          = part['obj'].enthalpy_mass                     # [J/kg], inlet enthalpy of particles

    PartParams['mdot_in']       = -mdot_p_flux[i_file]*BedParams['Ay_b'][-1]
    PartParams['mflux_in']      = PartParams['mdot_in']/BedParams['Ay'][-1]  # [kg/s/m^2], inlet mass flux of particles
    PartParams['Vdot_in']       = PartParams['mdot_in']/PartParams['rho_in']        # [m^3/s], inlet volumetric flow rate of particles
    PartParams['u_in']          = PartParams['Vdot_in']/BedParams['Ay'][-1]  # [m/s], inlet superficial velocity of particles
    PartParams['v_in']          = PartParams['u_in']/PartParams['phi_bs_max']   # [m/s], inlet interstital velocity of particles
    PartParams['mvdot_in']      = PartParams['mdot_in']*abs(PartParams['v_in']) # [kg-m/s], inlet momentum rate of particles
    PartParams['mhdot_in']      = PartParams['mdot_in']*PartParams['h_in']          # [J/s], inlet enthalpy rate of particles
    
    #%% SECTION: Gas
    gas['Wk']                   = gas['obj'].molecular_weights
    
    GasParams['id']             = str(gas['id'])
    GasParams['mdot_in']        = mdot_g_flux[i_file] * BedParams['Ay'][0] # Inlet mass flow rate [kg/s]
    GasParams['T_in']           = Tg_in[i_file]            # Inlet temperature [K]
    BedParams['T_b']            = GasParams['T_in']
    GasParams['P_in']           = Pg_in[i_file]            # Inlet pressure [Pa]

    GasParams['X_in'] = Xk_bg_in

    # Gas inlet mass fractions
    GasParams['Y_in'] = (np.array(GasParams['X_in']) * gas['Wk']) / sum(np.array(GasParams['X_in']) * gas['Wk'])
    
    # Set inlet mass fractions for each species
    gas['Yk_in'] = {species: GasParams['Y_in'][i] for i, species in enumerate(gas['species_names'])}

    # Check mass fractions
    if not np.isclose(sum(GasParams['Y_in']), 1) or len(gas['Yk_in']) != gas['kspec']:
        raise ValueError("Check mass fractions")

    # Set the gas state
    gas['obj'].TPY = GasParams['T_in'] , GasParams['P_in'] , GasParams['Y_in']

    # Retrieve gas inlet properties
    GasParams['rho_in']     = gas['obj'].density                    # Inlet density [kg/m^3]
    GasParams['cp_in']      = gas['obj'].cp_mass                    # Inlet heat capacity [J/kg-K]
    GasParams['h_in']       = gas['obj'].enthalpy_mass              # Inlet enthalpy [J/kg]
    GasParams['hk_in']      = gas['obj'].standard_enthalpies_RT*gas['R']*GasParams['T_in']/gas['Wk']  

    # Additional calculations
    GasParams['mflux_in']   = GasParams['mdot_in'] / float(BedParams['Ay'][0])  # Inlet mass flux [kg/s/m^2]
    GasParams['Vdot_in']    = GasParams['mdot_in'] / GasParams['rho_in']        # Inlet volumetric flow rate [m^3/s]
    GasParams['u_in']       = GasParams['Vdot_in'] / float(BedParams['Ay'][0])  # Inlet superficial velocity [m/s]
    GasParams['v_in']       = GasParams['u_in'] / (1 - PartParams['phi_bs_max']) # Inlet interstitial velocity [m/s]
    GasParams['mvdot_in']   = GasParams['mdot_in'] * GasParams['v_in']          # Inlet momentum rate [kg-m/s]
    GasParams['mhdot_in']   = GasParams['mdot_in'] * GasParams['h_in']          # Inlet enthalpy rate [J/s]
    GasParams['k']          = gas['obj'].thermal_conductivity                   # Inlet thermal conductivity [W/m-K]

    # Equilibrate the gas state at temperature and pressure
    gas['obj'].equilibrate('TP')
    GasParams['Yk_out'] = gas['obj'].Y                                          # Outlet mass fractions
    GasParams['Xk_out'] = gas['obj'].X                                          # Outlet mole fractions

    #%% SECTION: Surface parameters to solve the surface species phase
    # Properties for calculation: specific density of alumina and Ni
    rho_Al = 3.65e3
    rho_Ni = 8.91e3 # Ni metal
    #rho_Ni = 6.67e3 # NiO
    
    SurfParams['active_radius'] = active_radius[i_file]
    
    SurfParams['Rmax']          = 0.5*PartParams['dp']                               # [m] outer radius of particle 
    SurfParams['Rmin']          = (1-SurfParams['active_radius'])*SurfParams['Rmax'] # [m] solid-core (non-porous) radius of particle 
    SurfParams['Reff']          = SurfParams['Rmin']
    SurfParams['phi']           = phi_p[i_file]                          # [-] porosity of the particle 
    SurfParams['tau']           = tau_p[i_file]                          # [-] tortuosity of the particle 
    SurfParams['Rpore']         = r_pore_p[i_file]                       # [m] pore radius 
    SurfParams['R_Ni']          = 20e-9                                  # [m] catalyst nanoparticle radius
    SurfParams['cat_loading']   = cat_loading[i_file]
    SurfParams['Vpart']         = (4/3)*np.pi*(SurfParams['Rmax']**3 - SurfParams['Rmin']**3)
    SurfParams['Apart']         = 4*np.pi*SurfParams['Rmax']**2
    
    SurfParams['Ni_mass']       = (SurfParams['cat_loading']/(1-SurfParams['cat_loading']))*rho_Al*\
        (1-SurfParams['phi'])*SurfParams['Vpart']
    SurfParams['Ni_vol']        = SurfParams['Ni_mass']/rho_Ni
    SurfParams['Ni_vol_ratio']  = SurfParams['Ni_vol']/SurfParams['Vpart']
    
    SurfParams['a_Ni_cat']      = (3)*(SurfParams['Ni_vol_ratio'])/SurfParams['R_Ni']  # [m^-1] specific surface area of catalyst
    SurfParams['a_surf']        = 6*SurfParams['phi']/SurfParams['Rpore'] + 6*(1-SurfParams['phi'])/PartParams['dp']

    SurfParams['a_cat']         = SurfParams['a_Ni_cat'] + SurfParams['a_surf']

    # Calculate required properties of porous particle
    SurfParams['vol_Ratio']     = SurfParams['phi'] / (1 - SurfParams['phi']) 
    SurfParams['eff_factor']    = SurfParams['phi'] / SurfParams['tau']
    SurfParams['B_g']           = SurfParams['vol_Ratio']**2 * PartParams['dp']**2 \
                                * SurfParams['eff_factor'] / 72 
    
    # Calculate catalyst mass per area, mcat_per_area [g_cat / m^2]
    SurfParams['mcat_per_area'] = 1e3 * SurfParams['Ni_mass'] / (SurfParams['a_cat']*SurfParams['Vpart'])
    
    # Calculate geometry of particle
    SurfParams['deltar']        = (SurfParams['Rmax'] - SurfParams['Rmin'])
    SurfParams['rcell']         = (SurfParams['Rmax'] + SurfParams['Rmin'])/2
    
    # Number of particle cell
    SurfParams['n_p']           = n_p
    
    # Multi cell particle, n_p will be > 2
    if n_p > 1:
        # Evenly spaced points
        if SurfParams['geo'] == 1:
            dr_int = (SurfParams['Rmax'] - SurfParams['Rmin']) / (SurfParams['n_p'] - 1)
            SurfParams['deltar'] = np.concatenate(([0.5 * dr_int], np.ones(SurfParams['n_p'] - 2) * dr_int, [0.5 * dr_int]))

            SurfParams['rface'] = np.zeros(SurfParams['n_p'] + 1)
            SurfParams['rface'][0] = SurfParams['Rmin']
            for iface in range(1, SurfParams['n_p'] + 1):
                SurfParams['rface'][iface] = SurfParams['rface'][iface - 1] + SurfParams['deltar'][iface - 1]

            SurfParams['rcell'] = np.zeros(SurfParams['n_p'])
            SurfParams['rcell'][0] = SurfParams['Rmin'] + SurfParams['deltar'][0] / 2
            for icell in range(1, SurfParams['n_p']):
                SurfParams['rcell'][icell] = SurfParams['rcell'][icell - 1] + SurfParams['deltar'][icell - 1] / 2 + SurfParams['deltar'][icell] / 2

            # Calculate cell volumes
            SurfParams['Vcell'] = np.zeros(SurfParams['n_p'])
            for icell in range(SurfParams['n_p']):
                SurfParams['Vcell'][icell] = (4 / 3) * np.pi * (SurfParams['rface'][icell + 1] ** 3 - SurfParams['rface'][icell] ** 3)

        # Denser points in X % of particle radius (SurfParams['geo'] = 2)
        elif SurfParams['geo'] == 2:
            innerFrac       = np.ceil(SurfParams['n_p']/3)
            outerFrac       = SurfParams['n_p'] - innerFrac
            outerPercent    = 0.25
            innerPercent    = 1 - outerPercent

            outerDelta  = (SurfParams['Rmax']-(innerPercent)*SurfParams['Rmax'])
            innerDelta  = ((innerPercent)*SurfParams['Rmax']-SurfParams['Rmin'])
            deltar1     = outerDelta/outerFrac
            deltar2     = (innerDelta)/innerFrac
            SurfParams['deltar']    = np.zeros(SurfParams['n_p'])
            SurfParams['rface']     = np.zeros(SurfParams['n_p'] + 1)
            SurfParams['rface'][0]  = SurfParams['Rmin']

            # Calculate deltar and rface
            for iGeo in range(SurfParams['n_p']):
                if iGeo < innerFrac :
                    SurfParams['deltar'][iGeo] = deltar2
                    SurfParams['rface'][iGeo + 1] = SurfParams['rface'][iGeo] + deltar2
                else:
                    SurfParams['deltar'][iGeo] = deltar1
                    SurfParams['rface'][iGeo + 1] = SurfParams['rface'][iGeo] + deltar1

            # Calculate rcell and Vcell
            SurfParams['rcell'] = np.zeros(SurfParams['n_p'])
            SurfParams['rcell'][0] = SurfParams['Rmin'] + deltar2 / 2
            SurfParams['Vcell'] = np.zeros(SurfParams['n_p'])

            for icell in range(SurfParams['n_p']):
                SurfParams['Vcell'][icell] = (4 / 3) * np.pi * (SurfParams['rface'][icell + 1]**3 - SurfParams['rface'][icell]**3)
                if icell > 0 and icell < innerFrac :
                    SurfParams['rcell'][icell] = SurfParams['rcell'][icell - 1] + SurfParams['deltar'][icell]
                elif icell == innerFrac:
                    SurfParams['rcell'][icell] = SurfParams['rcell'][icell - 1] + (deltar2 / 2) + (deltar1 / 2)
                elif icell > innerFrac:
                    SurfParams['rcell'][icell] = SurfParams['rcell'][icell - 1] + SurfParams['deltar'][icell]
    
        # Calculate Aface
        SurfParams['Aface'] = np.zeros(SurfParams['n_p'] + 1)
        for iface in range(SurfParams['n_p'] + 1):
            SurfParams['Aface'][iface] = 4 * np.pi * SurfParams['rface'][iface]**2
        
        # Non-dimensional particle radius
        SurfParams['rRmax'] = np.append(SurfParams['rcell'] / SurfParams['Rmax'], 1)
    
    #%% SECTION: Environment
    env['Wk']       = env['obj'].molecular_weights
    
    EnvParams['id']             = str(env['id'])
    EnvParams['T']              = T_amb[i_file]         # [K], Ambient temperature
    EnvParams['P']              = P_amb[i_file]         # [Pa], Ambient pressure
    EnvParams['q_sol']          = q_sol_w[i_file]       # [W/m^2], Concentrated solar flux at wall (before flux spreading)
    EnvParams['q_sol_aper']     = q_sol_ap[i_file]      # [W/m^2], Concentrated solar flux at aperture (before flux spreading)
    EnvParams['f']              = 0.08                  # Stand in value for view factor of surface to ambient which depends on actual rec geometry which has not been finalized
    
    # Flux Profile Setting
    if EnvParams['flux_profile'] == 1:
        EnvParams['q_sol_dist'] = np.ones((BedParams['n_y']))
    else:
        
        if flux_profile_shape == 1:
            # Flux distribution profile
            power = 8.0
            ymin = BedParams['y'][0]
            ymax = BedParams['y'][-1]

            # Y values
            y = np.linspace(ymin, ymax, BedParams['n_y'])
            #y_mid = (ymax + ymin) / 2
            y_mid = BedParams['y'][-1] / 4
            y_range = (ymax - ymin) / 2

            # Symmetric platykurtic function using cosine
            x = np.cos((y - y_mid) / y_range * np.pi / 2) ** power
            x[:int(BedParams['n_y']/4)] = 1
        
            x[int(3*BedParams['n_y']/4):int(BedParams['n_y'])] = 0
        
            # Normalize to 0–1
            x = x / max(x)
            x = x * 1.0
        
        elif flux_profile_shape == 2:
            ymin = BedParams['y'][0]
            ymax = BedParams['y'][-1]
            y = np.linspace(ymin, ymax, BedParams['n_y'])
        
            # Linear profile
            x = 1/(y+0.5)
        
        elif flux_profile_shape == 3:
            # Conical profile
            y = BedParams['y']
            #x = 1 - 0.4*(BedParams['y'] - BedParams['y'][-1]/4)**2
            x = 1 - 0.75*(BedParams['y'] - BedParams['y'][-1]/2)**2
            
            x[:int(BedParams['n_y']/2)] = 1
        
        plt.plot(x,y)
        plt.show()
        
        EnvParams['q_sol_dist'] = x
    
    # Calculate average wall flux
    q_abs = EnvParams['q_sol_dist'] * BedParams['n_w'] * EnvParams['q_sol']
    EnvParams['q_abs_avg'] = np.average(q_abs)
    
    #%% Convert all numpy array to list to make it json serializable
    if BedParams['geo'] == 1:
        BedParams['dz_b']   = BedParams['dz_b'].tolist()    
        BedParams['dx_b']   = BedParams['dx_b'].tolist()  
        BedParams['dz']     = BedParams['dz'].tolist()
        
        WallParams['dz_b']  = WallParams['dz_b'].tolist()
        WallParams['dx_b']  = WallParams['dx_b'].tolist() 
        WallParams['dz']    = WallParams['dz'].tolist()
    
        BedParams['Az_in_b']    = BedParams['Az_in_b'].tolist()
        BedParams['Az_out_b']   = BedParams['Az_out_b'].tolist()
        BedParams['Az_in']      = BedParams['Az_in'].tolist()
        BedParams['Az_out']     = BedParams['Az_out'].tolist()
        BedParams['Az_out_tot_b'] = BedParams['Az_out_tot_b'].tolist()
        
        WallParams['Az_in_b']   = WallParams['Az_in_b'].tolist()
        WallParams['Az_out_b']  = WallParams['Az_out_b'].tolist()
        WallParams['Az']        = WallParams['Az'].tolist()
        
    elif BedParams['geo'] == 2:
        BedParams['d_in_b']     = BedParams['d_in_b'].tolist()
        BedParams['d_out_b']    = BedParams['d_out_b'].tolist()
        BedParams['d_in']       = BedParams['d_in'].tolist()
        BedParams['d_out']      = BedParams['d_out'].tolist()
        
        WallParams['d_in_b']    = WallParams['d_in_b'].tolist()
        WallParams['d_out_b']   = WallParams['d_out_b'].tolist()
        WallParams['d_in']      = WallParams['d_in'].tolist()
        WallParams['d_out']     = WallParams['d_out'].tolist()
    
        BedParams['Ar_in_b']    = BedParams['Ar_in_b'].tolist()
        BedParams['Ar_out_b']   = BedParams['Ar_out_b'].tolist()
        BedParams['Ar_in']      = BedParams['Ar_in'].tolist()
        BedParams['Ar_out']     = BedParams['Ar_out'].tolist()
        BedParams['Ar_out_tot_b'] = BedParams['Ar_out_tot_b'].tolist()
        
        WallParams['Ar_in_b']   = WallParams['Ar_in_b'].tolist()
        WallParams['Ar_out_b']  = WallParams['Ar_out_b'].tolist()
    
    BedParams['Ay']         = BedParams['Ay'].tolist()
    BedParams['Ay_b']       = BedParams['Ay_b'].tolist()
    BedParams['Ay_cond']    = BedParams['Ay_cond'].tolist()
    
    WallParams['Ay']        = WallParams['Ay'].tolist()
    WallParams['Ay_b']      = WallParams['Ay_b'].tolist()
    WallParams['Ay_cond']   = WallParams['Ay_cond'].tolist()
    
    BedParams['D_hyd']      = BedParams['D_hyd'].tolist()
    BedParams['D_hyd_b']    = BedParams['D_hyd_b'].tolist()
    
    BedParams['dVol']       = BedParams['dVol'].tolist()
    BedParams['dVol_b']     = BedParams['dVol_b'].tolist()
    
    WallParams['dVol']      = WallParams['dVol'].tolist()
    WallParams['dVol_b']    = WallParams['dVol_b'].tolist()
    
    BedParams['dy']         = BedParams['dy'].tolist()
    BedParams['dy_b']       = BedParams['dy_b'].tolist()
    BedParams['dy_cond']    = BedParams['dy_cond'].tolist()
    BedParams['index_b']    = BedParams['index_b'].tolist()
    BedParams['y']          = BedParams['y'].tolist()
    BedParams['y_b']        = BedParams['y_b'].tolist()
    BedParams['y_bnd']      = BedParams['y_bnd'].tolist()
    
    WallParams['dy']        = WallParams['dy'].tolist()
    WallParams['dy_b']      = WallParams['dy_b'].tolist()
    WallParams['dy_cond']   = WallParams['dy_cond'].tolist()
    WallParams['y']         = WallParams['y'].tolist()
    WallParams['y_b']       = WallParams['y_b'].tolist()
    
    gas['Wk']               = gas['Wk'].tolist()
    GasParams['X_in']       = GasParams['X_in'].tolist()
    GasParams['Xk_out']     = GasParams['Xk_out'].tolist()
    GasParams['Y_in']       = GasParams['Y_in'].tolist()
    GasParams['Yk_out']     = GasParams['Yk_out'].tolist()
    GasParams['hk_in']      = GasParams['hk_in'].tolist()
    
    if n_p > 1:
        SurfParams['Aface'] = SurfParams['Aface'].tolist()
        SurfParams['deltar'] = SurfParams['deltar'].tolist()
        SurfParams['rcell'] = SurfParams['rcell'].tolist()
        SurfParams['rface'] = SurfParams['rface'].tolist()
        SurfParams['rRmax'] = SurfParams['rRmax'].tolist()
        SurfParams['Vcell'] = SurfParams['Vcell'].tolist()
    
    env['Wk']               = env['Wk'].tolist()
    
    EnvParams['q_sol_dist'] = EnvParams['q_sol_dist'].tolist()
    
    gas.pop('obj', None)
    gas_surf.pop('obj', None)
    part.pop('obj', None)
    wall.pop('obj', None)
    surf.pop('obj', None)
    env.pop('obj', None)
    
    #%%
    save = {
        "gas"           : gas,
        "gas_surf"      : gas_surf,
        "part"          : part,
        "wall"          : wall,
        "surf"          : surf,
        "env"           : env,
        "GasParams"     : GasParams,
        "PartParams"    : PartParams,
        "BedParams"     : BedParams,
        "WallParams"    : WallParams,
        "FinParams"     : FinParams,
        "SurfParams"    : SurfParams,
        "EnvParams"     : EnvParams
        }
    
    # Improve file naming to include case condition
    """
    file_name_to_save = PartParams['id'] + '_' + BedParams['geo_name'] + '-' + \
        f"{BedParams['geo_size']:.3f}" + '_' + 'y' + '-' + f"{BedParams['y'][-1]:.1f}" + '_' + 'dp' + '-' + \
        f"{1e6*PartParams['dp']:.0f}" + '_'  + 'Tin' + '-' + f"{GasParams['T_in']-273.15:.0f}" + '_' + \
        'Pin' + '-' + f"{GasParams['P_in']/1e5:.0f}" + '_' + 'mfluxin' + '-' + f"{GasParams['mflux_in']:.2f}" + '_' + \
        'qsol_ap' + '-' + f"{EnvParams['q_sol_aper']/1e3:.0f}" + '_' + 'qabs_avg' + '-' + f"{EnvParams['q_abs_avg']/1e3:.0f}"
    """
    file_name_to_save = 'OD' + '-' + f"{BedParams['geo_size']:.2f}" + '_' + 'y' + '-' + f"{BedParams['y'][-1]:.1f}" + '_'  \
        +'dp' + '-' + f"{1e6*PartParams['dp']:.0f}" + '_' + 'Tin' + '-' + f"{GasParams['T_in']-273.15:.0f}" + '_' \
        + 'Pin' + '-' + f"{GasParams['P_in']/1e5:.0f}" + '_' + 'mfluxin' + '-' + f"{GasParams['mflux_in']:.2f}" + '_' \
        + 'qsol_ap' + '-' + f"{EnvParams['q_sol_aper']/1e3:.0f}" + '_' + 'qabs_avg' + '-' + f"{EnvParams['q_abs_avg']/1e3:.0f}" \
        + BedParams['custom_name']
    
    file_name_to_run = file_name_to_save    
    
    with open(os.path.join(infiledate_path, file_name_to_save + ".json"), 'w') as file_out:
        json.dump(save, file_out)
