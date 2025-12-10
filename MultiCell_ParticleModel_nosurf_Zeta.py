#%% MultiCell Particle Model
# with surface chemistry
# and solving zeta
# Not integrated to reactor model
import pdb
import os
import sys
import numpy as np
import cantera as ct
import time
import json
import matplotlib.pyplot as plt

plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['mathtext.rm'] = 'Cambria'
plt.rcParams['font.family'] ='Cambria'
plt.rcParams['savefig.dpi'] = 1200

from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from solarReact_PartModEqs import solarReact_PartModEqs_MultiCellPart
from solarReact_PartModEqs import solarReact_PartModEqs_MultiCellPart_Zeta

functions_folder = os.path.join(".", 'Functions')
sys.path.append(functions_folder)

from ParticleModel_Boundary_flux import ParticleModel_Boundary_flux
from KineticFun import KineticFun

#%% Single particle unpack
def solarReact_SinglePartUnpack(SV, ind, gas, BedParams, SurfParams):
    # Initialize Yk_p Xk_p to hold species mass and mole fractions
    Yk_p = np.zeros((BedParams['n_y'], gas['kspec'], SurfParams['n_p']))
    Xk_p = np.zeros((BedParams['n_y'], gas['kspec'], SurfParams['n_p']))
    Zk_p = np.zeros((BedParams['n_y'], surf['kspec'], SurfParams['n_p']))
    
    for i_p in range(SurfParams['n_p']):
        for i_spec in range(gas['kspec']):
            Yk_p[:, i_spec, i_p] = SV[ind['start'] + int(ind['Yk_p'][i_spec] + (i_p) * (gas['kspec'] ))]

        # Calculate mole fractions for each bed position j_y
        for j_y in range(BedParams['n_y']):
            gas['obj'].Y = Yk_p[j_y, :, i_p]
            Xk_p[j_y, :, i_p] = gas['obj'].X
        
        for i_spec in range(surf['kspec']):
            Zk_p[:, i_spec, i_p] = SV[ind['start'] + ind['Zk_p'][i_spec] + \
                 int((i_p) * (ind['vars'] / SurfParams['n_p']))]
        
    
    return Yk_p, Xk_p, Zk_p

#%% Solver Function that calculates the residual equations
def solarReact_SolverFun(SV):       
    
    # Unpack variables(ParticleModel trial only)
    (Yk_p, Xk_p, Zk_p) = solarReact_SinglePartUnpack(SV, ind, gas, BedParams, SurfParams)
    
    # Particle model governing equations
    partmod_res, jk_b, sdot_g, sdot_g_bulk, eta_cat_k = solarReact_PartModEqs_MultiCellPart_Zeta(gas, gas_surf, surf, GasParams, PartParams, BedParams, SurfParams, ind,  \
                              T_p, Yk_p, P_p, Xk_p, P_bg, T_bg, Yk_bg, Xk_bg, v_bg, v_bs, phi_bg, Zk_p)
        
    residuals = partmod_res
    
    residuals_reshape = residuals.reshape(SurfParams['n_p'], (gas['kspec'] + surf['kspec'] ))
    
    return residuals

#%% Function to construct the Jacobian sparsity matrix
def jacobian_pattern(n_y, vars_per_cell, vars_tot):
    JPat = lil_matrix((vars_tot, vars_tot), dtype=int)
    
    for i_y in range(1, n_y + 1):
        i1_start = (i_y - 1) * vars_per_cell
        i1_end = i_y * vars_per_cell
        
        # Coupling with the previous cell
        if i_y > 1:
            i2_start = (i_y - 2) * vars_per_cell
            i2_end = (i_y - 1) * vars_per_cell
            JPat[i1_start:i1_end, i2_start:i2_end] = 1
            
        # Self-coupling (diagonal)
        i2_start = (i_y - 1) * vars_per_cell
        i2_end = i_y * vars_per_cell
        JPat[i1_start:i1_end, i2_start:i2_end] = 1

        # Coupling with the next cell
        if i_y < n_y:
            i2_start = i_y * vars_per_cell
            i2_end = (i_y + 1) * vars_per_cell
            JPat[i1_start:i1_end, i2_start:i2_end] = 1
            
    return JPat

#%% Define the dictionaries and model parameters
gas = {}
gas_surf = {}
surf = {}
GasParams = {}
BedParams = {}

gas['jreac']    = 4
gas['jDMR']     = 0 
gas['jSMR1']    = 1
gas['jSMR2']    = 2
gas['jWGS']     = 3

BedParams['kinetics'] = 1

# Define constant variables and parameters
PartParams = {'dp': 600e-6}
PartParams['Sherwood'] = 'Gunn'
SurfParams = {}

# Chemistry on or off
SurfParams['chem'] = 1  # 1 = w chem, 2 = no chem
SurfParams['geo'] = 2   # 1 = evenly spaced, 2 = dense outer
SurfParams['integrate'] = 1
SurfParams['delta_t'] = 1.5

if SurfParams['chem'] == 1:
    SurfParams['chem_tag'] = 'w_chem'
elif SurfParams['chem'] == 2:
    SurfParams['chem_tag'] = 'no_chem'

# Define particle parameters
SurfParams['n_p'] = 9
SurfParams['active_radius'] = 0.50

GasParams['id'] = 'reformate-part-nosurf'

file = 2
if file == 1:
    GasParams['filename'] = 'Mechanism/sm_Ni_Rakhi.yaml'
elif file == 2:
    GasParams['filename'] = 'Mechanism/sm_Ni_Delgado.yaml'
elif file == 3:
    GasParams['filename'] = 'Mechanism/sm_Ni_Maier.yaml'

rho_Al = 3.34e3
rho_Ni = 8.9e3

#%% Define surface parameters
SurfParams['Rmax']          = PartParams['dp']/2                                # [m] outer radius of particle 
SurfParams['Rmin']          = (1-SurfParams['active_radius'])*SurfParams['Rmax'] # [m] solid-core (non-porous) radius of particle 
SurfParams['phi']           = 0.3                                               # [-] porosity of the particle 
SurfParams['tau']           = 4.5                                                # [-] tortuosity of the particle 
SurfParams['Rpore']         = 50e-9                                              # [m] pore diameter 
SurfParams['R_Ni']          = 5e-9
SurfParams['cat_loading']   = 0.10
SurfParams['Vpart']         = (4/3)*np.pi*(SurfParams['Rmax']**3 - SurfParams['Rmin']**3)
SurfParams['Apart']         = 4*np.pi*SurfParams['Rmax']**2

SurfParams['Ni_mass']       = SurfParams['cat_loading']*(1/(1-SurfParams['cat_loading']))*rho_Al*(1-SurfParams['phi'])*SurfParams['Vpart']
SurfParams['Ni_vol']        = SurfParams['Ni_mass']/rho_Ni
SurfParams['Ni_vol_ratio']  = SurfParams['Ni_vol']/SurfParams['Vpart']

SurfParams['a_Ni_cat']      = (3)*(SurfParams['Ni_vol_ratio'])/SurfParams['R_Ni']  # [m^-1] specific surface area of catalyst
SurfParams['a_surf']        = 6*SurfParams['phi']/SurfParams['Rpore'] + 6*(1-SurfParams['phi'])/PartParams['dp']

SurfParams['a_cat'] = SurfParams['a_Ni_cat'] #+ SurfParams['a_surf']

# Calculate porous particle properties
SurfParams['vol_ratio'] = SurfParams['phi'] / (1 - SurfParams['phi'])
SurfParams['eff_factor'] = SurfParams['phi'] / SurfParams['tau']
SurfParams['B_g'] = (SurfParams['vol_ratio'] ** 2 * PartParams['dp'] ** 2 *
                     SurfParams['eff_factor'] / 72)           # Permeability based on Kozeny-Carman equation

SurfParams['mcat_per_area'] = 1e3*SurfParams['Ni_mass'] / (4*SurfParams['phi']*SurfParams['Vpart']/SurfParams['Rpore'])

# Calculate geometry of particle
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
    outerPercent    = 0.1
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
SurfParams['rRmax'] = np.zeros(SurfParams['n_p'] + 1)
SurfParams['rRmax'][:SurfParams['n_p']] = SurfParams['rcell'] / SurfParams['Rmax']
SurfParams['rRmax'][SurfParams['n_p']] = 1

#%% Define gas and surface objects
gas['obj'] = ct.Solution(GasParams['filename'], GasParams['id'])
gas['kspec'] = gas['obj'].n_species
gas['Wk'] = gas['obj'].molecular_weights

gas_surf['obj'] = ct.Solution(GasParams['filename'], 'reformate-part')
surf['obj'] = ct.Interface(GasParams['filename'], 'surf', [gas_surf['obj']])
surf['kspec'] = surf['obj'].n_species
surf['Wk'] = surf['obj'].molecular_weights
SurfParams['sitDens'] = surf['obj'].site_density

# Define species indices for the gas phase
gas['kH2'] = gas['obj'].species_index('H2')
gas['kCH4'] = gas['obj'].species_index('CH4')
gas['kH2O'] = gas['obj'].species_index('H2O')
gas['kCO'] = gas['obj'].species_index('CO')
gas['kCO2'] = gas['obj'].species_index('CO2')
if gas['kspec'] == 6:
    gas['kN2'] = gas['obj'].species_index('N2')

# Define species indices for the surface phase
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

# Initialize uniform coverage
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

# Read data
BedParams['n_y'] = 1
Yk_bg = np.zeros((BedParams['n_y'], gas['kspec']))

v_bg = np.array(([0.8]))
v_bs = np.array(([0]))
T_bg = np.array(([800 + 273.15]))
T_bs = np.array(([800 + 273.15]))
phi_bg = np.array(([0.5]))
T_p = np.zeros((BedParams['n_y']))

"""
rat_H2O = 3.0
rat_CO2 = 0.000

vol_N2 = 0/100
rat_N2 = (vol_N2 * (rat_CO2 + rat_H2O + 1)) / (1 - vol_N2)

X_CH4 = 1/(rat_CO2 + rat_H2O + 1 + rat_N2)
"""
X_from_reactor = np.zeros((5, gas['kspec']))
X_from_reactor[0,:] = np.array(([0.00156294, 0.249599, 0.748789, 9.96883e-06, 3.88151e-05]))

X_from_reactor[1,:] = np.array(([0.0952108,	0.209511, 0.669032,	0.0116554, 0.014591]))

X_from_reactor[2,:] = np.array(([0.300039,	0.11605,	0.496477,	0.0244489,	0.0629843]))

X_from_reactor[3,:] = np.array(([0.447298,	0.0463905,	0.372771,	0.0282714,	0.105269]))

X_from_reactor[4,:] = np.array(([0.527048,	0.0080508,	0.305926,	0.0290084,	0.129967]))

i_dat = 0

Xk_bg = np.zeros((BedParams['n_y'], gas['kspec']))
for j_y in range(BedParams['n_y']):
    Xk_bg[j_y, gas['kH2']] = X_from_reactor[i_dat,gas['kH2']] #1e-20
    Xk_bg[j_y, gas['kCH4']] = X_from_reactor[i_dat,gas['kCH4']] #X_CH4
    Xk_bg[j_y, gas['kH2O']] = X_from_reactor[i_dat,gas['kH2O']] #rat_H2O * X_CH4
    Xk_bg[j_y, gas['kCO2']] = X_from_reactor[i_dat,gas['kCO2']] #rat_CO2*X_CH4 
    Xk_bg[j_y, gas['kCO']] = X_from_reactor[i_dat,gas['kCO']] #1e-20
    if gas['kspec'] == 7:
        Xk_bg[j_y, gas['kN2']] = 1 - np.sum(Xk_bg[j_y, :])
    
P_bg = np.array(([10e5]))
P_p = np.zeros((BedParams['n_y']))

# Calculate the relative velocity of gas and particle
U_inf = v_bg + v_bs

# Define indices values
ind = {
    'vars': ((gas['kspec'] + surf['kspec']) * SurfParams['n_p']),
    'tot': ((gas['kspec'] + surf['kspec']) * SurfParams['n_p']) * BedParams['n_y'],
    'start': np.arange(BedParams['n_y']) * ((gas['kspec'] + surf['kspec']) * SurfParams['n_p'])
}

# Define Yk_p and Zk_p indices
ind['Yk_p'] = np.zeros(SurfParams['n_p']*(gas['kspec']), dtype=int)
ind['Zk_p'] = np.zeros(SurfParams['n_p']*(surf['kspec']), dtype=int)
partmod_vars_start = 0
for i_p in range(SurfParams['n_p']):
    ind['Yk_p'][(i_p * (gas['kspec'] )):(i_p * (gas['kspec'] ) + (gas['kspec'] ))] = \
        partmod_vars_start + np.arange(gas['kspec'])
    ind['Zk_p'][(i_p * (surf['kspec'] )):(i_p * (surf['kspec']) + (surf['kspec']))] = \
        partmod_vars_start + (gas['kspec'] ) +  np.arange(surf['kspec'])
    partmod_vars_start += (gas['kspec'] + surf['kspec'] )

# Initial guess based on composition at the reactor's scale
Yk_p_0 = np.zeros((BedParams['n_y'], gas['kspec']))
Xk_p_0 = np.zeros((BedParams['n_y'], gas['kspec']))
Yk_p_eq = np.zeros((BedParams['n_y'], gas['kspec']))
Xk_p_eq = np.zeros((BedParams['n_y'], gas['kspec']))
SurfParams['Zk_p_init'] = np.zeros((BedParams['n_y'], surf['kspec'], SurfParams['n_p']))

for j_y in range(BedParams['n_y']):
    gas['obj'].X = Xk_bg[j_y, :]
    Yk_bg[j_y, :] = gas['obj'].Y
    
    for i_p in range(SurfParams['n_p']):
        T_p[j_y] = T_bs[j_y]
        P_p[j_y] = P_bg[j_y]
        
        # Equilibrium concentration
        gas['obj'].TPY = T_p[j_y], P_p[j_y], Yk_bg[j_y, :]
        gas['obj'].equilibrate('TP')
        Yk_p_0[j_y,:] = gas['obj'].Y
        Xk_p_0[j_y,:] = gas['obj'].X
        Yk_p_eq[j_y,:] = gas['obj'].Y
        Xk_p_eq[j_y,:] = gas['obj'].X
        
        
        if BedParams['kinetics'] == 2:
            gas_surf['obj'].TPY = T_p[j_y], P_p[j_y], np.concatenate((Yk_bg[j_y, :], [0]))
            gas_surf['obj'].equilibrate('TP')
        
            surf['obj'].TP = T_p[j_y], P_p[j_y]
            surf['obj'].coverages = SurfParams['Zk_p_0'][:]
            surf['obj'].advance_coverages(1.5e2, 1e-7, 1e-14, 1e-1, 5e8, 20)    
            SurfParams['Zk_p_init'][j_y, :, i_p] = surf['obj'].coverages

Yk_p_0_lin = np.zeros((SurfParams['n_p'], gas['kspec']))  
for i_spec in range(gas['kspec']):
    Yk_p_0_lin[:, i_spec] = np.linspace(Yk_p_eq[0,i_spec], Yk_bg[0, i_spec], SurfParams['n_p'])
    
# Initialize state vector (SV_0)
SV_0 = np.zeros(ind['tot'])
for i_p in range(SurfParams['n_p']):
    for i_spec in range(gas['kspec']):
        SV_0[ind['start'] + int(ind['Yk_p'][i_spec] + \
             (i_p) * (ind['vars'] / SurfParams['n_p']))] = Yk_p_0[:, i_spec]
    for i_spec in range(surf['kspec']):
        SV_0[ind['start'] + int(ind['Zk_p'][i_spec] + \
             (i_p) * (ind['vars'] / SurfParams['n_p']))] = SurfParams['Zk_p_init'][:, i_spec, i_p]

SV_0_reshape = SV_0.reshape(SurfParams['n_p'], (gas['kspec'] + surf['kspec']))
            
# Construct the residual eqns by calling solarReact_PartModEqs and solarReact_SolverFun
# (Test scripts)
(res_0) =  solarReact_SolverFun(SV_0)

res_0_reshape = res_0.reshape(SurfParams['n_p'], (gas['kspec'] + surf['kspec']))

pdb.set_trace()

boundsLow =  np.zeros((ind['tot']))
boundsUp = np.ones((ind['tot']))

varBounds = ( boundsLow, boundsUp ) 

JPat = jacobian_pattern(SurfParams['n_p'], ind['vars'], ind['tot'])
JPat = JPat.toarray()

solve_type = 1

ftol = 1e-8 
xtol = 1e-10
max_nfev = 100
status_list = [5]
# Status:
    # -1 : improper input parameters status returned from MINPACK.
    # 0 : the maximum number of function evaluations is exceeded.
    # 1 : gtol termination condition is satisfied.
    # 2 : ftol termination condition is satisfied.
    # 3 : xtol termination condition is satisfied.2
    # 4 : Both ftol and xtol termination conditions are satisfied.
stat = 0
N_run = 0
max_loop = 1
x = SV_0
N_Fev = 0

t1 = time.time()

while stat not in status_list and N_run < max_loop:
    
    if solve_type == 1:
        # least_squares WITH JACOBIAN pattern, forces the use of "lsmr" trust-region solver
        Solution = least_squares(solarReact_SolverFun, x, bounds=varBounds, \
                         ftol = ftol, xtol = xtol, jac_sparsity = jacobian_pattern(SurfParams['n_p'], ind['vars'], ind['tot']), \
                             max_nfev = max_nfev, verbose = 2)
    
    elif solve_type == 2:
        # least_squares WITHOUT JACOBIAN pattern method: 'trf', 'dogbox'
        Solution = least_squares(solarReact_SolverFun, x, bounds=varBounds, method='trf', \
                         ftol = ftol, xtol = xtol, max_nfev = max_nfev, verbose = 2)
    
    elif solve_type == 3:
        # least_squares without Jacobian pattern method: 'lm -> does not support Bounds'
        Solution = least_squares(solarReact_SolverFun, x, method = 'lm', ftol = ftol, \
                             xtol = xtol, jac_sparsity = None, max_nfev = max_nfev, verbose = 2)
    
    stat = Solution['status']
    N_run += 1
    x = Solution['x']
    N_Fev = N_Fev + Solution['nfev']
    print( f"Total iteration loop = {N_run :.0f}" )
    print( f"Total function evaluation = {N_Fev :.0f}" )
    
    # Re-run if not converged
    if Solution['status'] not in status_list and N_run < max_loop:
        print( '\n', 'Solution not yet converged, re-starting run' )
    else:
        print( '\n', 'Ending run' )

t2 = time.time()
print( '\n', 'Time to solve = ', t2-t1, ' secs' )

#%% Unpack and plot 
(Yk_p_sol, Xk_p_sol, Zk_p_sol) = solarReact_SinglePartUnpack(Solution['x'], ind, gas, BedParams, SurfParams)

Yk_p_bound_sol = np.zeros((BedParams['n_y'], gas['kspec'], 1))
Xk_p_bound_sol = np.zeros((BedParams['n_y'], gas['kspec'], 1))

for i_spec in range(gas['kspec']):
    Yk_p_bound_sol[0, i_spec, 0] = (Yk_p_sol[0, i_spec, -1] + Yk_bg[0, i_spec])/2
    Xk_p_bound_sol[0, i_spec, 0] = (Xk_p_sol[0, i_spec, -1] + Xk_bg[0, i_spec])/2

Yk_p_sol = np.concatenate((Yk_p_sol, Yk_p_bound_sol), axis = 2)
Xk_p_sol = np.concatenate((Xk_p_sol, Xk_p_bound_sol), axis = 2)

#%% Calculate jk_bound
jk_b_sol = np.zeros((BedParams['n_y'], gas['kspec']))
for j_y in range(BedParams['n_y']):
    jk_b_sol[j_y, :], _, _ = ParticleModel_Boundary_flux(gas, PartParams, P_p[j_y], \
                                                         Yk_p_sol[j_y, :, SurfParams['n_p'] - 1], \
                                                         T_p[j_y], P_bg[j_y], \
                                                         Yk_bg[j_y, :], T_bg[j_y], U_inf[j_y], 0.5)

#%% Calculate jk_bound
"""
jk_b_sol = np.zeros((BedParams['n_y'], gas['kspec']))
for j_y in range(BedParams['n_y']):
    jk_b_sol[j_y, :], _, _ = ParticleModel_Boundary_flux(gas, PartParams, P_p[j_y, SurfParams['n_p'] - 1], \
                                                         Yk_p_sol[j_y, :, SurfParams['n_p'] - 1], \
                                                             T_p[j_y, SurfParams['n_p'] - 1], P_bg[j_y], \
                                                                 Yk_bg[j_y, :], T_bg[j_y], U_inf[j_y], 0.5)

#%% Calculate Damkohler number
# Average properties at the boundary
b = {}
b['P_b'] =  0.5 * (P_p[0, SurfParams['n_p'] - 1] + P_bg[0])
b['Yk_b'] = 0.5 * (Yk_p_sol[0, :, SurfParams['n_p'] - 1] + Yk_bg[0, :])
b['T_b'] = 0.5 * (T_p[0, SurfParams['n_p'] - 1] + T_bg[0])

# Set the gas properties at boundary conditions
gas['obj'].TPY = b['T_b'], b['P_b'], b['Yk_b']
b['rho_mole_b'] = gas['obj'].density_mole
D_k_b = gas['obj'].mix_diff_coeffs
b['rho_b'] = gas['obj'].density
mu_b = gas['obj'].viscosity                      # Gas viscosity [Pa s]
c_p_b = gas['obj'].cp_mass                       # Heat capacity [J/kg-K]
b['Xk_b'] = gas['obj'].X
k_b = gas['obj'].thermal_conductivity

# Reynolds number
Re = b['rho_b'] * U_inf * PartParams['dp'] / mu_b
# Schmidt number
Sc = (mu_b / (b['rho_b'] * D_k_b))
# Sherwood number correlation
Sh_p = 2 + 0.6 * Re**0.5 * Sc**(1/3)

k_m_g = np.zeros((gas['kspec']))
for k in range(gas['kspec']):
    k_m_g[k]     = Sh_p[k] * D_k_b[k] / PartParams['dp'] 

# Reaction
sdot_g = np.zeros((BedParams['n_y'], gas['kspec'], SurfParams['n_p']))
sdot_g_avg = np.zeros((BedParams['n_y'], gas['kspec']))
Yk_p_sol_avg = np.zeros((BedParams['n_y'], gas['kspec']))
C_k = np.zeros((BedParams['n_y'], gas['kspec']))
k_eff = np.zeros((BedParams['n_y'], gas['kspec']))

for i_y in range(BedParams['n_y']):
    for i_p in range(SurfParams['n_p']):
        # Set the gas and surface phase object and retrieve properties
        gas['obj'].set_unnormalized_mass_fractions(Yk_p_sol[i_y, :, i_p])
        gas['obj'].TP = T_p[i_y, i_p], P_p[i_y, i_p]
        
        # Gas species production rate from catalytic reaction from global mechanism
        sdot_g[i_y, :, i_p] = KineticFun(ind, gas, Yk_p_sol[i_y, :, i_p], Xk_p_sol[i_y, :, i_p], T_p[i_y], P_p[i_y], SurfParams)
    
    for i_spec in range(gas['kspec']):
        sdot_g_avg[i_y, i_spec] = np.average(sdot_g[i_y, i_spec, :])
        Yk_p_sol_avg[i_y, i_spec] = np.average(Yk_p_sol[i_y, i_spec, :])
        C_k[i_y, i_spec] = Yk_p_sol_avg[i_y, i_spec] * P_p[i_y, 0] / (8.314e3 * T_p[i_y, 0])
        k_eff[i_y, i_spec] = sdot_g_avg[i_y, i_spec] / C_k[i_y, i_spec]

Da_p = k_eff[0,:]/k_m_g[:]
"""
#%% Process equilibrium composition
rRmax_eq_plot = [SurfParams['rRmax'][0], 1]
Xk_p_eq_plot = np.array([Xk_p_eq, Xk_p_eq])
        
#%%
lnsz = 1.5  # Line width
fsz = 15    # Font size
fsz2 = 10    # legend size
fsz3 = 13   # axis  label size
lwd = 2     # Line width for the axes

#%% plot gas phase mole fraction
for j_y in range(BedParams['n_y']):
    plt.figure(j_y+1, figsize = (4,5))
    n_cell = j_y+1
    ax = plt.gca()
    ax.plot(SurfParams['rRmax'], Xk_p_sol[j_y, gas['kH2'], :], 'r-', linewidth=lnsz)
    ax.plot(SurfParams['rRmax'], Xk_p_sol[j_y, gas['kCH4'], :], 'y-', linewidth=lnsz)
    ax.plot(SurfParams['rRmax'], Xk_p_sol[j_y, gas['kCO'], :], 'k-', linewidth=lnsz)
    ax.plot(SurfParams['rRmax'], Xk_p_sol[j_y, gas['kCO2'], :], 'g-', linewidth=lnsz)
    ax.plot(SurfParams['rRmax'], Xk_p_sol[j_y, gas['kH2O'], :], 'b-', linewidth=lnsz)
    
    ax.plot(rRmax_eq_plot, Xk_p_eq_plot[:, 0, gas['kH2']], 'r--', linewidth=lnsz)
    ax.plot(rRmax_eq_plot, Xk_p_eq_plot[:, 0, gas['kCH4']], 'y--', linewidth=lnsz)
    ax.plot(rRmax_eq_plot, Xk_p_eq_plot[:, 0, gas['kCO']], 'k--', linewidth=lnsz)
    ax.plot(rRmax_eq_plot, Xk_p_eq_plot[:, 0, gas['kCO2']], 'g--', linewidth=lnsz)
    ax.plot(rRmax_eq_plot, Xk_p_eq_plot[:, 0, gas['kH2O']], 'b--', linewidth=lnsz)
    
    #plt.title('Reactor cell n (from bottom) = {}'.format(n_cell))
    ax.set_ylabel('Gas species mole fraction ${X}_{\mathrm{k,p}}$ [$\mathrm{-}$]',fontsize=fsz)
    ax.set_xlabel('$r / R_{\mathrm{max}}$ [$\mathrm{-}$]',fontsize=fsz)
    #ax.set_ylim(0, 0.35)
    #ax.set_yticks([0,0.2,0.4,0.6,0.8])
    ax.tick_params(axis = 'x', labelsize=fsz3)
    ax.tick_params(axis = 'y', labelsize=fsz3)
    ax.legend(['$\mathrm{H_{2}}$','$\mathrm{CH_{4}}$', \
            '$\mathrm{CO}$','$\mathrm{CO_{2}}$','$\mathrm{H_{2}O}$'],fontsize=fsz2,loc='best', \
            ncol=2)
"""
BedParams['y'] = np.array([0.05, 0.10])
# plot jk,b in and out of particle
plt.figure(3, figsize = (4,5))
ax = plt.gca()
ax.plot(jk_b_sol[:, gas['kH2']], BedParams['y'], 'r-', linewidth=lnsz)
ax.plot(jk_b_sol[:, gas['kCH4']], BedParams['y'], 'y-', linewidth=lnsz)
ax.plot(jk_b_sol[:, gas['kCO']], BedParams['y'], 'k-', linewidth=lnsz)
ax.plot(jk_b_sol[:, gas['kCO2']], BedParams['y'], 'g-', linewidth=lnsz)
ax.plot(jk_b_sol[:, gas['kH2O']], BedParams['y'], 'b-', linewidth=lnsz)
#ax.axhline(y = BedParams['y'][BedParams['index_fb'][0]], color='k', linestyle='--')
#ax.text(0.5, 0.46, 'Freeboard zone', fontsize = fsz2)
ax.set_ylabel('Position [m]', fontsize=fsz)
ax.set_xlabel('Species $\ j_{\mathrm{k,b}}$ [$\mathrm{kg \ m^{-2} \ s^{-1}}$]', fontsize=fsz)
ax.set_xlim([-0.5, 0.5])
ax.legend(['$\mathrm{H_{2}}$','$\mathrm{CH_{4}}$', \
         '$\mathrm{CO}$','$\mathrm{CO_{2}}$','$\mathrm{H_{2}O}$'],fontsize=fsz2,loc='best', \
         ncol=2)
"""

for j_y in range(BedParams['n_y']):
    plt.figure(2+j_y+1, figsize = (4,5))
    n_cell = j_y+1
    ax = plt.gca()
    ax.plot(SurfParams['rRmax'][:-1], Zk_p_sol[j_y, surf['kO'], :], '-', color = 'c',  linewidth=lnsz, label = '$\mathrm{O(s)}$')
    ax.plot(SurfParams['rRmax'][:-1], Zk_p_sol[j_y, surf['kCO'], :], '-', color = 'k', linewidth=lnsz, label = '$\mathrm{CO(s)}$')
    ax.plot(SurfParams['rRmax'][:-1], Zk_p_sol[j_y, surf['kH'], :], '-', color = 'r', linewidth=lnsz, label = '$\mathrm{H(s)}$')
    ax.plot(SurfParams['rRmax'][:-1], Zk_p_sol[j_y, surf['kNi'], :], '-', color = 'cornflowerblue', linewidth=lnsz, label = '$\mathrm{Ni(s)}$')
    
    #plt.title('Reactor cell n (from bottom) = {}'.format(n_cell))
    ax.set_ylabel('Surface species site fraction ${Z}_{\mathrm{k,p}}$ [$\mathrm{-}$]',fontsize=fsz)
    ax.set_xlabel('$r / R_{\mathrm{max}}$ [$\mathrm{-}$]',fontsize=fsz)
    ax.tick_params(axis = 'x', labelsize=fsz3)
    ax.tick_params(axis = 'y', labelsize=fsz3)
    ax.legend(fontsize=fsz2,loc='best', ncol=2)

plt.show()

#%% Save output file that collects "Result" dict
Xk_p_sol =  Xk_p_sol.tolist()
Xk_p_eq = Xk_p_eq.tolist()
Yk_p_sol =  Yk_p_sol.tolist()
Yk_p_eq = Yk_p_eq.tolist()
SurfParams['rRmax'] = SurfParams['rRmax'].tolist()

save = {
        "Xk_p_sol"      : Xk_p_sol,
        "Xk_p_eq"       : Xk_p_eq,
        "Yk_p_sol"      : Yk_p_sol,
        "Yk_p_eq"       : Yk_p_eq,
        "rRmax"         : SurfParams['rRmax'],
        "n_y"           : BedParams['n_y'],
        "n_p"           : SurfParams['n_p'],
        "kspec"         : gas['kspec'],
        "chem_tag"      : SurfParams['chem_tag']
    }

#%%
file_path = os.path.join(".",'Data_Files', 'Particle_Model_Files')
file_name_to_save = 'PartMod' + '_' + 'dp' + '-' + f"{1e6*PartParams['dp']:.0f}" + '_' + 'phi' + '-' + f"{SurfParams['phi']:.2f}" + \
                    '_' + 'tau' + '-' + f"{SurfParams['tau']:.1f}" + '_' + 'Rpore' + '-' + f"{1e6*SurfParams['Rpore']:.3f}" + \
                    '_' + 'Tbg' + '-' + f"{T_bg[0]-273.15:.0f}" + '_' + 'Pbg' + '-' + f"{P_bg[0]/1e5:.0f}" + \
                    '_' + 'vbg' + '-' + f"{v_bg[0]:.1f}" + '_' + 'catloading' + '-' + f"{SurfParams['cat_loading']:.2f}" #+ \
                    #'_' + 'ratH2O' + '-' + f"{rat_H2O:.2f}" + 'ratCO2' + '-' + f"{rat_CO2:.2f}" + SurfParams['chem_tag']

#with open(os.path.join(file_path, file_name_to_save + ".json"), 'w') as file_out:
#    json.dump(save, file_out)
     