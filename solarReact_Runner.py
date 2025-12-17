#%% solarReact_Runner
#   Run a single case by: 1. Reading the input (.json) file, 2. Initialize case,
#                         3. Solve res eqs with non-linear equation solver

import numpy as np
import cantera as ct
import json
import time

# Import scripts for Model Solving
from solarReact_Init import solarReact_Init
from solarReact_Init_Multi import solarReact_Init_Multi
from solarReact_Init_Multi import solarReact_Init_Part_y

from solarReact_SolverFun import solarReact_SolverFun
from solarReact_SolverFun import solarReact_SolverFun_Part_y

from solarReact_ExtractPlotter import solarReact_ExtractPlotter
from solarReact_ExtractPlotter import solarReact_ExtractPlotter_Part_y
from solarReact_ExtractPlotter import solarReact_Plotter_Part

from solarReact_Saver import solarReact_Saver

# Import numerical solver and matrix for jacobian
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

#%% Function to construct the Jacobian sparsity matrix
def jacobian_pattern(n_y, vars_per_cell, vars_tot):
    JPat = lil_matrix((vars_tot, vars_tot), dtype=int)
    
    # Set coupled diagonal
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

#%% Main function that runs solarReact_Runner
def solarReact_Runner(InFilePath, SolFilePath, FileName, Restart):
    
    with open(InFilePath, 'r') as file_in:
        read = json.load(file_in) 
    
    # Re-distribute
    gas             = read['gas']
    gas_surf        = read['gas_surf']
    part            = read['part']
    wall            = read['wall']
    surf            = read['surf']
    env             = read['env']
    GasParams       = read['GasParams']
    PartParams      = read['PartParams']
    BedParams       = read['BedParams']
    WallParams      = read['WallParams']
    FinParams       = read['FinParams']
    SurfParams      = read['SurfParams']
    EnvParams       = read['EnvParams']    

    #%% Re-convert list from json to numpy array
    if BedParams['geo'] == 1:
        BedParams['dz_b']       = np.array([BedParams['dz_b']]).reshape(len(BedParams['dz_b']))
        BedParams['dx_b']       = np.array([BedParams['dx_b']]).reshape(len(BedParams['dx_b']))
        BedParams['dz']         = np.array([BedParams['dz']]).reshape(len(BedParams['dz']))
        
        WallParams['dz_b']      = np.array([WallParams['dz_b']]).reshape(len(WallParams['dz_b']))
        WallParams['dx_b']      = np.array([WallParams['dx_b']]).reshape(len(WallParams['dx_b'])) 
        WallParams['dz']        = np.array([WallParams['dz']]).reshape(len(WallParams['dz']))
        
        BedParams['Az_in_b']    = np.array([BedParams['Az_in_b']]).reshape(len(BedParams['Az_in_b']))
        BedParams['Az_out_b']   = np.array([BedParams['Az_out_b']]).reshape(len(BedParams['Az_out_b']))
        BedParams['Az_in']      = np.array([BedParams['Az_in']]).reshape(len(BedParams['Az_in']))
        BedParams['Az_out']     = np.array([BedParams['Az_out']]).reshape(len(BedParams['Az_out']))
        BedParams['Az_out_tot_b'] = np.array([BedParams['Az_out_tot_b']]).reshape(len(BedParams['Az_out_tot_b']))
        
        WallParams['Az_in_b']   = np.array([WallParams['Az_in_b']]).reshape(len(WallParams['Az_in_b']))
        WallParams['Az_out_b']  = np.array([WallParams['Az_out_b']]).reshape(len(WallParams['Az_out_b']))
        WallParams['Az']        = np.array([WallParams['Az']]).reshape(len(WallParams['Az']))
        
    elif BedParams['geo'] == 2:
        BedParams['d_in_b']     = np.array([BedParams['d_in_b']]).reshape(len(BedParams['d_in_b']))
        BedParams['d_out_b']    = np.array([BedParams['d_out_b']]).reshape(len(BedParams['d_out_b']))
        BedParams['d_in']       = np.array([BedParams['d_in']]).reshape(len(BedParams['d_in']))
        BedParams['d_out']      = np.array([BedParams['d_out']]).reshape(len(BedParams['d_out']))
        
        WallParams['d_in_b']    = np.array([WallParams['d_in_b']]).reshape(len(WallParams['d_in_b']))
        WallParams['d_out_b']   = np.array([WallParams['d_out_b']]).reshape(len(WallParams['d_out_b']))
        WallParams['d_in']      = np.array([WallParams['d_in']]).reshape(len(WallParams['d_in']))
        WallParams['d_out']     = np.array([WallParams['d_out']]).reshape(len(WallParams['d_out']))
    
        BedParams['Ar_in_b']    = np.array([BedParams['Ar_in_b']]).reshape(len(BedParams['Ar_in_b']))
        BedParams['Ar_out_b']   = np.array([BedParams['Ar_out_b']]).reshape(len(BedParams['Ar_out_b']))
        BedParams['Ar_in']      = np.array([BedParams['Ar_in']]).reshape(len(BedParams['Ar_in']))
        BedParams['Ar_out']     = np.array([BedParams['Ar_out']]).reshape(len(BedParams['Ar_out']))
        BedParams['Ar_out_tot_b'] = np.array([BedParams['Ar_out_tot_b']]).reshape(len(BedParams['Ar_out_tot_b']))
        
        WallParams['Ar_in_b']   = np.array([WallParams['Ar_in_b']]).reshape(len(WallParams['Ar_in_b']))
        WallParams['Ar_out_b']  = np.array([WallParams['Ar_out_b']]).reshape(len(WallParams['Ar_out_b']))
    
    BedParams['Ay']         = np.array([BedParams['Ay']]).reshape(len(BedParams['Ay']))
    BedParams['Ay_b']       = np.array([BedParams['Ay_b']]).reshape(len(BedParams['Ay_b']))
    BedParams['Ay_cond']    = np.array([BedParams['Ay_cond']]).reshape(len(BedParams['Ay_cond']))
    
    WallParams['Ay']        = np.array([WallParams['Ay']]).reshape(len(WallParams['Ay']))
    WallParams['Ay_b']      = np.array([WallParams['Ay_b']]).reshape(len(WallParams['Ay_b']))
    WallParams['Ay_cond']   = np.array([WallParams['Ay_cond']]).reshape(len(WallParams['Ay_cond']))
    
    BedParams['D_hyd']      = np.array([BedParams['D_hyd']]).reshape(len(BedParams['D_hyd']))
    BedParams['D_hyd_b']    = np.array([BedParams['D_hyd_b']]).reshape(len(BedParams['D_hyd_b']))

    BedParams['dVol']       = np.array([BedParams['dVol']]).reshape(len(BedParams['dVol']))
    BedParams['dVol_b']     = np.array([BedParams['dVol_b']]).reshape(len(BedParams['dVol_b']))

    WallParams['dVol']      = np.array([WallParams['dVol']]).reshape(len(WallParams['dVol']))
    WallParams['dVol_b']    = np.array([WallParams['dVol_b']]).reshape(len(WallParams['dVol_b']))  

    BedParams['dy']         = np.array([BedParams['dy']]).reshape(len(BedParams['dy']))
    BedParams['dy_b']       = np.array([BedParams['dy_b']]).reshape(len(BedParams['dy_b']))
    BedParams['dy_cond']    = np.array([BedParams['dy_cond']]).reshape(len(BedParams['dy_cond']))
    BedParams['index_b']    = np.array([BedParams['index_b']]).reshape(len(BedParams['index_b']))
    BedParams['y']          = np.array([BedParams['y']]).reshape(len(BedParams['y']))
    BedParams['y_b']        = np.array([BedParams['y_b']]).reshape(len(BedParams['y_b']))
    BedParams['y_bnd']      = np.array([BedParams['y_bnd']]).reshape(len(BedParams['y_bnd']))  
    
    WallParams['dy']        = np.array([WallParams['dy']]).reshape(len(WallParams['dy']))
    WallParams['dy_b']      = np.array([WallParams['dy_b']]).reshape(len(WallParams['dy_b']))
    WallParams['dy_cond']   = np.array([WallParams['dy_cond']]).reshape(len(WallParams['dy_cond']))
    WallParams['y']         = np.array([WallParams['y']]).reshape(len(WallParams['y']))
    WallParams['y_b']       = np.array([WallParams['y_b']]).reshape(len(WallParams['y_b']))
    
    gas['Wk']               = np.array([gas['Wk']]).reshape(len(gas['Wk']))
    GasParams['X_in']       = np.array([GasParams['X_in']]).reshape(len(GasParams['X_in']))
    GasParams['Xk_out']     = np.array([GasParams['Xk_out']]).reshape(len(GasParams['Xk_out']))
    GasParams['Y_in']       = np.array([GasParams['Y_in']]).reshape(len(GasParams['Y_in']))
    GasParams['Yk_out']     = np.array([GasParams['Yk_out']]).reshape(len(GasParams['Yk_out']))
    GasParams['hk_in']      = np.array([GasParams['hk_in']]).reshape(len(GasParams['hk_in']))
    
    env['Wk']               = np.array([env['Wk']]).reshape(len(env['Wk']))
    
    EnvParams['q_sol_dist'] = np.array([EnvParams['q_sol_dist']]).reshape(len(EnvParams['q_sol_dist']))
    
    if SurfParams['n_p'] > 1:
        SurfParams['Aface'] = np.array([SurfParams['Aface']]).reshape(len(SurfParams['Aface']))
        SurfParams['deltar'] = np.array([SurfParams['deltar']]).reshape(len(SurfParams['deltar']))
        SurfParams['rcell'] = np.array([SurfParams['rcell']]).reshape(len(SurfParams['rcell']))
        SurfParams['rface'] = np.array([SurfParams['rface']]).reshape(len(SurfParams['rface']))
        SurfParams['rRmax'] = np.array([SurfParams['rRmax']]).reshape(len(SurfParams['rRmax']))
        SurfParams['Vcell'] = np.array([SurfParams['Vcell']]).reshape(len(SurfParams['Vcell']))
    SurfParams['Zk_p_0']    = np.array([SurfParams['Zk_p_0']]).reshape(len(SurfParams['Zk_p_0']))
    
    #%% Cantera object re-definition since it can't be serialized through .json file
    gas['obj']      = ct.Solution(GasParams['filename'], GasParams['id'])
    gas_surf['obj'] = ct.Solution(GasParams['filename'], 'reformate-part')
    surf['obj']     = ct.Interface(GasParams['filename'], 'surf', [gas_surf['obj']])
    wall['obj']     = ct.Solution('Mechanism/CARBO_air.yaml', WallParams['id'])
    part['obj']     = ct.Solution('Mechanism/CARBO_air.yaml', PartParams['id'])
    env['obj']      = ct.Solution('Mechanism/CARBO_air.yaml', 'air')
    
    #%% Setup the SV_guess of the solver

    # Initialize SV
    if PartParams['simple_part'] == 1:    
        [SV_guess, ind, scale, Bounds] = solarReact_Init(gas, gas_surf, part, wall, surf, GasParams, PartParams, BedParams, WallParams, SurfParams, EnvParams)
    elif PartParams['multi_part'] == 1:
        [SV_guess, ind, scale, Bounds] = solarReact_Init_Multi(gas, gas_surf, part, wall, surf, GasParams, PartParams, BedParams, WallParams, SurfParams, EnvParams)
    
    # Reshape SV for easier inspection
    SV_guess_reshape = SV_guess.reshape(BedParams['n_y'], ind['vars'])
    
    # Set x = SV_guess for the initial value of the SolverFun
    x = SV_guess
    
    # Test Jac Pat
    JPat = jacobian_pattern(BedParams['n_y'], ind['vars'], ind['tot'])
    JPat_full = JPat.toarray() 
    
    # (Test scripts) Setup solverfunction that calculates residual functions
    scale['solver_run'] = 1
    (res_0) =  solarReact_SolverFun(SV_guess, gas, gas_surf, part, wall, surf, env, GasParams, PartParams, BedParams, WallParams, FinParams, SurfParams, EnvParams, ind, scale)
    
    # Reshape initial residual value for easier inspection
    res_0_reshape = res_0.reshape(BedParams['n_y'], ind['vars'])
    
    breakpoint()
    
    #%% Solver Setup
    jacobian = 1 # 1: with user-defined jacobian, 2: with finite-difference solver jacobian
    ftol = 1e-10
    xtol = 1e-12
    max_nfev = 200
    status_list = [4]
    # Status:
        # -1 : improper input parameters status returned from MINPACK.
        # 0 : the maximum number of function evaluations is exceeded.
        # 1 : gtol termination condition is satisfied.
        # 2 : ftol termination condition is satisfied.
        # 3 : xtol termination condition is satisfied.2
        # 4 : Both ftol and xtol termination conditions are satisfied.
    stat = 0
    N_run = 0
    N_Fev = 0
    max_loop = 1
    
    #%% 1st solver run with original scaling
    scale['solver_run'] = 1
    t1 = time.time()
    print( '\n', '###### Starting main run ######', '\n' )
    while stat not in status_list and N_run < max_loop:
        # least_squares WITH JACOBIAN pattern, forces the use of "lsmr" trust-region solver
        if jacobian == 1:
            Solution = least_squares(solarReact_SolverFun, x, args=(gas, gas_surf, part, wall, surf, env, GasParams, PartParams, \
                        BedParams, WallParams, FinParams, SurfParams, EnvParams, ind, scale), bounds=Bounds, ftol = ftol, \
                        xtol = xtol, jac_sparsity = jacobian_pattern(BedParams['n_y'], ind['vars'], ind['tot']), \
                        max_nfev = max_nfev, verbose = 2)
        
        else:
            Solution = least_squares(solarReact_SolverFun, x, args=(gas, gas_surf, part, wall, surf, env, GasParams, PartParams, \
                        BedParams, WallParams, FinParams, SurfParams, EnvParams, ind, scale), bounds=Bounds, ftol = ftol, \
                        xtol = xtol, max_nfev = max_nfev, verbose = 2)
        
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
            print( '\n', '****** Ending main run ******' )
            scale['solver_run'] = 2
            
    t2 = time.time()
    print( '\n', 'Time to solve main run = ', t2-t1, ' secs', '\n' )
    
    #%% Restart functionalities (2nd or more solver run)
    if Restart == True:
        N_run = 0
        N_Fev = 0
        max_loop = 1
        max_nfev = 100
        t1 = time.time()
        print( '\n', '###### Starting restart run ######', '\n' )
        
        while stat not in status_list and N_run < max_loop:
            # least_squares WITH JACOBIAN pattern, forces the use of "lsmr" trust-region solver
            if jacobian == 1:
                Solution = least_squares(solarReact_SolverFun, x, args=(gas, gas_surf, part, wall, surf, env, GasParams, PartParams, \
                        BedParams, WallParams, FinParams, SurfParams, EnvParams, ind, scale), bounds=Bounds, ftol = ftol, \
                        xtol = xtol, jac_sparsity = jacobian_pattern(BedParams['n_y'], ind['vars'], ind['tot']), \
                        max_nfev = max_nfev, verbose = 2)
        
            else:
                Solution = least_squares(solarReact_SolverFun, x, args=(gas, gas_surf, part, wall, surf, env, GasParams, PartParams, \
                        BedParams, WallParams, FinParams, SurfParams, EnvParams, ind, scale), bounds=Bounds, ftol = ftol, \
                        xtol = xtol, max_nfev = max_nfev, verbose = 2)
    
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
                print( '\n', '****** Ending restart run ******' )
                
        t2 = time.time()
        print( '\n', 'Time to solve restart run = ', t2-t1, ' secs', '\n' )
    
    #%% Post Process and Plot
    Result = solarReact_ExtractPlotter(gas, gas_surf, surf, part, wall, GasParams, PartParams, BedParams, WallParams, FinParams, SurfParams, EnvParams, ind, scale, Solution, SolFilePath)
    
    #%% Solve a high resolution particle model for selected reactor height
    max_nfev_Part_y = 100 #150
    
    # Pick desired reactor height point out of the total BedParams['n_y']
    Part_y_0    = 0
    Part_y_005  = np.ceil(0.05*BedParams['n_y'])
    Part_y_01   = np.ceil(0.1*BedParams['n_y'])
    Part_y_02   = np.ceil(0.2*BedParams['n_y'])
    Part_y_025  = np.ceil(0.25*BedParams['n_y'])
    Part_y_03   = np.ceil(0.3*BedParams['n_y'])
    Part_y_04   = np.ceil(0.4*BedParams['n_y'])
    Part_y_05   = np.ceil(0.5*BedParams['n_y'])
    Part_y_06   = np.ceil(0.6*BedParams['n_y'])
    Part_y_07   = np.ceil(0.7*BedParams['n_y'])
    Part_y_075  = np.ceil(0.75*BedParams['n_y'])
    Part_y_08   = np.ceil(0.8*BedParams['n_y'])
    Part_y_09   = np.ceil(0.9*BedParams['n_y'])
    Part_y_095  = np.ceil(0.95*BedParams['n_y'])
    Part_y_1    = BedParams['n_y'] - 1
    """
    Part_y      = np.array([Part_y_0, Part_y_005, Part_y_01, Part_y_02, Part_y_025, Part_y_03, \
                            Part_y_04, Part_y_05, Part_y_06, Part_y_07, Part_y_075, Part_y_08, \
                            Part_y_09, Part_y_095, Part_y_1], dtype=int)
    """
    Part_y      = np.array([Part_y_0,Part_y_025, \
                            Part_y_05, Part_y_075, \
                            Part_y_1], dtype=int)
        
    # Compile height array for multicell particle model result
    Bed_Part_y = np.zeros((len(Part_y)))
        
    Result_Part_y = {}
    Result_Part_y['Yk_p'] = np.zeros((len(Part_y), gas['kspec'], SurfParams['n_p'] + 1))
    Result_Part_y['Xk_p'] = np.zeros((len(Part_y), gas['kspec'], SurfParams['n_p'] + 1))    
    Result_Part_y['Xk_p_eq'] = np.zeros((len(Part_y), gas['kspec']))
    Result_Part_y['jk_b'] = np.zeros((len(Part_y), gas['kspec']))
    Result_Part_y['eta_cat_k'] = np.zeros((len(Part_y), gas['kspec']))
    
    print( '\n', '###### Starting multi cell particle model run ######' )

    # Loop over selected 
    for i_y in range(len(Part_y)):
        
        print( '\n', f"Particle model at y = {BedParams['y'][Part_y[i_y]]:.2f} [m]", '\n' )
        
        Result_y = {}
        
        Result_y['T_p'] = np.array(([ Result['T_bs'][Part_y[i_y]] ]))
        Result_y['P_p'] = np.array(([ Result['P_bg'][Part_y[i_y]] ]))
        Result_y['P_bg'] = np.array(([ Result['P_bg'][Part_y[i_y]] ]))
        Result_y['T_bg'] = np.array(([ Result['T_bg'][Part_y[i_y]] ]))
        Result_y['Yk_bg'] = np.zeros((1, gas['kspec']))
        Result_y['Yk_bg'][0,:] = Result['Yk_bg'][Part_y[i_y],:]
        Result_y['Xk_bg'] = np.zeros((1, gas['kspec']))
        Result_y['Xk_bg'][0,:]  = Result['Xk_bg'][Part_y[i_y],:]
        Result_y['v_bg']= np.array(([ Result['v_bg'][Part_y[i_y]] ])) 
        Result_y['v_bs'] = np.array(([ Result['v_bs'][Part_y[i_y]] ]))
        Result_y['phi_bg'] = np.array(([ Result['phi_bg'][Part_y[i_y]] ]))
        
        Bed_Part_y[i_y] = BedParams['y'][Part_y[i_y]]
        
        # Setup SV_guess through initialization
        SV_guess_Part_y, ind_Part_y, Bounds_Part_y, Xk_p_eq = solarReact_Init_Part_y(gas, gas_surf, part, wall, surf, GasParams, PartParams, BedParams, WallParams, SurfParams, EnvParams, Result_y)
    
        x = SV_guess_Part_y
    
        # least_squares WITH JACOBIAN pattern, forces the use of "lsmr" trust-region solver
        Solution_Part_y = least_squares(solarReact_SolverFun_Part_y, x, args=(gas, gas_surf, part, wall, surf, env, GasParams, \
                PartParams, BedParams, WallParams, FinParams, SurfParams, EnvParams, ind_Part_y, scale, Result_y), bounds=Bounds_Part_y, \
                ftol = ftol, xtol = xtol, jac_sparsity = jacobian_pattern(SurfParams['n_p'], ind_Part_y['vars'], ind_Part_y['tot']), \
                max_nfev = max_nfev_Part_y, verbose = 2)
        
        Result_Part_y['Yk_p'][i_y, :, :],  Result_Part_y['Xk_p'][i_y, :, :], Result_Part_y['Xk_p_eq'][i_y, :], Result_Part_y['jk_b'][i_y, :], \
            Result_Part_y['eta_cat_k'][i_y, :] = solarReact_ExtractPlotter_Part_y(gas, gas_surf, surf, PartParams, BedParams, SurfParams, ind_Part_y, \
            Solution_Part_y, SolFilePath, Result_y, Part_y, i_y, Xk_p_eq)    
    
    # Plot comparison of eta_cat and jk_b from simple particle and multi cell particle model
    solarReact_Plotter_Part(gas, BedParams, SolFilePath, Result, Result_Part_y, Bed_Part_y)
    
    #%% Save
    solarReact_Saver(gas, gas_surf, part, wall, surf, env, GasParams, PartParams, BedParams, WallParams, FinParams, SurfParams, EnvParams, Result, SolFilePath)
    
    return gas, gas_surf, part, wall, surf, env, GasParams, PartParams, BedParams, WallParams, FinParams, EnvParams, ind, scale, Result
    
    #%%
    breakpoint()