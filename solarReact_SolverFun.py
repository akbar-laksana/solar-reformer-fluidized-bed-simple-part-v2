from solarReact_Unpack import solarReact_Unpack
from solarReact_Unpack import solarReact_Unpack_Part_y

from solarReact_PartModEqs import solarReact_PartModEqs_SimplePart
from solarReact_PartModEqs import solarReact_PartModEqs_MultiCellPart

from solarReact_FluidizedBedEqs import solarReact_FluidizedBedEqs
from solarReact_WallEqs import solarReact_WallEqs

from solarReact_WallEqs import solarReact_WallEqs_Isothermal

import matplotlib.pyplot as plt

#%% Solver Function that combines all the residual equations of wall, bed, and particle
def solarReact_SolverFun(SV, gas, gas_surf, part, wall, surf, env, GasParams, PartParams, BedParams, WallParams, FinParams, SurfParams, EnvParams, ind, scale):
    
    # Reshape SV for easier inspection
    SV_reshape = SV.reshape(BedParams['n_y'], ind['vars'])
    
    # Unpack variables
    if PartParams['simple_part'] == 1:
        [T_we, T_wb, T_bs, T_bg, phi_bs, P_bg, phi_bg, Yk_bg, Xk_bg, T_p, Yk_p, Yk_p_int, P_p, Xk_p, Xk_p_int, v_bg, v_bs] \
            = solarReact_Unpack(SV, gas, part, GasParams, PartParams, BedParams, SurfParams, ind, scale)
    elif PartParams['multi_part'] == 1:
        [T_we, T_wb, T_bs, T_bg, phi_bs, P_bg, phi_bg, Yk_bg, Xk_bg, T_p, Yk_p, P_p, Xk_p, v_bg, v_bs] \
            = solarReact_Unpack(SV, gas, part, GasParams, PartParams, BedParams, SurfParams, ind, scale)
            
    # Receiver-Reactor wall heat transfer governing equations
    if BedParams['energy'] == 1:
        wall_res = solarReact_WallEqs(wall, env, BedParams, WallParams, PartParams, EnvParams, ind, T_we, T_wb)
    
    # Particle model governing equations
    if PartParams['simple_part'] == 1:
        part_res, jk_bound, sdot_g = solarReact_PartModEqs_SimplePart(gas, gas_surf, surf, GasParams, PartParams, BedParams, SurfParams, ind, \
                                    T_p, Yk_p, Yk_p_int, P_p, Xk_p, Xk_p_int, P_bg, T_bg, Yk_bg, v_bg, v_bs, phi_bg, phi_bs)
    elif PartParams['multi_part'] == 1:
        part_res, jk_bound, sdot_g, sdot_g_bulk, eta_cat_k = solarReact_PartModEqs_MultiCellPart(gas, gas_surf, surf, GasParams, PartParams, BedParams, SurfParams, ind, \
                                                    T_p, Yk_p, P_p, Xk_p, P_bg, T_bg, Yk_bg, Xk_bg, v_bg, v_bs, phi_bg)
    
    # Fluidized bed transport and heat transfer governing equations
    bed_res = solarReact_FluidizedBedEqs(ind, gas, part, wall, surf, WallParams, FinParams, BedParams, \
                                   PartParams, GasParams, SurfParams, EnvParams, T_we, T_wb, T_bs, T_bg, phi_bs, \
                                   phi_bg, P_bg, Yk_bg, Xk_bg, T_p, Yk_p, P_p, Xk_p, v_bg, v_bs, jk_bound, sdot_g)    
    
    # Residual equations
    if BedParams['energy'] == 1:
        residuals = wall_res + part_res + bed_res
    else:
        residuals = part_res + bed_res
    
    # Scale the residual equations
    if scale['solver_run'] == 1: # 1st time the solver is run, use original scaling
        residuals = residuals / scale['res']
    elif scale['solver_run'] == 2: # 2st (or more) time the solver is run, use adjusted scaling
        residuals = residuals / scale['res_2']    
    
    # Reshape residuals for easier inspection
    residuals_reshape = residuals.reshape(BedParams['n_y'], ind['vars'])
    
    return residuals

#%% Solver function for particle at a single height only
def solarReact_SolverFun_Part_y(SV, gas, gas_surf, part, wall, surf, env, GasParams, PartParams, BedParams, WallParams, FinParams, SurfParams, EnvParams, ind, scale, Result_y):
         
    # Unpack variables(ParticleModel trial only)
    (Yk_p, Xk_p) = solarReact_Unpack_Part_y(SV, ind, gas, BedParams, SurfParams)
    
    # Particle model governing equations
    part_res, jk_bound, sdot_g, sdot_g_bulk, eta_cat_k = solarReact_PartModEqs_MultiCellPart(gas, gas_surf, surf, GasParams, PartParams, BedParams, SurfParams, ind,  \
                              Result_y['T_p'], Yk_p, Result_y['P_p'], Xk_p, Result_y['P_bg'], Result_y['T_bg'], \
                              Result_y['Yk_bg'], Result_y['Xk_bg'], Result_y['v_bg'], Result_y['v_bs'], Result_y['phi_bg'])
    
    residuals = part_res
    
    return residuals