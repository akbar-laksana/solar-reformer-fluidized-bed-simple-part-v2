import numpy as np
import sys
import os
import matplotlib.pyplot as plt

functions_folder = os.path.join(".", 'Functions')
sys.path.append(functions_folder)

from Thermal_Cond import Thermal_Cond
from FB_Cond import FB_Cond
from solarReact_GasProps import solarReact_GasProps
from solarReact_SolidProps import solarReact_SolidProps
from ParticleModel_Boundary_flux import ParticleModel_Boundary_flux
from hT_wb_conv_v2 import hT_wb_conv_v2
from FinCalcs import FinCalcs
from KineticFun import KineticFun

from solarReact_Unpack import solarReact_Unpack
from solarReact_Unpack import solarReact_Unpack_Part_y

plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['mathtext.rm'] = 'Cambria'
plt.rcParams['font.family'] ='Cambria'
plt.rcParams['savefig.dpi'] = 1200

def solarReact_ExtractPlotter(gas, gas_surf, surf, part, wall, GasParams, PartParams, BedParams, WallParams, FinParams, SurfParams, EnvParams, ind, scale, Solution, SolFilePath):
    #%% Unpack solution
    Result = {}
        
    # Unpack the result from SV result
    if PartParams['simple_part'] == 1:
        [Result['T_we'], Result['T_wb'], Result['T_bs'], Result['T_bg'], Result['phi_bs'], \
         Result['P_bg'], Result['phi_bg'], Result['Yk_bg'], Result['Xk_bg'], Result['T_p'], \
         Result['Yk_p'], Result['Yk_p_int'], Result['P_p'], Result['Xk_p'], Result['Xk_p_int'], \
         Result['v_bg'], Result['v_bs']] = \
        solarReact_Unpack(Solution['x'], gas, part, GasParams, PartParams, BedParams, SurfParams, ind, scale)
    
        # Calculate jk_bound
        Result['jk_b'] = np.zeros((BedParams['n_y'], gas['kspec']))
        Result['Jk_b'] = np.zeros((BedParams['n_y'], gas['kspec']))
        for i_y in range(BedParams['n_y']):
            Result['jk_b'][i_y, :], _, _ = ParticleModel_Boundary_flux(gas, PartParams, Result['P_p'][i_y], \
                                                                       Result['Yk_p_int'][i_y, :], \
                                                                       Result['T_p'][i_y], Result['P_bg'][i_y], \
                                                                       Result['Yk_bg'][i_y, :], Result['T_bg'][i_y], \
                                                                       (Result['v_bg'][i_y] + Result['v_bs'][i_y]), Result['phi_bg'][i_y])
            for i_spec in range(gas['kspec']):
                Result['Jk_b'][i_y, i_spec] = Result['jk_b'][i_y, i_spec]/gas['Wk'][i_spec]
    
    elif PartParams['multi_part'] == 1:
        [Result['T_we'], Result['T_wb'], Result['T_bs'], Result['T_bg'], Result['phi_bs'], \
         Result['P_bg'], Result['phi_bg'], Result['Yk_bg'], Result['Xk_bg'], Result['T_p'], \
         Result['Yk_p'], Result['P_p'], Result['Xk_p'], Result['v_bg'], Result['v_bs']] = \
        solarReact_Unpack(Solution['x'], gas, part, GasParams, PartParams, BedParams, SurfParams, ind, scale)
    
        Result['Yk_p_bound'] = np.zeros((BedParams['n_y'], gas['kspec'], 1))
        Result['Xk_p_bound'] = np.zeros((BedParams['n_y'], gas['kspec'], 1))
        for i_spec in range(gas['kspec']):
            Result['Yk_p_bound'][:, i_spec, 0] = (Result['Yk_p'][:, i_spec, -1] + Result['Yk_bg'][:, i_spec])/2
            Result['Xk_p_bound'][:, i_spec, 0] = (Result['Xk_p'][:, i_spec, -1] + Result['Xk_bg'][:, i_spec])/2
        
        # Calculate jk_bound
        Result['jk_b'] = np.zeros((BedParams['n_y'], gas['kspec']))
        Result['Jk_b'] = np.zeros((BedParams['n_y'], gas['kspec']))
        for i_y in range(BedParams['n_y']):
            Result['jk_b'][i_y, :], _, _ = ParticleModel_Boundary_flux(gas, PartParams, Result['P_p'][i_y], \
                                                                   Result['Yk_p'][i_y, :, -1], \
                                                                   Result['T_p'][i_y], Result['P_bg'][i_y], \
                                                                   Result['Yk_bg'][i_y, :], Result['T_bg'][i_y], \
                                                                   (Result['v_bg'][i_y] + Result['v_bs'][i_y]), Result['phi_bg'][i_y])
            for i_spec in range(gas['kspec']):
                Result['Jk_b'][i_y, i_spec] = Result['jk_b'][i_y, i_spec]/gas['Wk'][i_spec]
    
    #%% Grab and Calculate Reactor Performance metrics:
    #%% Grab max wall temperature
    Result['T_w_max'] = np.max(Result['T_we'])
    
    #%% Calculate gas temperature increase
    T_bg_in = GasParams['T_in']
    T_bg_out = Result['T_bg'][-1]
    Result['Delta_T_bg'] = T_bg_out - T_bg_in 
    
    #%% Calculate gas pressure drop
    P_bg_in = GasParams['P_in']
    P_bg_out = Result['P_bg'][-1]
    Result['Delta_P_bg'] = P_bg_in - P_bg_out
    
    #%% Wall correlations
    # Wall radiative heat terms
    # Wall solar radiation input and emissive radiation from the wall
    sigma = 5.67E-8
    if BedParams['geo'] == 1:
        q_rad_in  = WallParams['Az_out_b'] * WallParams['abs'] * EnvParams['q_sol']
        q_rad_out = EnvParams['f'] * sigma * WallParams['Az_out_b'] * WallParams['emis'] * (Result['T_we'][BedParams['index_b']]**4 - EnvParams['T']**4)
    elif BedParams['geo'] == 2:
        q_rad_in  = WallParams['Ar_out_b'] * WallParams['abs'] * EnvParams['q_sol']
        q_rad_out = EnvParams['f'] * sigma * WallParams['Ar_out_b'] * WallParams['emis'] * (Result['T_we'][BedParams['index_b']]**4 - EnvParams['T']**4)
    
    # Wall convection to environment
    T_f = (Result['T_we'] + EnvParams['T'])/2
    g = 9.81
    n_y = BedParams['n_y']

    rho_inf = np.zeros(n_y)
    mu_inf  = np.zeros(n_y)
    C_p_inf = np.zeros(n_y)
    k_inf   = np.zeros(n_y)
    
    for i_y in range(n_y):
        gas['obj'].TP = T_f[i_y], EnvParams['P']
    
        rho_inf[i_y]    = gas['obj'].density
        mu_inf[i_y]     = gas['obj'].viscosity
        C_p_inf[i_y]    = gas['obj'].cp_mass
        k_inf[i_y]      = gas['obj'].thermal_conductivity

    k_we                = Thermal_Cond(WallParams['id'],Result['T_we'])
    k_wb                = Thermal_Cond(WallParams['id'],Result['T_wb'])

    k_we_avg            = 0.5*(k_we[:-1] + k_we[1:])
    k_wb_avg            = 0.5*(k_wb[:-1] + k_wb[1:])

    k_w_z               = Thermal_Cond(WallParams['id'],0.5*(Result['T_we']+Result['T_wb']))

    # Coefficient of volume expansion, thermal diffusivity, and kinematic viscosity
    beta    = 1 / T_f                           # [1/K]
    alpha   = k_inf / (rho_inf * C_p_inf)       # Thermal diffusivity [m^2/s]
    nu      = mu_inf / rho_inf                  # Kinematic viscosity [m^2/s]

    # Rayleigh Number
    Ra = g * beta * (Result['T_we'] - EnvParams['T']) * WallParams['y']**3 / (alpha * nu)
    
    # Nusselt Number
    Nus = np.zeros(n_y)
    for i_y in range(n_y):
        if Ra[i_y] < 1e7:
            Nus[i_y] = 0.59 * Ra[i_y]**0.25
        else:
            Nus[i_y] = 0.1 * Ra[i_y]**0.33

    # Heat transfer coefficient of external wall
    h = Nus * k_inf / WallParams['y']
    h[0] = h[1]

    # Convective heat transfer to environment
    if BedParams['geo'] == 1:
        q_conv = h[BedParams['index_b']] * WallParams['Az_out_b'] * (Result['T_we'][BedParams['index_b']] - EnvParams['T'])
    elif BedParams['geo'] == 2:
        q_conv = h[BedParams['index_b']] * WallParams['Ar_out_b'] * (Result['T_we'][BedParams['index_b']] - EnvParams['T'])
        
    #%% Average heat transfer in each reactor section
    q_net = q_rad_in - q_rad_out - q_conv
    if BedParams['geo'] == 1:
        Result['q_avg'] = np.sum(q_net)/np.sum(WallParams['Az_out_b'])
    elif BedParams['geo'] == 2:
        Result['q_avg'] = np.sum(q_net)/np.sum(WallParams['Ar_out_b'])
    
    #%% Bed correlation
    rho_g   = np.zeros(n_y)
    h_g     = np.zeros(n_y)
    mu_g    = np.zeros(n_y)
    cp_g    = np.zeros(n_y)
    k_g     = np.zeros(n_y)
    rho_s   = np.zeros(n_y)
    cp_s    = np.zeros(n_y)
    h_s     = np.zeros(n_y)
    rho_w   = np.zeros(n_y)
    cp_w    = np.zeros(n_y)

    dp              = PartParams['dp']                                            
    phi_mf          = PartParams['phi_bs_max']                                     
    phi_bg_avg      = np.concatenate(([Result['phi_bg'][0]], 0.5*(Result['phi_bg'][:-1] + Result['phi_bg'][1:]), [Result['phi_bg'][-1]]))

    for i_y in range(n_y):
        gas['obj'].TP   = Result['T_bg'][i_y], Result['P_bg'][i_y]                              
        rho_g[i_y]      = gas['obj'].density                                    
        h_g[i_y]        = gas['obj'].enthalpy_mass                              
        mu_g[i_y]       = gas['obj'].viscosity                                  
        cp_g[i_y]       = gas['obj'].cp_mass                                     
        k_g[i_y]        = gas['obj'].thermal_conductivity
        
        part['obj'].TP  = Result['T_bs'][i_y], Result['P_bg'][i_y]                    
        rho_s[i_y]      = part['obj'].density*(1-SurfParams['phi'])                                    
        cp_s[i_y]       = part['obj'].cp_mass                                  
        h_s[i_y]        = part['obj'].enthalpy_mass                          
        
        wall['obj'].TP  = 0.5*(Result['T_we'][i_y] + Result['T_wb'][i_y]), EnvParams['P']
        rho_w[i_y]      = wall['obj'].density                                   
        cp_w[i_y]       = wall['obj'].cp_mass                                  
    
    k_s             = Thermal_Cond(PartParams['id'], Result['T_bs'])                       

    # Estimate effective bed conduction
    k_g_avg         = np.concatenate(([k_g[0]], 0.5*(k_g[:-1] + k_g[1:]), [k_g[-1]]))      
    k_s_avg         = np.concatenate(([k_s[0]], 0.5*(k_s[:-1] + k_s[1:]), [k_s[-1]]))
    k_s_eff_avg     = FB_Cond(phi_bg_avg, k_s_avg/k_g_avg) * k_g_avg           

    #%% Calculate U_hat
    # Calculate minimum fluidization, superficial, and dimensionless excess gas velocities
    a               = 1.75*phi_mf*rho_g/(dp*(1-phi_mf)**3)                   
    b               = 150*phi_mf**2*mu_g/(dp**2*(1-phi_mf)**3)              
    c               = -(phi_mf*rho_s - (1-phi_mf)*rho_g)*g                  
    
    U_mf            = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)                     
    Ug              = Result['v_bg'] * Result['phi_bg']                                           
    Us              = Result['v_bs'] * Result['phi_bs']                                         
    Result['U_hat'] = (rho_s*cp_s/(g*k_g))**(1/3)*(Ug - U_mf)
    Result['U_hat_avg'] = np.average(Result['U_hat'])

    # Calculate laminar Archimedes numbers and heat transfer functionity
    Ar_lam          = np.sqrt(dp**3 * g) * (rho_s - rho_g) / mu_g             
    Pr_gs           = k_g / ( 2 * cp_s * mu_g )                                                   
    
    # Calculate particle-wall heat transfer coefficient with convective and radiative components
    hT_wb_conv      = hT_wb_conv_v2(Result['U_hat'], Ar_lam, Pr_gs, dp, k_g)                                      
    emis_hx         = 1 /(1/WallParams['emis'] + 1/PartParams['emis'] - 1)           
    hT_wb_rad       = 5.67e-8 * emis_hx *(Result['T_bs']**2 + Result['T_wb']**2)*(Result['T_bs'] + Result['T_wb'])          
    
    #%% Calculate heat transfer with fin
    F_fin = np.zeros(BedParams['n_y'])
    
    # Wall with fin
    if BedParams['fin'] == 1:
        for i_y in range(BedParams['n_y']):    
            [_, _, F_fin[i_y]] = FinCalcs(Result['T_wb'][i_y] , hT_wb_conv[i_y], FinParams)
            
        # Enhanced wall-bed heat transfer coefficient with fin
        Result['hT_wb'] = (F_fin * hT_wb_conv) + hT_wb_rad
        
        if BedParams['geo'] == 1:
            q_wb            = Result['hT_wb'] * BedParams['Az_out'] * (Result['T_wb'] - Result['T_bs'])
        elif BedParams['geo'] == 2:    
            q_wb            = Result['hT_wb'] * BedParams['Ar_out'] * (Result['T_wb'] - Result['T_bs'])        
        
        Result['hT_wb_no_fin'] = hT_wb_conv + hT_wb_rad
        
    # Finless wall
    else:
        Result['hT_wb'] = hT_wb_conv + hT_wb_rad
        
        if BedParams['geo'] == 1:
            q_wb            = Result['hT_wb'] * BedParams['Az_out'] * (Result['T_wb'] - Result['T_bs'])
        elif BedParams['geo'] == 2:    
            q_wb            = Result['hT_wb'] * BedParams['Ar_out'] * (Result['T_wb'] - Result['T_bs']) 
    
    # Average heat transfer coefficient
    Result['hT_wb_avg']  = np.average(Result['hT_wb'])
    
    #%% Species calculation
    # Inlet condition
    Yk_bg_in   = GasParams['Y_in']
    phi_bg_in  = Result['phi_bg'][0]
    v_bg_in    = Result['v_bg'][0]
    
    # Outlet condition
    Yk_bg_out   = Result['Yk_bg'][-1,:]
    phi_bg_out  = Result['phi_bg'][-1]
    v_bg_out    = Result['v_bg'][-1]
    
    gas['obj'].TPY = T_bg_out, P_bg_out, Yk_bg_out
    rho_g_out = gas['obj'].density
    hk_g_out = gas['obj'].standard_enthalpies_RT*gas['R']*T_bg_out/gas['Wk']
    h_g_out = gas['obj'].enthalpy_mass

    gas['obj'].TPY = T_bg_in, P_bg_in, Yk_bg_in
    hk_g_in = gas['obj'].standard_enthalpies_RT*gas['R']*T_bg_in/gas['Wk']
    h_g_in = gas['obj'].enthalpy_mass
    
    mdot_g_in  = GasParams['mdot_in']
    mdotk_g_in = Yk_bg_in * mdot_g_in#[:, np.newaxis]
    
    mdot_g_out  = phi_bg_out * rho_g_out * (v_bg_out*BedParams['Ay'][-1])
    mdotk_g_out = Yk_bg_out * mdot_g_out#[:, np.newaxis]

    DH_syn       = mdotk_g_out[gas['kH2']]*hk_g_out[gas['kH2']] + \
        mdotk_g_out[gas['kCO']]*hk_g_out[gas['kCO']] - mdot_g_in * h_g_in
    
    if BedParams['geo'] == 1:    
        q_avail = BedParams['n_w'] * WallParams['Az_out_b'] * EnvParams['q_sol']
    elif BedParams['geo'] == 2:
        q_avail = BedParams['n_w'] * WallParams['Ar_out_b'] * EnvParams['q_sol']
    
    DH_gas = mdot_g_out * h_g_out - mdot_g_in * h_g_in
    #DH_gas = mdotk_g_out * hk_g_out - mdotk_g_in * hk_g_in
    
    #%% Efficiency calculation
    Result['eta_sol_to_syn'] = abs(DH_syn)/np.sum(abs(q_avail))
    Result['eta_sol'] = abs(DH_gas) / np.sum(abs(q_avail))
    Result['eta_th_1'] = np.sum(abs(q_net))/np.sum(q_avail)
    Result['eta_th_2'] = np.sum(abs(q_wb))/np.sum(q_avail)
    
    #%% Calculate conversion
    # Inlet
    gas['obj'].TPX = GasParams['T_in'], GasParams['P_in'], GasParams['X_in']
    MW_in = gas['obj'].mean_molecular_weight
    
    # Outlet
    gas['obj'].TPX = Result['T_bg'][-1], Result['P_bg'][-1], Yk_bg_out
    MW_out = gas['obj'].mean_molecular_weight
    
    n_rat = MW_in / MW_out
    
    Result['CH4_conv'] = (GasParams['X_in'][gas['kCH4']] - \
                              n_rat * Result['Xk_bg'][-1, gas['kCH4']]) \
                             / GasParams['X_in'][gas['kCH4']]
                             
    #%% Gas Flows calculation
    Result['mdot_in'] = GasParams['mdot_in']
    mdot_CH4_in = GasParams['mdot_in']*GasParams['Y_in'][gas['kCH4']]
    gas['obj'].TPY = GasParams['T_in'], GasParams['P_in'], 'CH4:1'
    rho_CH4_inlet = gas['obj'].density
    Result['voldot_CH4_in'] = mdot_CH4_in/rho_CH4_inlet*6e4 # [in LPM]
    
    Result['mdot_out'] = mdot_g_out
    rho_g_out = rho_g[-1]
    Result['voldot_g_out'] = mdot_g_out/rho_g_out*6e4 # [in LPM]
    
    #%% Reactor time-space calculation
    Result['space_time'] = BedParams['Vol_tot']/GasParams['Vdot_in']
    
    #%% Catalyst effectiveness factor calculation
    """
    # Initialize species production and consumption rate
    sdot_g = np.zeros((BedParams['n_y'], gas['kspec'], SurfParams['n_p']))
    sdot_g_bulk = np.zeros((BedParams['n_y'], gas['kspec']))
    
    # Define reaction rate arrays for catalyst effectiveness factor calculation
    Result['eta_cat_k']   = np.zeros((n_y, gas['kspec']))
    integ_sdot_g_p = np.zeros(gas['kspec'])
   
    if BedParams['kinetics'] == 1:
        R_j_p       = np.zeros((n_y, gas['jreac'], SurfParams['n_p']))
        R_j_bulk    = np.zeros((n_y, gas['jreac']))
        Result['eta_cat_j']   = np.zeros((n_y, gas['jreac']))
        integ_Rdot_j_p = np.zeros(gas['jreac'])
    
    # Calculate production rate
    for i_y in range(BedParams['n_y']):
        
        if BedParams['kinetics'] == 1:
            # Reaction rate at bulk condition
            sdot_g_bulk[i_y, :], R_j_bulk[i_y, :] = KineticFun(ind, gas, Result['Xk_bg'][i_y, :], Result['T_bg'][i_y], Result['P_bg'][i_y], SurfParams)
        elif BedParams['kinetics'] == 2:
            # Gas species production rate from detailed surface chemistry
            gas_surf['obj'].set_unnormalized_mass_fractions(np.concatenate((Result['Yk_bg'][i_y, :], [0])))
            gas_surf['obj'].TP = Result['T_bg'][i_y], Result['P_bg'][i_y]
            
            # Set the surface species composition and integrate to steady state
            surf['obj'].set_unnormalized_coverages(SurfParams['Zk_p_init'][i_y, :, 0])
            surf['obj'].TP = Result['T_bg'][i_y], Result['P_bg'][i_y]
            if SurfParams['integrate'] == 1:
                surf['obj'].advance_coverages(SurfParams['delta_t'], 1e-7, 1e-14, 1e-1, 1e8, 20)
        
            sdot_g_bulk[i_y, :] = surf['obj'].get_net_production_rates('reformate-part')[:-1]
        
        # Set the gas species
        gas['obj'].TPY = Result['T_p'][i_y], Result['P_p'][i_y], Result['Yk_p'][i_y, :]
            
        if BedParams['kinetics'] == 1:
            # Gas species production rate from catalytic reaction from global mechanism
            # and Reaction rate at particle condition
            sdot_g[i_y, :], R_j_p[i_y, :] = KineticFun(ind, gas, Result['Xk_p'][i_y, :], Result['T_p'][i_y], Result['P_p'][i_y], SurfParams)
            
        elif BedParams['kinetics'] == 2:
            # Gas species production rate from detailed surface chemistry
            # Set the gas and surface phase object and retrieve properties
            gas_surf['obj'].set_unnormalized_mass_fractions(np.concatenate((Result['Yk_p'][i_y, :], [0])))
            gas_surf['obj'].TP = Result['T_p'][i_y], Result['P_p'][i_y]
                    
            # Set the surface species composition and integrate to steady state
            surf['obj'].set_unnormalized_coverages(SurfParams['Zk_p_init'][i_y, :])
            surf['obj'].TP = Result['T_p'][i_y], Result['P_p'][i_y]
            if SurfParams['integrate'] == 1:
                surf['obj'].advance_coverages(SurfParams['delta_t'], 1e-7, 1e-14, 1e-1, 1e8, 20)
            
            sdot_g[i_y, :] = surf['obj'].get_net_production_rates('reformate-part')[:-1]
    
    
    # Catalyst effectivenes factor = local reaction rate x volume / bulk reaction rate x volume
    for i_y in range(n_y):
        for i_spec in range(gas['kspec']):
            integ_sdot_g_p[i_spec] = 0
            for i_p in range(SurfParams['n_p']):
                integ_sdot_g_p[i_spec] = integ_sdot_g_p[i_spec] + \
                    ((sdot_g[i_y, i_spec, i_p] * SurfParams['rface'][i_p+1]**2 + sdot_g[i_y, i_spec, i_p] * SurfParams['rface'][i_p]**2)/2) * \
                    (SurfParams['rface'][i_p+1] - SurfParams['rface'][i_p])
 
            # Effectiveness factor for each species at different bed locations
            Result['eta_cat_k'][i_y, i_spec] = 3 * (integ_sdot_g_p[i_spec])/(SurfParams['Rmax']**3 * sdot_g_bulk[i_y, i_spec])
            
        if BedParams['kinetics'] == 1:
            for i_reac in range(gas['jreac']):
                integ_Rdot_j_p[i_reac] = 0
                for i_p in range(SurfParams['n_p']):
                    integ_Rdot_j_p[i_reac] = integ_Rdot_j_p[i_reac] + \
                        ((R_j_p[i_y, i_reac, i_p] * SurfParams['rface'][i_p+1]**2 + R_j_p[i_y, i_reac, i_p] * SurfParams['rface'][i_p]**2)/2) * \
                        (SurfParams['rface'][i_p+1] - SurfParams['rface'][i_p])
     
                # Effectiveness factor for each species at different bed locations
                Result['eta_cat_j'][i_y, i_reac] = 3 * (integ_Rdot_j_p[i_reac])/(SurfParams['Rmax']**3 * R_j_bulk[i_y, i_reac])
    """
    #%% Print key result
    print(f"Maximum wall temperature = {Result['T_w_max'] - 273.15:.2f} [\degree C]")
    print(f"Gas temperature increase = {Result['Delta_T_bg']:.2f} [K]")
    print(f"Gas pressure drop = {Result['Delta_P_bg'] / 1e3:.2f} [kPa]")
    print(f"Average U_hat = {Result['U_hat_avg']:.2f} [-]")
    print(f"Average particle-wall heat transfer coeff. = {Result['hT_wb_avg']:.2f} [W m^{-2} K^{-1}]")
    print(f"Average heat flux = {Result['q_avg']  / 1e3:.2f} [kW m^{-2}]")
    print(f"Thermal efficiency q_net = {Result['eta_th_1'] * 100:.2f} [%]")
    print(f"Thermal efficiency q_wb = {Result['eta_th_2'] * 100:.2f} [%]")
    print(f"Solar to syngas efficiency = {Result['eta_sol_to_syn'] * 100:.2f} [%]")
    print(f"Solar efficiency = {Result['eta_sol'] * 100:.2f} [%]")
    print(f"Methane conversion = {Result['CH4_conv'] * 100:.2f} [%]")
    print(f"Inlet gas mass flow rate = {Result['mdot_in']:.5f} [kg s^{-1}]")
    print(f"Inlet methane flow rate = {Result['voldot_CH4_in']:.2f} [LPM]")
    print(f"Outlet gas mass flow rate = {Result['mdot_out']:.5f} [kg s^{-1}]")
    print(f"Outlet gas flow rate = {Result['voldot_g_out']:.2f} [LPM]")
    print(f"Reactor space-time = {Result['space_time']:.2f} [s]")
       
    #%% plot gas phase mole fraction
    # Plot
    lnsz = 1.5  # Line width
    fsz = 15    # Font size
    fsz2 = 10    # legend size
    fsz3 = 13   # axis  label size
    lwd = 2     # Line width for the axes
    
    save = 1
    
    # Plot reactor condition
    # plot temperature
    plt.figure(figsize = (4.75,5))
    ax = plt.gca()
    ax.plot(Result['T_bg'] - 273.15, BedParams['y'], 'g-', label='Fluidizing Gas', linewidth=lnsz)
    ax.plot(Result['T_bs'] - 273.15, BedParams['y'], 'b--', label='Catalyst Particle', linewidth=lnsz)
    ax.plot(Result['T_wb'] - 273.15, BedParams['y'], 'r-', label='Internal Wall', linewidth=lnsz)
    ax.plot(Result['T_we'] - 273.15, BedParams['y'], 'k--', label='External Wall', linewidth=lnsz)
    ax.set_ylabel('Position [m]', fontsize=fsz)
    ax.set_xlabel('Temperature [°C]', fontsize=fsz)
    ax.set_xlim([600, 1000])
    ax.set_ylim([0, BedParams['y'][-1]])
    ax.legend(loc='best', fontsize=fsz2)
    if save == 1:
        plt.savefig(SolFilePath + '/T_b-T_wb-T_we.jpg',bbox_inches='tight', dpi = 900)
    
    # plot velocity
    plt.figure(figsize = (4.75,5))
    ax = plt.gca()
    ax.plot(Result['v_bg'], BedParams['y'], 'b-', label='Fluidizing Gas [m/s]', linewidth=lnsz)
    ax.plot(Result['v_bs']*100, BedParams['y'], 'k-', label='Catalyst Particle [cm/s]', linewidth=lnsz)
    ax.set_ylabel('Position [m]', fontsize=fsz)
    ax.set_xlabel('Velocity [m/s or cm/s]', fontsize=fsz)
    ax.set_ylim([0, BedParams['y'][-1]])
    ax.legend(loc='best', fontsize=fsz2)
    if save == 1:
        plt.savefig(SolFilePath + '/v_bg-v_bs.jpg',bbox_inches='tight', dpi = 900)
    
    # plot pressure
    plt.figure(figsize = (4.75,5))
    ax = plt.gca()
    ax.plot(Result['P_bg']/1e5, BedParams['y'], 'b-', linewidth=lnsz)
    ax.set_ylabel('Position [m]', fontsize=fsz)
    ax.set_xlabel('Pressure [bar]', fontsize=fsz)
    ax.set_xlim([0.9*Result['P_bg'][0]/1e5, 1.1*Result['P_bg'][-1]/1e5])
    ax.set_ylim([0, BedParams['y'][-1]])
    if save == 1:
        plt.savefig(SolFilePath + '/P_bg.jpg',bbox_inches='tight', dpi = 900)
    
    # plot volume fraction
    plt.figure(figsize = (4.75,5))
    ax = plt.gca()
    ax.plot(Result['phi_bs'], BedParams['y'], 'b-', label='$\phi_{\mathrm{bs}}$', linewidth=lnsz)
    ax.plot(Result['phi_bg'], BedParams['y'], 'k--', label='$\phi_{\mathrm{bg}}$', linewidth=lnsz)
    ax.set_ylabel('Position [m]', fontsize=fsz)
    ax.set_xlabel('Volume fraction [-]', fontsize=fsz)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, BedParams['y'][-1]])
    ax.legend(loc='best', fontsize=fsz2)
    if save == 1:
        plt.savefig(SolFilePath + '/phi_bs-phi_bg.jpg',bbox_inches='tight', dpi = 900)
    
    # plot Xk_bg of reactor
    plt.figure(5, figsize = (4.75,5))
    ax = plt.gca()
    ax.plot(Result['Xk_bg'][:, gas['kH2']], BedParams['y'], 'r-', linewidth=lnsz)
    ax.plot(Result['Xk_bg'][:, gas['kCH4']], BedParams['y'], 'y-', linewidth=lnsz)
    ax.plot(Result['Xk_bg'][:, gas['kCO']], BedParams['y'], 'k-', linewidth=lnsz)
    ax.plot(Result['Xk_bg'][:, gas['kCO2']], BedParams['y'], 'g-', linewidth=lnsz)
    xkbg = ax.plot(Result['Xk_bg'][:, gas['kH2O']], BedParams['y'], 'b-', linewidth=lnsz)
    
    ax.plot(Result['Xk_p'][:, gas['kH2']], BedParams['y'], 'r--', linewidth=lnsz)
    ax.plot(Result['Xk_p'][:, gas['kCH4']], BedParams['y'], 'y--', linewidth=lnsz)
    ax.plot(Result['Xk_p'][:, gas['kCO']], BedParams['y'], 'k--', linewidth=lnsz)
    ax.plot(Result['Xk_p'][:, gas['kCO2']], BedParams['y'], 'g--', linewidth=lnsz)
    xkp = ax.plot(Result['Xk_p'][:, gas['kH2O']], BedParams['y'], 'b--', linewidth=lnsz)
    
    ax.plot(Result['Xk_p_int'][:, gas['kH2']], BedParams['y'], 'r:', linewidth=lnsz)
    ax.plot(Result['Xk_p_int'][:, gas['kCH4']], BedParams['y'], 'y:', linewidth=lnsz)
    ax.plot(Result['Xk_p_int'][:, gas['kCO']], BedParams['y'], 'k:', linewidth=lnsz)
    ax.plot(Result['Xk_p_int'][:, gas['kCO2']], BedParams['y'], 'g:', linewidth=lnsz)
    xkpint = ax.plot(Result['Xk_p_int'][:, gas['kH2O']], BedParams['y'], 'b:', linewidth=lnsz)
    
    ax.set_ylabel('Position [m]', fontsize=fsz)
    ax.set_xlabel('Species Xk_bg [-]', fontsize=fsz)
    ax.set_xlim([0, 0.8])
    ax.set_ylim([0, BedParams['y'][-1]])
    spec = ax.legend(['$\mathrm{H_{2}}$', '$\mathrm{CH_{4}}$', '$\mathrm{CO}$', \
               '$\mathrm{CO_{2}}$','$\mathrm{H_{2}O}$'],fontsize=fsz2,loc='best', ncol=2)
    
    loc = ax.legend([xkbg, xkpint, xkp], ['$X_{\mathrm{k,b,g}}$','$X_{\mathrm{k,p}}$','$X_{\mathrm{k,p,int}}$'],\
              fontsize=fsz2,loc='best', ncol=2)
    
    ax.add_artist(spec)
    #ax.add_artist(loc)
    
    if save == 1:
        plt.savefig(SolFilePath + '/Xk_bg.jpg',bbox_inches='tight', dpi = 900)
    
    # plot jk,b in and out of particle
    plt.figure(figsize = (4.75,5))
    ax = plt.gca()
    ax.plot(Result['jk_b'][:, gas['kH2']], BedParams['y'], 'r-', linewidth=lnsz)
    ax.plot(Result['jk_b'][:, gas['kCH4']], BedParams['y'], 'y-', linewidth=lnsz)
    ax.plot(Result['jk_b'][:, gas['kCO']], BedParams['y'], 'k-', linewidth=lnsz)
    ax.plot(Result['jk_b'][:, gas['kCO2']], BedParams['y'], 'g-', linewidth=lnsz)
    ax.plot(Result['jk_b'][:, gas['kH2O']], BedParams['y'], 'b-', linewidth=lnsz)
    ax.axvline(x = 0, color='k', linestyle='--', linewidth=1.0)
    #ax.text(0.5, 0.46, 'Freeboard zone', fontsize = fsz2)
    ax.set_ylabel('Position [m]', fontsize=fsz)
    ax.set_xlabel('Species $\ j_{\mathrm{k,b}}$ [$\mathrm{kg \ m^{-2} \ s^{-1}}$]', fontsize=fsz)
    ax.set_ylim([0, BedParams['y'][-1]])
    ax.legend(['$\mathrm{H_{2}}$', '$\mathrm{CH_{4}}$', '$\mathrm{CO}$', \
               '$\mathrm{CO_{2}}$','$\mathrm{H_{2}O}$'],fontsize=fsz2,loc='best', \
             ncol=2)
    if save == 1:
        plt.savefig(SolFilePath + '/jk_b.jpg',bbox_inches='tight', dpi = 900)
    
    # plot U_hat across reactor
    plt.figure(figsize = (4.75,5))
    ax = plt.gca()
    ax.plot(Result['U_hat'], BedParams['y'], '-', color = 'r',  linewidth=lnsz, label = '$\hat{U}$')
    ax.set_ylabel('Position [m]', fontsize=fsz)
    ax.set_xlabel('$\hat{U}$ [$\mathrm{-}$]',fontsize=fsz)
    ax.set_ylim([0, BedParams['y'][-1]])
    if save == 1:
        plt.savefig(SolFilePath + '/h_Tw.jpg',bbox_inches='tight', dpi = 900)
    
    # plot h_Tw across reactor
    plt.figure(figsize = (4.75,5))
    ax = plt.gca()
    ax.plot(Result['hT_wb'], BedParams['y'], '-', color = 'b',  linewidth=lnsz, label = '${h}_{\mathrm{T,w}}$')
    if BedParams['fin'] == 1:
        ax.plot(Result['hT_wb_no_fin'], BedParams['y'], '--', color = 'b',  linewidth=lnsz, label = '${h}_{\mathrm{T,w,no \ fin}}$')
    ax.set_ylabel('Position [m]', fontsize=fsz)
    ax.set_xlabel('${h}_{\mathrm{T,w}}$ [$\mathrm{W \ m^{-2} \ K^{-1}}$]',fontsize=fsz)
    ax.set_ylim([0, BedParams['y'][-1]])
    ax.legend(loc='best', fontsize=fsz2)
    if save == 1:
        plt.savefig(SolFilePath + '/h_Tw.jpg',bbox_inches='tight', dpi = 900)
    
    """
    # plot cat eff of species
    plt.figure(9, figsize = (4.75,5))
    ax = plt.gca()
    ax.plot(Result['eta_cat_k'][:, gas['kH2']], BedParams['y'], 'r-', linewidth=lnsz)
    ax.plot(Result['eta_cat_k'][:, gas['kCH4']], BedParams['y'], 'y-', linewidth=lnsz)
    ax.plot(Result['eta_cat_k'][:, gas['kCO']], BedParams['y'], 'k-', linewidth=lnsz)
    ax.plot(Result['eta_cat_k'][:, gas['kCO2']], BedParams['y'], 'g-', linewidth=lnsz)
    ax.plot(Result['eta_cat_k'][:, gas['kH2O']], BedParams['y'], 'b-', linewidth=lnsz)
    #ax.axvline(x = 0, color='k', linestyle='--', linewidth=1.0)
    #ax.text(0.5, 0.46, 'Freeboard zone', fontsize = fsz2)
    ax.set_ylabel('Position [m]', fontsize=fsz)
    ax.set_xlabel('Catalyst effectiveness factor of species $\eta_{\mathrm{cat,k}}$ [$\mathrm{-}$]', fontsize=fsz)
    ax.set_ylim([0, BedParams['y'][-1]])
    ax.legend(['$\mathrm{H_{2}}$', '$\mathrm{CH_{4}}$', '$\mathrm{CO}$', \
               '$\mathrm{CO_{2}}$','$\mathrm{H_{2}O}$'],fontsize=fsz2,loc='best', \
             ncol=2)
    if save == 1:
        plt.savefig(SolFilePath + '/eta_cat_b.jpg',bbox_inches='tight', dpi = 900)
    
    
    # plot cat eff of reaction
    plt.figure(10, figsize = (4.75,5))
    ax = plt.gca()
    ax.plot(Result['eta_cat_j'][:, gas['jDMR']], BedParams['y'], '-', color='blueviolet', linewidth=lnsz)
    ax.plot(Result['eta_cat_j'][:, gas['jSMR1']], BedParams['y'], '-', color='cyan', linewidth=lnsz)
    ax.plot(Result['eta_cat_j'][:, gas['jSMR2']], BedParams['y'], '-', color='goldenrod', linewidth=lnsz)
    ax.plot(Result['eta_cat_j'][:, gas['jWGS']], BedParams['y'], '-', color='lightcoral', linewidth=lnsz)
    #ax.axvline(x = 0, color='k', linestyle='--', linewidth=1.0)
    #ax.text(0.5, 0.46, 'Freeboard zone', fontsize = fsz2)
    ax.set_ylabel('Position [m]', fontsize=fsz)
    ax.set_xlabel('Catalyst effectiveness factor of reaction $\eta_{\mathrm{cat,j}}$ [$\mathrm{-}$]', fontsize=fsz)
    ax.set_ylim([0, BedParams['y'][-1]])
    ax.legend(['$\mathrm{DMR}$', '$\mathrm{SMR \ 1}$', '$\mathrm{SMR \ 2}$', \
               '$\mathrm{WGS}$'],fontsize=fsz2,loc='best', \
             ncol=2)
    plt.savefig(SolFilePath + '/eta_cat_b.jpg',bbox_inches='tight', dpi = 900)
    """
    n_fig = 9
    
    """
    Yk_p_plot = np.concatenate((Result['Yk_p'], Result['Xk_p_bound']), axis = 2)
    Xk_p_plot = np.concatenate((Result['Xk_p'], Result['Xk_p_bound']), axis = 2)
    
    plot_0 = 0
    plot_005 = np.ceil(0.05*BedParams['n_y'])
    plot_01 = np.ceil(0.1*BedParams['n_y'])
    plot_02 = np.ceil(0.2*BedParams['n_y'])
    plot_025 = np.ceil(0.25*BedParams['n_y'])
    plot_03 = np.ceil(0.3*BedParams['n_y'])
    plot_04 = np.ceil(0.4*BedParams['n_y'])
    plot_05 = np.ceil(0.5*BedParams['n_y'])
    plot_06 = np.ceil(0.6*BedParams['n_y'])
    plot_07 = np.ceil(0.7*BedParams['n_y'])
    plot_075 = np.ceil(0.75*BedParams['n_y'])
    plot_08 = np.ceil(0.8*BedParams['n_y'])
    plot_09 = np.ceil(0.9*BedParams['n_y'])
    plot_095 = np.ceil(0.95*BedParams['n_y'])
    plot_1 = BedParams['n_y'] - 1
    plot_xkp_iy = np.array([plot_0, plot_025, \
                            plot_05, plot_075, \
                            plot_1], dtype=int)
    
    for i_y in range(len(plot_xkp_iy)):
        plt.figure(n_fig+1, figsize = (4.75,5))
        ax = plt.gca()
        ax.plot(SurfParams['rRmax'], Xk_p_plot[plot_xkp_iy[i_y], gas['kH2'], :], 'r-', linewidth=lnsz)
        ax.plot(SurfParams['rRmax'], Xk_p_plot[plot_xkp_iy[i_y], gas['kCH4'], :], 'y-', linewidth=lnsz)
        ax.plot(SurfParams['rRmax'], Xk_p_plot[plot_xkp_iy[i_y], gas['kCO'], :], 'k-', linewidth=lnsz)
        ax.plot(SurfParams['rRmax'], Xk_p_plot[plot_xkp_iy[i_y], gas['kCO2'], :], 'g-', linewidth=lnsz)
        ax.plot(SurfParams['rRmax'], Xk_p_plot[plot_xkp_iy[i_y], gas['kH2O'], :], 'b-', linewidth=lnsz)
        
        plt.title('Height from bottom of reactor = {:.2f} m'.format(BedParams['y'][plot_xkp_iy[i_y]]))
        ax.set_ylabel('Gas species mole fraction ${X}_{\mathrm{k,p}}$ [$\mathrm{-}$]',fontsize=fsz)
        ax.set_xlabel('$r / R_{\mathrm{max}}$ [$\mathrm{-}$]',fontsize=fsz)
        ax.set_ylim(0, 1)
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
        ax.tick_params(axis = 'x', labelsize=fsz3)
        ax.tick_params(axis = 'y', labelsize=fsz3)
        ax.legend(['$\mathrm{H_{2}}$','$\mathrm{CH_{4}}$','$\mathrm{CO}$','$\mathrm{CO_{2}}$','$\mathrm{H_{2}O}$'],\
                  fontsize=fsz2,loc='best',ncol=2)
        plt.savefig(SolFilePath + '/Xk_p_'+ str(BedParams['y'][plot_xkp_iy[i_y]]) +'.jpg',bbox_inches='tight', dpi = 900)
        
        n_fig += 1
    """
    plt.show()
    
    return Result

#%% Extract and Plot Result for Particle Model only at a single height
def solarReact_ExtractPlotter_Part_y(gas, gas_surf, surf, PartParams, BedParams, SurfParams, ind, Solution, SolFilePath, Result_y, Part_y, i_y, Xk_p_eq):
    
    Result_Part_y = {}
    
    #%% Extract Result
    (Result_Part_y['Yk_p'], Result_Part_y['Xk_p']) = solarReact_Unpack_Part_y(Solution['x'], ind, gas, BedParams, SurfParams)

    Yk_p_bound_sol = np.zeros((1, gas['kspec'], 1))
    Xk_p_bound_sol = np.zeros((1, gas['kspec'], 1))

    for i_spec in range(gas['kspec']):
        Yk_p_bound_sol[0, i_spec, 0] = (Result_Part_y['Yk_p'][0, i_spec, -1] + Result_y['Yk_bg'][0, i_spec])/2
        Xk_p_bound_sol[0, i_spec, 0] = (Result_Part_y['Xk_p'][0, i_spec, -1] + Result_y['Xk_bg'][0, i_spec])/2

    Result_Part_y['Yk_p'] = np.concatenate((Result_Part_y['Yk_p'], Yk_p_bound_sol), axis = 2)
    Result_Part_y['Xk_p'] = np.concatenate((Result_Part_y['Xk_p'], Xk_p_bound_sol), axis = 2)
    Result_Part_y['Xk_p_eq'] = Xk_p_eq
    
    #%% Process equilibrium composition
    rRmax_eq_plot = [SurfParams['rRmax'][0], 1]
    Xk_p_eq_plot = np.array([Xk_p_eq, Xk_p_eq])
    
    #%% Calculate jk_bound
    U_inf = abs(Result_y['v_bg'] - Result_y['v_bs'])
    Result_Part_y['jk_b'], _, _ = ParticleModel_Boundary_flux(gas, PartParams, Result_y['P_p'], \
                                    Result_Part_y['Yk_p'][0, :, SurfParams['n_p'] - 1], \
                                    Result_y['T_p'], Result_y['P_bg'], Result_y['Yk_bg'][0,:], \
                                    Result_y['T_bg'], U_inf, Result_y['phi_bg'])
    
    #%% Catalyst effectiveness factor calculation
    # Initialize species production and consumption rate
    sdot_g = np.zeros((1, gas['kspec'], SurfParams['n_p']))
    sdot_g_bulk = np.zeros((1, gas['kspec']))
    
    # Define reaction rate arrays for catalyst effectiveness factor calculation
    Result_Part_y['eta_cat_k']   = np.zeros((1, gas['kspec']))
    integ_sdot_g_p = np.zeros(gas['kspec'])
   
    if BedParams['kinetics'] == 1:
        R_j_p                       = np.zeros((1, gas['jreac'], SurfParams['n_p']))
        R_j_bulk                    = np.zeros((1, gas['jreac']))
        Result_Part_y['eta_cat_j']  = np.zeros((1, gas['jreac']))
        integ_Rdot_j_p              = np.zeros(gas['jreac'])
    
    # Calculate production rate
    if BedParams['kinetics'] == 1:
        # Reaction rate at bulk condition
        sdot_g_bulk[0, :], R_j_bulk[0, :] = KineticFun(ind, gas, Result_y['Xk_bg'][0, :], Result_y['T_bg'], Result_y['P_bg'], SurfParams)
    elif BedParams['kinetics'] == 2:
        # Gas species production rate from detailed surface chemistry
        gas_surf['obj'].set_unnormalized_mass_fractions(np.concatenate((Result_y['Yk_bg'][0, :], [0])))
        gas_surf['obj'].TP = Result_y['T_bg'], Result_y['P_bg']
            
        # Set the surface species composition and integrate to steady state
        surf['obj'].set_unnormalized_coverages(SurfParams['Zk_p_init'][0, :, 0])
        surf['obj'].TP = Result_y['T_bg'], Result_y['P_bg']
        if SurfParams['integrate'] == 1:
            surf['obj'].advance_coverages(SurfParams['delta_t'], 1e-7, 1e-14, 1e-1, 1e8, 20)
        
        sdot_g_bulk[0, :] = surf['obj'].get_net_production_rates('reformate-part')[:-1]
        
    for i_p in range(SurfParams['n_p']):
        # Set the gas species
        gas['obj'].TPY = Result_y['T_p'], Result_y['P_p'], Result_Part_y['Yk_p'][0, :, i_p]
            
        if BedParams['kinetics'] == 1:
            # Gas species production rate from catalytic reaction from global mechanism
            # and Reaction rate at particle condition
            sdot_g[0, :, i_p], R_j_p[0, :, i_p] = KineticFun(ind, gas, Result_Part_y['Xk_p'][0, :, i_p], Result_y['T_p'], Result_y['P_p'], SurfParams)
            
        elif BedParams['kinetics'] == 2:
            # Gas species production rate from detailed surface chemistry
            # Set the gas and surface phase object and retrieve properties
            gas_surf['obj'].set_unnormalized_mass_fractions(np.concatenate((Result_Part_y['Yk_p'][0, :, i_p], [0])))
            gas_surf['obj'].TP = Result_y['T_p'], Result_y['P_p']
                    
            # Set the surface species composition and integrate to steady state
            surf['obj'].set_unnormalized_coverages(SurfParams['Zk_p_init'][0, :, i_p])
            surf['obj'].TP = Result_y['T_p'], Result_y['P_p']
            if SurfParams['integrate'] == 1:
                surf['obj'].advance_coverages(SurfParams['delta_t'], 1e-7, 1e-14, 1e-1, 1e8, 20)
            
            sdot_g[0, :, i_p] = surf['obj'].get_net_production_rates('reformate-part')[:-1]
        
    # Catalyst effectivenes factor = local reaction rate x volume / bulk reaction rate x volume
    for i_spec in range(gas['kspec']):
        integ_sdot_g_p[i_spec] = 0
        for i_p in range(SurfParams['n_p']):
            integ_sdot_g_p[i_spec] = integ_sdot_g_p[i_spec] + \
                ((sdot_g[0, i_spec, i_p] * SurfParams['rface'][i_p+1]**2 + sdot_g[0, i_spec, i_p] * SurfParams['rface'][i_p]**2)/2) * \
                (SurfParams['rface'][i_p+1] - SurfParams['rface'][i_p])
 
        # Effectiveness factor for each species at different bed locations
        Result_Part_y['eta_cat_k'][0, i_spec] = 3 * (integ_sdot_g_p[i_spec])/(SurfParams['Rmax']**3 * sdot_g_bulk[0, i_spec])
            
    if BedParams['kinetics'] == 1:
        for i_reac in range(gas['jreac']):
            integ_Rdot_j_p[i_reac] = 0
            for i_p in range(SurfParams['n_p']):
                integ_Rdot_j_p[i_reac] = integ_Rdot_j_p[i_reac] + \
                    ((R_j_p[0, i_reac, i_p] * SurfParams['rface'][i_p+1]**2 + R_j_p[0, i_reac, i_p] * SurfParams['rface'][i_p]**2)/2) * \
                    (SurfParams['rface'][i_p+1] - SurfParams['rface'][i_p])
     
            # Effectiveness factor for each species at different bed locations
            Result_Part_y['eta_cat_j'][0, i_reac] = 3 * (integ_Rdot_j_p[i_reac])/(SurfParams['Rmax']**3 * R_j_bulk[0, i_reac])
    
    #%% Plotter
    lnsz = 1.5  # Line width
    lnsz2 = 1   # Line width
    fsz = 15    # Font size
    fsz2 = 10   # legend size
    fsz3 = 13   # axis  label size
    lwd = 2     # Line width for the axes
    
    save = 1
    
    plt.figure(figsize = (4.75,5))
    ax = plt.gca()
    ax.plot(SurfParams['rRmax'], Result_Part_y['Xk_p'][0, gas['kH2'], :], 'r-', linewidth=lnsz)
    ax.plot(SurfParams['rRmax'], Result_Part_y['Xk_p'][0, gas['kCH4'], :], 'y-', linewidth=lnsz)
    ax.plot(SurfParams['rRmax'], Result_Part_y['Xk_p'][0, gas['kCO'], :], 'k-', linewidth=lnsz)
    ax.plot(SurfParams['rRmax'], Result_Part_y['Xk_p'][0, gas['kCO2'], :], 'g-', linewidth=lnsz)
    ax.plot(SurfParams['rRmax'], Result_Part_y['Xk_p'][0, gas['kH2O'], :], 'b-', linewidth=lnsz)
    
    ax.plot(rRmax_eq_plot, Xk_p_eq_plot[:, 0, gas['kH2']], 'r--', linewidth=lnsz2)
    ax.plot(rRmax_eq_plot, Xk_p_eq_plot[:, 0, gas['kCH4']], 'y--', linewidth=lnsz2)
    ax.plot(rRmax_eq_plot, Xk_p_eq_plot[:, 0, gas['kCO']], 'k--', linewidth=lnsz2)
    ax.plot(rRmax_eq_plot, Xk_p_eq_plot[:, 0, gas['kCO2']], 'g--', linewidth=lnsz2)
    ax.plot(rRmax_eq_plot, Xk_p_eq_plot[:, 0, gas['kH2O']], 'b--', linewidth=lnsz2)
    
    plt.title('Height from bottom of reactor = {:.2f} m'.format(BedParams['y'][Part_y[i_y]]))
    ax.set_ylabel('Gas species mole fraction ${X}_{\mathrm{k,p}}$ [$\mathrm{-}$]',fontsize=fsz)
    ax.set_xlabel('$r / R_{\mathrm{max}}$ [$\mathrm{-}$]',fontsize=fsz)
    ax.set_ylim(0, 1)
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1])
    ax.tick_params(axis = 'x', labelsize=fsz3)
    ax.tick_params(axis = 'y', labelsize=fsz3)
    ax.legend(['$\mathrm{H_{2}}$','$\mathrm{CH_{4}}$','$\mathrm{CO}$','$\mathrm{CO_{2}}$','$\mathrm{H_{2}O}$'],\
                  fontsize=fsz2,loc='best',ncol=2)
    if save == 1:
        plt.savefig(SolFilePath + '/Xk_p_'+ f"{BedParams['y'][Part_y[i_y]]:.2f}" +'.jpg',bbox_inches='tight', dpi = 900)
    
    plt.show()
    
    return Result_Part_y['Yk_p'], Result_Part_y['Xk_p'] , Result_Part_y['Xk_p_eq'], Result_Part_y['jk_b'], Result_Part_y['eta_cat_k']

#%% 
def solarReact_Plotter_Part(gas, BedParams, SolFilePath, Result, Result_Part_y, Bed_Part_y):
    
    # Plot comparison of eta_cat and jk_b from simple particle and multi cell particle model
    # plot jk,b in and out of particle
    lnsz = 1.5  # Line width
    fsz = 15    # Font size
    fsz2 = 10    # legend size
    fsz3 = 13   # axis  label size
    lwd = 2     # Line width for the axes
    
    save = 1
    
    plt.figure(figsize = (4.75,5))
    ax = plt.gca()
    ax.plot(Result['jk_b'][:, gas['kH2']], BedParams['y'], 'r-', linewidth=lnsz)
    ax.plot(Result['jk_b'][:, gas['kCH4']], BedParams['y'], 'y-', linewidth=lnsz)
    ax.plot(Result['jk_b'][:, gas['kCO']], BedParams['y'], 'k-', linewidth=lnsz)
    ax.plot(Result['jk_b'][:, gas['kCO2']], BedParams['y'], 'g-', linewidth=lnsz)
    ax.plot(Result['jk_b'][:, gas['kH2O']], BedParams['y'], 'b-', linewidth=lnsz)
    
    ax.plot(Result_Part_y['jk_b'][:, gas['kH2']], Bed_Part_y, 'r--', linewidth=lnsz)
    ax.plot(Result_Part_y['jk_b'][:, gas['kCH4']], Bed_Part_y, 'y--', linewidth=lnsz)
    ax.plot(Result_Part_y['jk_b'][:, gas['kCO']], Bed_Part_y, 'k--', linewidth=lnsz)
    ax.plot(Result_Part_y['jk_b'][:, gas['kCO2']], Bed_Part_y, 'g--', linewidth=lnsz)
    ax.plot(Result_Part_y['jk_b'][:, gas['kH2O']], Bed_Part_y, 'b--', linewidth=lnsz)
    
    ax.axvline(x = 0, color='k', linestyle=':', linewidth=1.0)

    ax.set_ylabel('Position [m]', fontsize=fsz)
    ax.set_xlabel('Species $\ j_{\mathrm{k,b}}$ [$\mathrm{kg \ m^{-2} \ s^{-1}}$]', fontsize=fsz)
    ax.set_ylim([0, BedParams['y'][-1]])
    ax.legend(['$\mathrm{H_{2}}$', '$\mathrm{CH_{4}}$', '$\mathrm{CO}$', \
               '$\mathrm{CO_{2}}$','$\mathrm{H_{2}O}$'],fontsize=fsz2,loc='best', \
             ncol=2)
    if save == 1:
        plt.savefig(SolFilePath + '/jk_b_comparison.jpg',bbox_inches='tight', dpi = 900)
    
    """
    # plot catalyst eff factor across reactor
    plt.figure(figsize = (4.75,5))
    ax = plt.gca()
    ax.plot(Result['eta_cat_k'][:, gas['kH2']], BedParams['y'], 'r-', linewidth=lnsz)
    ax.plot(Result['eta_cat_k'][:, gas['kCH4']], BedParams['y'], 'y-', linewidth=lnsz)
    ax.plot(Result['eta_cat_k'][:, gas['kCO']], BedParams['y'], 'k-', linewidth=lnsz)
    ax.plot(Result['eta_cat_k'][:, gas['kCO2']], BedParams['y'], 'g-', linewidth=lnsz)
    ax.plot(Result['eta_cat_k'][:, gas['kH2O']], BedParams['y'], 'b-', linewidth=lnsz)
    
    ax.plot(Result_Part_y['eta_cat_k'][:, gas['kH2']], Bed_Part_y, 'r--', linewidth=lnsz)
    ax.plot(Result_Part_y['eta_cat_k'][:, gas['kCH4']], Bed_Part_y, 'y--', linewidth=lnsz)
    ax.plot(Result_Part_y['eta_cat_k'][:, gas['kCO']], Bed_Part_y, 'k--', linewidth=lnsz)
    ax.plot(Result_Part_y['eta_cat_k'][:, gas['kCO2']], Bed_Part_y, 'g--', linewidth=lnsz)
    ax.plot(Result_Part_y['eta_cat_k'][:, gas['kH2O']], Bed_Part_y, 'b--', linewidth=lnsz)
    
    ax.set_ylabel('Position [m]', fontsize=fsz)
    ax.set_xlabel('$\eta_{\mathrm{cat,k}}$ [$\mathrm{-}$]',fontsize=fsz)
    ax.set_ylim([0, BedParams['y'][-1]])
    ax.legend(['$\mathrm{H_{2}}$', '$\mathrm{CH_{4}}$', '$\mathrm{CO}$', \
               '$\mathrm{CO_{2}}$','$\mathrm{H_{2}O}$'],fontsize=fsz2,loc='best', \
             ncol=2)
    
    if save == 1:
        plt.savefig(SolFilePath + '/eta_cat_k_comparison.jpg',bbox_inches='tight', dpi = 900)
    """
    plt.show()