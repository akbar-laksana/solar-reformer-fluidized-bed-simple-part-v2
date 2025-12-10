import numpy as np
import sys
import os

functions_folder = os.path.join(".", 'Functions')
sys.path.append(functions_folder)

from ParticleModel_Boundary_flux import ParticleModel_Boundary_flux
from ParticleModel_DGM_flux import ParticleModel_DGM_flux
from KineticFun import KineticFun

#%% Residual function for single cell particle model equations (only solving for Yk_p)
def solarReact_PartModEqs_SingleCellPart(gas, surf, GasParams, PartParams, BedParams, SurfParams, ind,  \
                          T_p, Yk_p, P_p, Xk_p, P_bg, T_bg, Yk_bg, v_bg, v_bs, phi_bg, phi_bs):
    
    # Initialize residual and property vectors calculated from Cantera objects
    res = np.zeros(ind['tot'])
    sdot_g = np.zeros((BedParams['n_y'], gas['kspec']))
    
    # Initialize vector of species' mass carried by mass along with its balances
    jk_b = np.zeros((BedParams['n_y'], gas['kspec']))
    
    # Calculate required fluxes and variables needed for particle model residual equations
    for i_y in range(BedParams['n_y']):
        # Calculate the relative velocity of gas and particle
        U_inf = v_bg[i_y] - v_bs[i_y]
        
        if BedParams['kinetics'] == 1:
            # Gas species production rate from catalytic reaction from global mechanism
            sdot_g[i_y, :], _ = KineticFun(ind, gas, Xk_p[i_y, :], T_p[i_y], P_p[i_y], SurfParams)
        
        elif BedParams['kinetics'] == 2:
            # Gas species production rate from detailed surface chemistry
            # Set the gas and surface phase object and retrieve properties
            gas['obj'].set_unnormalized_mass_fractions(Yk_p[i_y, :])
            gas['obj'].TP = T_p[i_y], P_p[i_y]
                
            # Set the surface species composition and integrate to steady state
            surf['obj'].set_unnormalized_coverages(SurfParams['Zk_p_init'][i_y, :])
            surf['obj'].TP = T_p[i_y], P_p[i_y]
            if SurfParams['integrate'] == 1:
                surf['obj'].advance_coverages(SurfParams['delta_t'], 1e-7, 1e-14, 1e-1, 1e8, 20)
        
            sdot_g[i_y, :] = surf['obj'].get_net_production_rates(GasParams['id'])
        
        # Boundary mass fluxes using Sherwood number correlation as outer boundary condition
        jk_b[i_y, :], _, _ = ParticleModel_Boundary_flux(gas, PartParams, P_p[i_y],
                                                         Yk_p[i_y, :], T_p[i_y], P_bg[i_y], 
                                                         Yk_bg[i_y, :], T_bg[i_y], U_inf, phi_bg[i_y])
    
        
    mdotk_pg = np.zeros((BedParams['n_y'], gas['kspec']))
    mdotk_g_gen = np.zeros((BedParams['n_y'], gas['kspec']))
    for i_spec in range(gas['kspec'] ):
        mdotk_pg[:, i_spec] = jk_b[:, i_spec] * phi_bs * BedParams['dVol'] * 6 / PartParams['dp']
        mdotk_g_gen[:, i_spec] = SurfParams['a_cat'] * sdot_g[:, i_spec] * gas['Wk'][i_spec] * (BedParams['dVol'] * phi_bs)
    
    # Set gas mass fractions residuals based on species equation
    for i_spec in range(gas['kspec'] ):
        res[ind['start'] + ind['Yk_p'][i_spec]] = mdotk_g_gen[:, i_spec] - mdotk_pg[:, i_spec]
    
    # Reshape residuals for easier inspection
    res_reshape = res.reshape(BedParams['n_y'], ind['vars'])
    
    return res, jk_b, sdot_g

#%% Residual function for multi cell particle model equations (only solving for Yk_p)
def solarReact_PartModEqs_SimplePart(gas, gas_surf, surf, GasParams, PartParams, BedParams, SurfParams, ind,  \
                          T_p, Yk_p, Yk_p_int, P_p, Xk_p, Xk_p_int, P_bg, T_bg, Yk_bg, v_bg, v_bs, phi_bg, phi_bs):
    
    # Initialize residual and property vectors calculated from Cantera objects
    res = np.zeros(ind['tot'])
    
    # Initialize species production and consumption rate
    sdot_g = np.zeros((BedParams['n_y'], gas['kspec']))
    
    # Initialize vector of species' mass flux
    jk_DGM  = np.zeros((BedParams['n_y'], gas['kspec']))
    jk_b    = np.zeros((BedParams['n_y'], gas['kspec']))

    # Calculate required fluxes and variables needed for particle model residual equations
    for i_y in range(BedParams['n_y']):
        # Calculate the relative velocity of gas and particle
        U_inf = abs(v_bg[i_y] - v_bs[i_y])
        
        # Set the gas species
        gas['obj'].TPY = T_p[i_y], P_p[i_y], Yk_p[i_y, :]
            
        if BedParams['kinetics'] == 1:
            # Gas species production rate from catalytic reaction from global mechanism
            sdot_g[i_y, :], _ = KineticFun(ind, gas, Xk_p[i_y, :], T_p[i_y], P_p[i_y], SurfParams)
            
        elif BedParams['kinetics'] == 2:
            # Gas species production rate from detailed surface chemistry
            # Set the gas and surface phase object and retrieve properties
            gas_surf['obj'].set_unnormalized_mass_fractions(np.concatenate([Yk_p[i_y, :],[0]]))
            gas_surf['obj'].TP = T_p[i_y], P_p[i_y]
                    
            # Set the surface species composition and integrate to steady state
            surf['obj'].set_unnormalized_coverages(SurfParams['Zk_p_init'][i_y, :])
            #surf['obj'].set_unnormalized_coverages(SurfParams['Zk_p_0'][:])
            surf['obj'].TP = T_p[i_y], P_p[i_y]
            if SurfParams['integrate'] == 1:
                surf['obj'].advance_coverages(SurfParams['delta_t'], 1e-7, 1e-14, 1e-1, 1e8, 20)
            
            sdot_g[i_y, :] = surf['obj'].get_net_production_rates('reformate-part')[:-1]
            
        # Calling the Dusty Gas Model for all cells other than the edges to calculate fluxes
        jk_DGM[i_y, :], _ = ParticleModel_DGM_flux(gas, SurfParams, SurfParams['Rmax'],SurfParams['Reff'],\
                                                   P_p[i_y], Xk_p[i_y, :], T_p[i_y], SurfParams['Rmax'],\
                                                   P_p[i_y], Xk_p_int[i_y, :], T_p[i_y])
            
        # Boundary mass fluxes using Sherwood number correlation as outer boundary condition
        jk_b[i_y, :], _, _ = ParticleModel_Boundary_flux(gas, PartParams, P_p[i_y], Yk_p_int[i_y, :], T_p[i_y],\
                                                         P_bg[i_y], Yk_bg[i_y, :], T_bg[i_y], U_inf, phi_bg[i_y])
    
    #sdot_g[0, :] = 0
    
    # Area of particles in each cell [m^2]
    A_sg = phi_bs * BedParams['dVol'] * 6 / PartParams['dp']
           
    # Calculate the particle model residual equations (species' mass balance) and scale
    for i_spec in range(gas['kspec'] ):
        if PartParams['interface'] == 1:
            res[ind['start'] + ind['Yk_p_int'][i_spec]] = jk_b[:, i_spec] * A_sg - \
                    (SurfParams['a_cat'] * (1 - SurfParams['phi']) * sdot_g[:, i_spec] * gas['Wk'][i_spec] * SurfParams['Vpart']) * (BedParams['dVol']*(1-phi_bg)/SurfParams['Vpart'])
            
        else:
            jk_b[:, i_spec] = jk_DGM[:, i_spec]
            
        res[ind['start'] + ind['Yk_p'][i_spec]] = jk_DGM[:, i_spec] * A_sg - \
                (SurfParams['a_cat'] * (1 - SurfParams['phi']) * sdot_g[:, i_spec] * gas['Wk'][i_spec] * SurfParams['Vpart']) * (BedParams['dVol']*(1-phi_bg)/SurfParams['Vpart'])
    
    #%% BC
    #res[ind['start'][0] + ind['Yk_p']]     = (Yk_p[0,:] - Yk_p_int[0,:])    
    
    #%% Reshape residuals for easier inspection
    res_reshape = res.reshape(BedParams['n_y'], ind['vars'])            
    
    return res, jk_b, sdot_g

#%% Residual function for multi cell particle model equations (only solving for Yk_p)
def solarReact_PartModEqs_MultiCellPart(gas, gas_surf, surf, GasParams, PartParams, BedParams, SurfParams, ind,  \
                          T_p, Yk_p, P_p, Xk_p, P_bg, T_bg, Yk_bg, Xk_bg, v_bg, v_bs, phi_bg):
    
    n_y = len(ind['start'])
    
    # Initialize residual and property vectors calculated from Cantera objects
    res = np.zeros((ind['tot']))
    
    # Initialize species production and consumption rate
    sdot_g = np.zeros((n_y, gas['kspec'], SurfParams['n_p']))
    sdot_g_bulk = np.zeros((n_y, gas['kspec']))
    
    # Initialize vector of species' mass carried by mass
    jk = np.zeros((n_y, gas['kspec'], SurfParams['n_p'] + 1))
    jk_b = np.zeros((n_y, gas['kspec']))
    jk_A = np.zeros((n_y, gas['kspec'], SurfParams['n_p'] + 1))
    
    jk_A_in  = np.zeros((n_y, gas['kspec'], SurfParams['n_p']))
    jk_A_out = np.zeros((n_y, gas['kspec'], SurfParams['n_p']))
    
    # Define reaction rate arrays for catalyst effectiveness factor calculation
    eta_cat_k   = np.zeros((n_y, gas['kspec']))
    integ_sdot_g_p = np.zeros(gas['kspec'])
    
    if BedParams['kinetics'] == 1:
        R_j_p       = np.zeros((n_y, gas['jreac'], SurfParams['n_p']))
        R_j_bulk    = np.zeros((n_y, gas['jreac']))
        eta_cat_j   = np.zeros((n_y, gas['jreac']))
        integ_Rdot_j_p = np.zeros(gas['jreac'])
        
    # Calculate required fluxes and variables needed for particle model residual equations
    for i_y in range(n_y):
        # Calculate the relative velocity of gas and particle
        U_inf = abs(v_bg[i_y] - v_bs[i_y])
        
        if BedParams['kinetics'] == 1:
            # Reaction rate at bulk condition
            sdot_g_bulk[i_y, :], _ = KineticFun(ind, gas, Xk_bg[i_y, :], T_bg[i_y], P_bg[i_y], SurfParams)
        elif BedParams['kinetics'] == 2:
            # Gas species production rate from detailed surface chemistry
            gas_surf['obj'].set_unnormalized_mass_fractions(np.concatenate((Yk_p[i_y, :, 0], [0])))
            gas_surf['obj'].TP = T_bg[i_y], P_bg[i_y]
            
            # Set the surface species composition and integrate to steady state
            surf['obj'].set_unnormalized_coverages(SurfParams['Zk_p_init'][i_y, :, 0])
            surf['obj'].TP = T_bg[i_y], P_bg[i_y]
            #if SurfParams['integrate'] == 1:
                #surf['obj'].advance_coverages(SurfParams['delta_t'], 1e-6, 1e-12, 1e-4, 1e8, 20)
        
            sdot_g_bulk[i_y, :] = surf['obj'].get_net_production_rates('reformate-part')[:-1]
            
        for i_p in range(SurfParams['n_p']):
            # Set the gas species
            gas['obj'].TPY = T_p[i_y], P_p[i_y], Yk_p[i_y, :, i_p]
            
            if BedParams['kinetics'] == 1:
                # Gas species production rate from catalytic reaction from global mechanism
                # and Reaction rate at particle condition
                sdot_g[i_y, :, i_p], R_j_p[i_y, :, i_p] = KineticFun(ind, gas, Xk_p[i_y, :, i_p], T_p[i_y], P_p[i_y], SurfParams)
            
            elif BedParams['kinetics'] == 2:
                # Gas species production rate from detailed surface chemistry
                # Set the gas and surface phase object and retrieve properties
                gas_surf['obj'].set_unnormalized_mass_fractions(np.concatenate((Yk_p[i_y, :, i_p], [0])))
                gas_surf['obj'].TP = T_p[i_y], P_p[i_y]
                    
                # Set the surface species composition and integrate to steady state
                surf['obj'].set_unnormalized_coverages(SurfParams['Zk_p_init'][i_y, :, i_p])
                surf['obj'].TP = T_p[i_y], P_p[i_y]
                if SurfParams['integrate'] == 1:
                    surf['obj'].advance_coverages(SurfParams['delta_t'], 1e-7, 1e-14, 1e-3, 1e8, 20)
            
                sdot_g[i_y, :, i_p] = surf['obj'].get_net_production_rates('reformate-part')[:-1]
            
            # Calling the Dusty Gas Model for all cells other than the edges to calculate fluxes
            if 0 < i_p < SurfParams['n_p']:
                jk_DGM, _ = ParticleModel_DGM_flux(gas, SurfParams, SurfParams['rface'][i_p],
                                               SurfParams['rcell'][i_p-1], P_p[i_y], Xk_p[i_y, :, i_p-1], 
                                               T_p[i_y], SurfParams['rcell'][i_p], 
                                               P_p[i_y], Xk_p[i_y, :, i_p], T_p[i_y])
                
            # Flux for i_p = 0 (inner most) is zero, and set temporary for outer boundary to be zero
            else:
                jk_DGM = 0
            
            # Construct jk and jk*A vector
            jk[i_y, :, i_p] = jk_DGM
            jk_A[i_y,:, i_p] = jk[i_y, :, i_p] * SurfParams['Aface'][i_p]
            
        # Boundary mass fluxes using Sherwood number correlation as outer boundary condition
        jk_b[i_y, :], _, _ = ParticleModel_Boundary_flux(gas, PartParams, P_p[i_y],
                                                         Yk_p[i_y, :, -1], 
                                                         T_p[i_y], P_bg[i_y], 
                                                         Yk_bg[i_y, :], T_bg[i_y], U_inf, phi_bg[i_y])
        # Apply the BC
        jk[i_y, :, -1] = jk_b[i_y, :]
        jk_A[i_y, :, -1] = jk[i_y, :, -1] * SurfParams['Aface'][-1]
        
        # Construct the mass flow going in and out of particle cell boundaries at specific y height
        for i_spec in range(gas['kspec']):
            jk_A_in[i_y, i_spec, :]     = jk_A[i_y, i_spec, :-1]
            jk_A_out[i_y, i_spec, :]    = jk_A[i_y, i_spec, 1:]
        
    # Calculate the particle model residual equations (species' mass balance) and scale
    for i_p in range(SurfParams['n_p']):
        for i_spec in range(gas['kspec'] ):
            if SurfParams['chem'] == 1:
                res[ind['start'] + int(ind['Yk_p'][i_spec + (i_p) * (gas['kspec'] )])] \
                    = (jk_A_in[:, i_spec, i_p] - jk_A_out[:, i_spec, i_p] + \
                        (SurfParams['a_cat'] * (1 - SurfParams['phi']) * sdot_g[:, i_spec, i_p] * gas['Wk'][i_spec] * SurfParams['Vcell'][i_p])) \
                        / (SurfParams['Vcell'][i_p])
                
            elif SurfParams['chem'] == 2:
                res[ind['start'] + int(ind['Yk_p'][i_spec + (i_p) * (gas['kspec'] )])] \
                    = (jk_A_in[:, i_spec, i_p] - jk_A_out[:, i_spec, i_p]) / (SurfParams['Vcell'][i_p])
    
    # Catalyst effectivenes factor = local reaction rate x volume / bulk reaction rate x volume
    for i_y in range(n_y):
        for i_spec in range(gas['kspec']):
            integ_sdot_g_p[i_spec] = 0
            for i_p in range(SurfParams['n_p']):
                integ_sdot_g_p[i_spec] = integ_sdot_g_p[i_spec] + \
                    ((sdot_g[i_y, i_spec, i_p] * SurfParams['rface'][i_p+1]**2 + sdot_g[i_y, i_spec, i_p] * SurfParams['rface'][i_p]**2)/2) * \
                    (SurfParams['rface'][i_p+1] - SurfParams['rface'][i_p])
 
            # Effectiveness factor for each species at different bed locations
            eta_cat_k[i_y, i_spec] = 3 * (integ_sdot_g_p[i_spec])/(SurfParams['Rmax']**3 * sdot_g_bulk[i_y, i_spec])
            
        if BedParams['kinetics'] == 1:
            for i_reac in range(gas['jreac']):
                integ_Rdot_j_p[i_reac] = 0
                for i_p in range(SurfParams['n_p']):
                    integ_Rdot_j_p[i_reac] = integ_Rdot_j_p[i_reac] + \
                        ((R_j_p[i_y, i_reac, i_p] * SurfParams['rface'][i_p+1]**2 + R_j_p[i_y, i_reac, i_p] * SurfParams['rface'][i_p]**2)/2) * \
                        (SurfParams['rface'][i_p+1] - SurfParams['rface'][i_p])
     
                # Effectiveness factor for each species at different bed locations
                eta_cat_j[i_y, i_reac] = 3 * (integ_Rdot_j_p[i_reac])/(SurfParams['Rmax']**3 * R_j_bulk[i_y, i_reac])
         
    #%% Reshape residuals for easier inspection
    res_reshape = res.reshape(n_y, ind['vars'])            
    
    return res, jk_b, sdot_g, sdot_g_bulk, eta_cat_k

#%% Residual function for multi cell particle model equations (only solving for Yk_p)
def solarReact_PartModEqs_MultiCellPart_Zeta(gas, gas_surf, surf, GasParams, PartParams, BedParams, SurfParams, ind,  \
                          T_p, Yk_p, P_p, Xk_p, P_bg, T_bg, Yk_bg, Xk_bg, v_bg, v_bs, phi_bg, Zk_p):
    
    n_y = len(ind['start'])
    
    # Initialize residual and property vectors calculated from Cantera objects
    res = np.zeros((ind['tot']))
    
    # Initialize species production and consumption rate
    sdot_g = np.zeros((n_y, gas['kspec'], SurfParams['n_p']))
    sdot_s = np.zeros((n_y, surf['kspec'], SurfParams['n_p']))
    sdot_g_bulk = np.zeros((n_y, gas['kspec']))
    
    # Initialize vector of species' mass carried by mass
    jk = np.zeros((n_y, gas['kspec'], SurfParams['n_p'] + 1))
    jk_b = np.zeros((n_y, gas['kspec']))
    jk_A = np.zeros((n_y, gas['kspec'], SurfParams['n_p'] + 1))
    
    jk_A_in  = np.zeros((n_y, gas['kspec'], SurfParams['n_p']))
    jk_A_out = np.zeros((n_y, gas['kspec'], SurfParams['n_p']))
    
    # Define reaction rate arrays for catalyst effectiveness factor calculation
    eta_cat_k   = np.zeros((n_y, gas['kspec']))
    integ_sdot_g_p = np.zeros(gas['kspec'])
    
    if BedParams['kinetics'] == 1:
        R_j_p       = np.zeros((n_y, gas['jreac'], SurfParams['n_p']))
        R_j_bulk    = np.zeros((n_y, gas['jreac']))
        eta_cat_j   = np.zeros((n_y, gas['jreac']))
        integ_Rdot_j_p = np.zeros(gas['jreac'])
        
    # Calculate required fluxes and variables needed for particle model residual equations
    for i_y in range(n_y):
        # Calculate the relative velocity of gas and particle
        U_inf = abs(v_bg[i_y] - v_bs[i_y])
        
        if BedParams['kinetics'] == 1:
            # Reaction rate at bulk condition
            sdot_g_bulk[i_y, :], R_j_bulk[i_y, :] = KineticFun(ind, gas, Xk_bg[i_y, :], T_bg[i_y], P_bg[i_y], SurfParams)
        elif BedParams['kinetics'] == 2:
            # Gas species production rate from detailed surface chemistry
            gas_surf['obj'].set_unnormalized_mass_fractions(np.concatenate((Yk_bg[i_y, :], [0])))
            gas_surf['obj'].TP = T_bg[i_y], P_bg[i_y]
            
            # Set the surface species composition and integrate to steady state
            surf['obj'].set_unnormalized_coverages(Zk_p[i_y, :, -1])
            surf['obj'].TP = T_bg[i_y], P_bg[i_y]
        
            sdot_g_bulk[i_y, :] = surf['obj'].get_net_production_rates('reformate-part')[:-1]
            
        for i_p in range(SurfParams['n_p']):
            # Set the gas species
            gas['obj'].TPY = T_p[i_y], P_p[i_y], Yk_p[i_y, :, i_p]
            
            if BedParams['kinetics'] == 1:
                # Gas species production rate from catalytic reaction from global mechanism
                # and Reaction rate at particle condition
                sdot_g[i_y, :, i_p], R_j_p[i_y, :, i_p] = KineticFun(ind, gas, Xk_p[i_y, :, i_p], T_p[i_y], P_p[i_y], SurfParams)
            
            elif BedParams['kinetics'] == 2:
                # Gas species production rate from detailed surface chemistry
                # Set the gas and surface phase object and retrieve properties
                gas_surf['obj'].set_unnormalized_mass_fractions(np.concatenate((Yk_p[i_y, :, i_p], [0])))
                gas_surf['obj'].TP = T_p[i_y], P_p[i_y]
                    
                # Set the surface species composition and integrate to steady state
                surf['obj'].coverages = Zk_p[i_y, :, i_p]
                surf['obj'].TP = T_p[i_y], P_p[i_y]

                sdot_s[i_y, :, i_p] = surf['obj'].get_net_production_rates('surf')
                sdot_g[i_y, :, i_p] = surf['obj'].get_net_production_rates('reformate-part')[:-1]
            
            # Calling the Dusty Gas Model for all cells other than the edges to calculate fluxes
            if 0 < i_p < SurfParams['n_p']:
                jk_DGM, _ = ParticleModel_DGM_flux(gas, SurfParams, SurfParams['rface'][i_p],
                                               SurfParams['rcell'][i_p-1], P_p[i_y], Xk_p[i_y, :, i_p-1], 
                                               T_p[i_y], SurfParams['rcell'][i_p], 
                                               P_p[i_y], Xk_p[i_y, :, i_p], T_p[i_y])
                
            # Flux for i_p = 0 (inner most) is zero, and set temporary for outer boundary to be zero
            else:
                jk_DGM = 0
            
            # Construct jk and jk*A vector
            jk[i_y, :, i_p] = jk_DGM
            jk_A[i_y,:, i_p] = jk[i_y, :, i_p] * SurfParams['Aface'][i_p]
            
        # Boundary mass fluxes using Sherwood number correlation as outer boundary condition
        jk_b[i_y, :], _, _ = ParticleModel_Boundary_flux(gas, PartParams, P_p[i_y],
                                                         Yk_p[i_y, :, -1], 
                                                         T_p[i_y], P_bg[i_y], 
                                                         Yk_bg[i_y, :], T_bg[i_y], U_inf, phi_bg[i_y])
        # Apply the BC
        jk[i_y, :, -1] = jk_b[i_y, :]
        jk_A[i_y, :, -1] = jk[i_y, :, -1] * SurfParams['Aface'][-1]
        
        # Construct the mass flow going in and out of particle cell boundaries at specific y height
        for i_spec in range(gas['kspec']):
            jk_A_in[i_y, i_spec, :]     = jk_A[i_y, i_spec, :-1]
            jk_A_out[i_y, i_spec, :]    = jk_A[i_y, i_spec, 1:]
        
    # Calculate the particle model residual equations (species' mass balance) and scale
    for i_p in range(SurfParams['n_p']):
        for i_spec in range(gas['kspec'] ):
            if SurfParams['chem'] == 1:
                res[ind['start'] + int(ind['Yk_p'][i_spec + (i_p) * (gas['kspec'] )])] \
                    = (jk_A_in[:, i_spec, i_p] - jk_A_out[:, i_spec, i_p] + \
                        (SurfParams['a_cat'] * (1 - SurfParams['phi']) * sdot_g[:, i_spec, i_p] * gas['Wk'][i_spec] * SurfParams['Vcell'][i_p])) \
                        / (SurfParams['Vcell'][i_p])
                
            elif SurfParams['chem'] == 2:
                res[ind['start'] + int(ind['Yk_p'][i_spec + (i_p) * (gas['kspec'] )])] \
                    = (jk_A_in[:, i_spec, i_p] - jk_A_out[:, i_spec, i_p]) / (SurfParams['Vcell'][i_p])
        
        for i_spec in range(surf['kspec'] ):
            res[ind['start'] + ind['Zk_p'][i_spec] + int((i_p) * (ind['vars'] / SurfParams['n_p']))] = sdot_s[:, i_spec, i_p] #* 1e5 #/ SurfParams['sitDens'] # (SurfParams['a_cat'] * SurfParams['phi'] * surf['Wk'][i_spec])
            
    # Catalyst effectivenes factor = local reaction rate x volume / bulk reaction rate x volume
    for i_y in range(n_y):
        for i_spec in range(gas['kspec']):
            integ_sdot_g_p[i_spec] = 0
            for i_p in range(SurfParams['n_p']):
                integ_sdot_g_p[i_spec] = integ_sdot_g_p[i_spec] + \
                    ((sdot_g[i_y, i_spec, i_p] * SurfParams['rface'][i_p+1]**2 + sdot_g[i_y, i_spec, i_p] * SurfParams['rface'][i_p]**2)/2) * \
                    (SurfParams['rface'][i_p+1] - SurfParams['rface'][i_p])
 
            # Effectiveness factor for each species at different bed locations
            eta_cat_k[i_y, i_spec] = 3 * (integ_sdot_g_p[i_spec])/(SurfParams['Rmax']**3 * sdot_g_bulk[i_y, i_spec])
            
        if BedParams['kinetics'] == 1:
            for i_reac in range(gas['jreac']):
                integ_Rdot_j_p[i_reac] = 0
                for i_p in range(SurfParams['n_p']):
                    integ_Rdot_j_p[i_reac] = integ_Rdot_j_p[i_reac] + \
                        ((R_j_p[i_y, i_reac, i_p] * SurfParams['rface'][i_p+1]**2 + R_j_p[i_y, i_reac, i_p] * SurfParams['rface'][i_p]**2)/2) * \
                        (SurfParams['rface'][i_p+1] - SurfParams['rface'][i_p])
     
                # Effectiveness factor for each species at different bed locations
                eta_cat_j[i_y, i_reac] = 3 * (integ_Rdot_j_p[i_reac])/(SurfParams['Rmax']**3 * R_j_bulk[i_y, i_reac])
         
    #%% Reshape residuals for easier inspection
    res_reshape = res.reshape(n_y, ind['vars'])            
    
    return res, jk_b, sdot_g, sdot_g_bulk, eta_cat_k