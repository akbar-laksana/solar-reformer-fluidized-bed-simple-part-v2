import numpy as np
import sys
import os

functions_folder = os.path.join(".", 'Functions')
sys.path.append(functions_folder)

from solarReact_GasProps import solarReact_GasProps
from solarReact_SolidProps import solarReact_SolidProps

#%% Extract reactor model state variables and scale as needed
def solarReact_Unpack(SV, gas, part, GasParams, PartParams, BedParams, SurfParams, ind, scale):
    
    #%% P_bg from SV or prescribed values
    if BedParams['gas_momentum'] == 1:
        P_bg    = SV[ind['start'] + ind['P_bg']] * scale['var'][ind['start'] + ind['P_bg']]
    else:
        # Estimated linear gas pressure
        P_bg        = np.zeros((BedParams['n_y']))
         
        dP_g        = BedParams['P_drop']*(BedParams['y_b'][-1] - BedParams['y_b'][0]) # Estimated pressure drop of ~15-20 kPa/m
        P_bg_out    = GasParams['P_in'] - dP_g                                      # Estimated outlet pressure of fluidizing gas [Pa]
         
        P_bg[BedParams['index_b']] = np.linspace(GasParams['P_in'], P_bg_out, BedParams['n_y_b'])   # [Pa]
        
    #%% phi_bs from SV
    if BedParams['solid_momentum'] == 1:
        phi_bs  = SV[ind['start'] + ind['phi_bs']] * scale['var'][ind['start'] + ind['phi_bs']]
    
    else:
        # Prescribed solid volume fraction profile with selection of piecewise parabola and linear
        phi_bs = 'piecewise'
        if phi_bs == 'piecewise':
            # Prescribed points
            Mid_fraction = 0.3
            Mid_number = int(np.ceil(BedParams['n_y'] * Mid_fraction))
        
            y_0, phi_bs_0 = BedParams['y_0'], PartParams['phi_bs_max'] - 0.1
            y_mid, phi_bs_mid = BedParams['y'][Mid_number-1], PartParams['phi_bs_max'] - 0.25
            y_top, phi_bs_top = BedParams['y'][-1], PartParams['phi_bs_max'] - 0.25

            # Lower parabola
            a_L = (phi_bs_0 - phi_bs_mid) / ((y_0 - y_mid)**2)

            # Upper parabola
            a_U = (phi_bs_top - phi_bs_mid) / ((y_top - y_mid)**2)

            # Build height arrays
            y_lower = np.linspace(y_0, y_mid, Mid_number)
            y_upper = np.linspace(y_mid, y_top, BedParams['n_y'] - Mid_number)

            # Calculate profiles
            phi_lower = a_L * (y_lower - y_mid)**2 + phi_bs_mid
            phi_upper = a_U * (y_upper - y_mid)**2 + phi_bs_mid
        
            # Concatenate array of solid volume fraction
            phi_bs = np.concatenate((phi_lower, phi_upper))
        
        elif phi_bs == 'linear':
            phi_bs      = np.zeros((BedParams['n_y']))
            phi_bs_0    = PartParams['phi_bs_max'] - 0.1
            phi_bs_top  = PartParams['phi_bs_max'] - 0.25
            
            phi_bs[BedParams['index_b']] = np.linspace(phi_bs_0, phi_bs_top, BedParams['n_y_b'])
    
    # Calculate bed gas volume fraction
    phi_bg = 1 - phi_bs
    
    #%% Estimated gas pressure
    # Get T from SV
    if BedParams['energy'] == 1:
        T_we    = SV[ind['start'] + ind['T_we']] * scale['var'][ind['start'] + ind['T_we']]
        T_wb    = SV[ind['start'] + ind['T_wb']] * scale['var'][ind['start'] + ind['T_wb']]
        T_bs    = SV[ind['start'] + ind['T_bs']] * scale['var'][ind['start'] + ind['T_bs']]
        T_bg    = SV[ind['start'] + ind['T_bg']] * scale['var'][ind['start'] + ind['T_bg']]
    
    #%% Initialize and set particle model variables
    T_p = np.ones((BedParams['n_y'])) * T_bs[:]
    P_p = np.ones((BedParams['n_y'])) * P_bg[:]
    
    #%%
    # Initialize bed gas species mass fractions and mole fractions
    Yk_bg = np.zeros((BedParams['n_y'], gas['kspec']))
    Xk_bg = np.zeros((BedParams['n_y'], gas['kspec']))
    
    # Yk_bg profile from SV
    for i_spec in range(gas['kspec']):
        Yk_bg[:, i_spec] = SV[ind['start'] + ind['Yk_bg'][i_spec]] * scale['var'][ind['start'] + ind['Yk_bg'][i_spec]]
        
    # Calculate mole fractions for each bed position `i_y`
    for i_y in range(BedParams['n_y']):
        gas['obj'].TPY      = T_bg[i_y], P_bg[i_y], Yk_bg[i_y, :]
        Xk_bg[i_y, :]       = gas['obj'].X
    
    if PartParams['simple_part'] == 1:
        # Yk_p profile from SV
        # Initialize Yk_p Xk_p to hold species mass and mole fractions
        Yk_p_int = np.zeros((BedParams['n_y'], gas['kspec']))
        Xk_p_int = np.zeros((BedParams['n_y'], gas['kspec']))
        Yk_p = np.zeros((BedParams['n_y'], gas['kspec']))
        Xk_p = np.zeros((BedParams['n_y'], gas['kspec']))
        
        # Yk_p profile
        for i_spec in range(gas['kspec']):
            if PartParams['interface'] == 1:
                Yk_p_int[:, i_spec] = SV[ind['start'] + ind['Yk_p_int'][i_spec]] * scale['var'][ind['start'] + ind['Yk_p_int'][i_spec]] 
            else:
                Yk_p_int[:, i_spec] = SV[ind['start'] + ind['Yk_bg'][i_spec]] * scale['var'][ind['start'] + ind['Yk_bg'][i_spec]] 
        
            Yk_p[:, i_spec]     = SV[ind['start'] + ind['Yk_p'][i_spec]] * scale['var'][ind['start'] + ind['Yk_p'][i_spec]]  
    
        # Calculate mole fractions for each bed position `i_y`
        for i_y in range(BedParams['n_y']):
            gas['obj'].TPY      = T_p[i_y], P_p[i_y], Yk_p[i_y, :]
            Xk_p[i_y, :]        = gas['obj'].X
            gas['obj'].TPY      = T_p[i_y], P_p[i_y], Yk_p_int[i_y, :]
            Xk_p_int[i_y, :]    = gas['obj'].X
    
    elif PartParams['multi_part'] == 1:
        # Yk_p profile depends on if it's multi- or single- cell particle model
        if SurfParams['n_p'] > 1:
            # Initialize Yk_p Xk_p to hold species mass and mole fractions
            Yk_p = np.zeros((BedParams['n_y'], gas['kspec'], SurfParams['n_p']))
            Xk_p = np.zeros((BedParams['n_y'], gas['kspec'], SurfParams['n_p']))
        
            for i_p in range(SurfParams['n_p']):
                for i_spec in range(gas['kspec']):
                    Yk_p[:, i_spec, i_p] = SV[ind['start'] + int(ind['Yk_p'][(i_spec + gas['kspec'] * i_p)]) ] * \
                        scale['var'][ind['start'] + int(ind['Yk_p'][(i_spec + gas['kspec'] * i_p)]) ]  
    
            # Calculate mole fractions for each bed position `i_y`
            for i_p in range(SurfParams['n_p']):
                for i_y in range(BedParams['n_y']):
                    gas['obj'].TPY      = T_p[i_y], P_p[i_y], Yk_p[i_y, :, i_p]
                    Xk_p[i_y, :, i_p]   = gas['obj'].X
        
        else:
            # Initialize Yk_p Xk_p to hold species mass and mole fractions
            Yk_p = np.zeros((BedParams['n_y'], gas['kspec']))
            Xk_p = np.zeros((BedParams['n_y'], gas['kspec']))
        
            # Yk_p profile
            for i_spec in range(gas['kspec']):
                Yk_p[:, i_spec] = SV[ind['start'] + ind['Yk_p'][i_spec]] * scale['var'][ind['start'] + ind['Yk_p'][i_spec]]  
        
            # Calculate mole fractions for each bed position `i_y`
            for i_y in range(BedParams['n_y']):
                gas['obj'].TPY      = T_p[i_y], P_p[i_y], Yk_p[i_y, :]
                Xk_p[i_y, :]        = gas['obj'].X
    
    #%% Calculate velocities (not part of the solution vector)
    n_y = BedParams['n_y']
    
    # Calculate gas properties at all of vertical nodes
    [rho_g, _, _, _, _, _, _, _] = solarReact_GasProps(gas, n_y, P_bg, T_bg, Yk_bg, T_bs)
    
    # Get particle physical and thermophysical properties
    [rho_s, _, _] = solarReact_SolidProps(part, SurfParams, n_y, P_bg, T_bs)
    
    # Gas velocity
    v_bg = GasParams['mdot_in'] / (rho_g * BedParams['Ay'] * phi_bg)  # [m/s]
    
    # Particle velocity   
    v_bs = PartParams['mdot_in'] / (rho_s * BedParams['Ay'] * phi_bs)  # [m/s]
    
    if PartParams['simple_part'] == 1:  
        return T_we, T_wb, T_bs, T_bg, phi_bs, P_bg, phi_bg, Yk_bg, Xk_bg, T_p, Yk_p, Yk_p_int, P_p, Xk_p, Xk_p_int, v_bg, v_bs
    elif PartParams['multi_part'] == 1:
        return T_we, T_wb, T_bs, T_bg, phi_bs, P_bg, phi_bg, Yk_bg, Xk_bg, T_p, Yk_p, P_p, Xk_p, v_bg, v_bs

#%% Single particle unpack
def solarReact_Unpack_Part_y(SV, ind, gas, BedParams, SurfParams):
    
    n_y = len(ind['start'])
    
    # Initialize Yk_p Xk_p to hold species mass and mole fractions
    Yk_p = np.zeros((n_y, gas['kspec'], SurfParams['n_p']))
    Xk_p = np.zeros((n_y, gas['kspec'], SurfParams['n_p']))
    
    for i_p in range(SurfParams['n_p']):
        for i_spec in range(gas['kspec'] ):
            Yk_p[:, i_spec, i_p] = SV[ind['start'] + int(ind['Yk_p'][i_spec] + (i_p) * (gas['kspec'] ))]
        
        # Calculate mole fractions for each bed position j_y
        for j_y in range(n_y):
            gas['obj'].Y = Yk_p[j_y, :, i_p]
            Xk_p[j_y, :, i_p] = gas['obj'].X
    
    return Yk_p, Xk_p