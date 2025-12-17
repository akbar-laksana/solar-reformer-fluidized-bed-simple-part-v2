import numpy as np
import sys
import os

from scipy.interpolate import interp1d

functions_folder = os.path.join(".", 'Functions')
sys.path.append(functions_folder)

from Thermal_Cond import Thermal_Cond
from FB_Cond import FB_Cond
from D_T import D_T
from hT_wb_conv_v2 import hT_wb_conv_v2
from solarReact_GasProps import solarReact_GasProps
from solarReact_SolidProps import solarReact_SolidProps
from DragModel import DragModel
from FinCalcs import FinCalcs

def solarReact_FluidizedBedEqs(ind, gas, part, wall, surf, WallParams, FinParams, BedParams, \
                               PartParams, GasParams, SurfParams, EnvParams, T_we, T_wb, T_bs, T_bg, phi_bs, \
                               phi_bg, P_bg, Yk_bg, Xk_bg, T_p, Yk_p, P_p, Xk_p, v_bg, v_bs, jk_b, sdot_g):
    
    #%% Initialize residual and property vectors calculated from Cantera objects
    res             = np.zeros(ind['tot'])      # initialized residual
    n_y             = BedParams['n_y']          # number of mesh points in the vertical direction of the channel flow 
    
    # Calculate gas properties at all of vertical nodes
    [rho_g, h_g, mu_g, k_g, cp_g, Dk_mix, hk_g, hk_sg] = solarReact_GasProps(gas, n_y, P_bg, T_bg, Yk_bg, T_bs)
    
    # Get particle physical and thermophysical properties
    [rho_s, h_s, cp_s] = solarReact_SolidProps(part, SurfParams, n_y, P_bg, T_bs)
    
    # Define constant and additional properties
    g               = 9.81                          # gravitational constant [m/s^2]
    dp              = PartParams['dp']              # mean sauter diameter of particles [m]
    
    phi_bg_avg      = np.concatenate(([phi_bg[0]], 0.5*(phi_bg[:-1]+phi_bg[1:]), [phi_bg[-1]])) # gas volume fraction at cell boundaries [--]
    
    #%% Calculate minimum fluidization, superficial, and dimensionless excess gas velocities 
    a   = 1.75*PartParams['phi_bs_max']*rho_g/(dp*PartParams['phi_bg_min']**3)      # 2nd degree polynomial term for quadratic solution
    b   = 150*PartParams['phi_bs_max']**2*mu_g/(dp**2*PartParams['phi_bg_min']**3)  # 1st degree polynomial term for quadratic solution
    c   = -(PartParams['phi_bs_max']*rho_s - PartParams['phi_bg_min']*rho_g)*g      # 0th degree polynomial term for quadratic solution
    
    U_mf    = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)    # minimum superficial fluidization gas velocity [m/s]
    U_g     = v_bg * phi_bg                         # superficial gas velocity [m/s]
    U_s     = v_bs * phi_bs                         # superficial particle velocity [m/s]
    
    #%% Calculate dimensionless parameters needed for convective bed-wall heat transfer coefficient
    U_hat   = (rho_s*cp_s/(g*k_g))**(1/3)*(U_g - U_mf)                      # dimensionless excess fluidization velocity [--]
    Ar_lam  = np.sqrt(dp**3*g)*(rho_s - rho_g)/mu_g                         # laminar Archimedes number [--]
    Pr_gs   = k_g / ( 2*cp_s * mu_g )                                       # two-phase fluidized bed Prandtl number [--]
    
    # Call function to get convective bed-wall heat transfer coefficient
    hT_wb_c = hT_wb_conv_v2(U_hat, Ar_lam, Pr_gs, dp, k_g)
    
    # Calculate radiative bed-wall heat transfer coefficient assuming two opaque gray surfaces
    emis_hx         = 1/(1/WallParams['emis'] + 1/PartParams['emis'] - 1)       # [-], effective emissivity for radiative transfer between particles and wall
    hT_wb_rad       = 5.67e-8*emis_hx*(T_bs**2 + T_wb**2)*(T_bs + T_wb)         # [W/m^2-K], radiative particle-wall heat transfer coefficient
    hT_wb           = hT_wb_c + hT_wb_rad                                       # [W/m^2-K], total particle-wall heat transfer coefficient
    
    #%% Fin factor calculation
    eta     = np.zeros(BedParams['n_y'])
    eta_o   = np.zeros(BedParams['n_y'])
    F_fin   = np.zeros(BedParams['n_y'])
    hT_wb_F = np.zeros(BedParams['n_y'])
    
    # Calculate heat transfer enhancement if the bed wall has fins
    if BedParams['fin'] == 1:
        for i_y in range(BedParams['n_y']):    
            [eta[i_y], eta_o[i_y], F_fin[i_y]] = FinCalcs(T_wb[i_y] , hT_wb_c[i_y], FinParams)
            
        # Enhanced wall-bed heat transfer coefficient with fin
        hT_wb_F = (F_fin * hT_wb_c) + hT_wb_rad
        
        if BedParams['geo'] == 1:
            q_wb            = hT_wb_F * BedParams['Az_out'] * (T_wb - T_bs)                 # [W], particle-wall heat transfer for each cell
        elif BedParams['geo'] == 2:    
            q_wb            = hT_wb_F * BedParams['Ar_out'] * (T_wb - T_bs)           
    
    # Finless wall
    else:
        if BedParams['geo'] == 1:
            q_wb            = hT_wb * BedParams['Az_out'] * (T_wb - T_bs)                 # [W], particle-wall heat transfer for each cell
        elif BedParams['geo'] == 2:    
            q_wb            = hT_wb * BedParams['Ar_out'] * (T_wb - T_bs)                 # [W], particle-wall heat transfer for each cell
    
    # Calculate particle-gas interfacial heat transfer coefficient (Gunn 1978)
    Re_p        = dp*abs(v_bg - v_bs)*rho_g/mu_g                                # Reynolds number of gas around particles [--]
    Pr_p        = mu_g * cp_g / k_g
    
    # Nusselt number correlation
    if BedParams['Nusselt'] == 'Gunn':
        Nu_p = (7 - 10*phi_bg + 5*phi_bg**2)*(1 + 0.7*Re_p**0.2 * Pr_p**(1/3)) \
            + (1.33 - 2.4*phi_bg + 1.2*phi_bg**2)*(Re_p**0.7 * Pr_p**(1/3))
    elif BedParams['Nusselt'] == 'other':     
        Nu_p  = 2 + (0.4*Re_p**0.5 + 0.06*Re_p**0.667)
    elif BedParams['Nusselt'] == 'Zhang':
        Nu_p = 2 + 0.6 * Re_p**0.5 * Pr_p**(1/3)
    
    h_part_g    = Nu_p*k_g/dp                                                   # Particle-gas heat transfer coefficient[W/m^2-K]
    
    # Area of particles in each cell [m^2]
    A_sg        = phi_bs * BedParams['dVol'] * 6 / dp                   
    
    #%% Calculate terminal velocity for the beta_drag parameter ( as described in MFIX documentation - Syamlal and O?Brien) %%
    beta_drag   = DragModel(PartParams, BedParams, phi_bs, phi_bg, rho_g, mu_g, v_bs, v_bg, U_hat, U_mf)
    
    #%% Set solids pressure term (elastic modulus), see Bouillard et al. 1989
    Gphi        = np.exp(- PartParams['c']*(phi_bg - PartParams['phi_bg_min']))*PartParams['Go']    # [-]
    
    #%% Set particle-wall friction loss term from Konno, H., Saito, S., 1969. J. Chem. Eng. of Japan 211
    wb_drag     = 2 * 0.0025 * phi_bs * rho_s * abs(v_bs) / (BedParams['D_hyd'])   # [kg/m^3-s]
    
    #%% Calculate gas and particle mass flow in and out of cell vertical boundaries [kg s^-1] 
    mdot_g_out      = phi_bg*rho_g*(v_bg*BedParams['Ay'])                       # [kg s^-1]
    mdot_g_in       = np.concatenate(([GasParams['mdot_in']], mdot_g_out[0:-1]))# [kg s^-1]
    
    mdot_s_in       = phi_bs*rho_s*v_bs*BedParams['Ay'] 
    mdot_s_out      = np.concatenate((mdot_s_in[1:BedParams['n_y']], [PartParams['mdot_in']]))  
    
    #%% Calculate gas species mass flow in and out of cell vertical boundaries [kg of k s^-1]                                      # [kg of k s^-1] 
    mdotk_g_out     = Yk_bg * mdot_g_out [:, np.newaxis]
    mdotk_g_in      = np.concatenate(([GasParams['Y_in']*GasParams['mdot_in']], mdotk_g_out[0:-1, :])) # [kg of k s^-1]                                      
    
    #%% Calculate gas and particle momentum fluxes in and out of cell vertical boundaries [kg m s^-2]
    mvdot_g_out     = mdot_g_out * abs(v_bg)  
    mvdot_g_in      = np.concatenate(([GasParams['mvdot_in']], mvdot_g_out[0:-1]))

    mvdot_s_in      = mdot_s_in * abs(v_bs)   
    mvdot_s_out     = np.concatenate((mvdot_s_in[1:], [PartParams['mvdot_in']])) 
    
    #%% Calculate gas and particle enthalpy fluxes in and out of cell vertical boundaries (in W)
    mhdot_g_out     = mdot_g_out * h_g
    mhdot_g_in      = np.concatenate(([GasParams['mhdot_in']], mhdot_g_out[0:-1]))
    
    mhdot_s_in      = mdot_s_in * h_s
    mhdot_s_out     = np.concatenate(( mhdot_s_in[1:], [PartParams['mhdot_in']]))
    
    #%% Estimate effective bed conduction
    T_bg_vec     =  np.concatenate(([T_bg[0]], T_bg, [T_bg[-1]]))
    T_bs_vec     =  np.concatenate(([T_bs[0]], T_bs, [T_bs[-1]]))
    
    k_s  = (1 - SurfParams['phi']) * Thermal_Cond(PartParams['id'], T_bs) # [W/m-K], Thermal conductivity of solid
    
    k_s_interp  = interp1d(BedParams['y'], k_s, kind='linear', fill_value="extrapolate")
    k_s_bnd     = np.concatenate((k_s_interp(BedParams['y_bnd'][:-1]), [k_s[-1]]))
    
    k_g_interp  = interp1d(BedParams['y'], k_g, kind='linear', fill_value="extrapolate")
    k_g_bnd     = np.concatenate((k_g_interp(BedParams['y_bnd'][:-1]), [k_g[-1]]))
    
    phi_g_interp    = interp1d(BedParams['y'], phi_bg, kind='linear', fill_value="extrapolate")
    phi_bg_bnd      = np.concatenate((phi_g_interp(BedParams['y_bnd'][:-1]), [phi_bg[-1]]))
    
    k_s_eff_bnd     = FB_Cond(phi_bg_bnd, k_s_bnd/k_g_bnd)*k_g_bnd # [W/m-K], effective thermal conductivity of bed at cell boundaries
    
    #%% Calculate vertical dispersion coefficient and add it to conduction.
    
    # Set D_hyd to minimum of measured values and actual D_hyd
    #D_hyd = np.minimum(0.02142, BedParams['D_hyd'])
    D_yy_s, _    = D_T(U_g, U_mf, BedParams['D_hyd'], PartParams['id'])         # assuming s,g,h,mv all have same coefficient1
    D_yy_g       = BedParams['D_hyd'] * (U_g-U_mf) / 150                               # from Breault (2006) Table 3 Eqn from Werther et al
    
    # Add in dispersion to thermal conductivity
    k_s_disp_interp = interp1d(BedParams['y'], D_yy_s * rho_s * cp_s * phi_bs, kind='linear', fill_value="extrapolate")
    k_s_disp_bnd    = np.concatenate(( k_s_disp_interp(BedParams['y_bnd'][:-1]), [(D_yy_s[-1] * rho_s[-1] * cp_s[-1] * phi_bs[-1])] ))
    
    if BedParams['thermal_dispersion'] == 1:
        k_comb          = k_s_eff_bnd + k_s_disp_bnd  # Combined solid conduction and thermal energy dispersion [W/m*K]
    elif BedParams['thermal_dispersion'] == 2:
        k_comb          = k_s_eff_bnd
    
    q_cond_y_svec   = BedParams['Ay_cond'] * k_comb * (T_bs_vec[:-1] - T_bs_vec[1:]) / BedParams['dy_cond']
    q_cond_y_s      = (q_cond_y_svec[:-1] - q_cond_y_svec[1:])
    
    q_cond_y_gvec   = BedParams['Ay_cond'] * k_g_bnd *(T_bg_vec[:-1] - T_bg_vec[1:]) / BedParams['dy_cond']
    q_cond_y_g      = (q_cond_y_gvec[:-1] - q_cond_y_gvec[1:])
    
    #%% Dispersion terms

    # Find gas dispersion flux
    j_dis_y_g_out = np.zeros((n_y, gas['kspec']))
    for i_spec in range(gas['kspec']):
        k_m_g_disp_interp   = interp1d(BedParams['y'], phi_bg * rho_g * (Dk_mix[:, i_spec] + D_yy_g[:]), kind='linear', fill_value="extrapolate")
        k_m_g_disp_bnd      = np.concatenate(( k_m_g_disp_interp(BedParams['y_bnd'][:-1]), [phi_bg[-1] * rho_g[-1] * (Dk_mix[-1, i_spec] + D_yy_g[-1])] ))
        j_dis_y_g_out[:, i_spec] = BedParams['Ay_cond'][1:] * k_m_g_disp_bnd[1:] * (Yk_bg[:,i_spec] - np.concatenate(( Yk_bg[1:, i_spec], [Yk_bg[-1,i_spec]] )) )  / BedParams['dy_cond'][1:]
    
    j_dis_y_g_in   = np.concatenate(( np.zeros((1, gas['kspec'])), j_dis_y_g_out[:-1, :])) 
    
    #%% Process the species particle flux values from the particle model
    mdotk_pg = np.zeros((BedParams['n_y'], gas['kspec']))
    mdotk_g_gen = np.zeros((BedParams['n_y'], gas['kspec']))
   
    for i_spec in range(gas['kspec']):
        mdotk_pg[:, i_spec] = jk_b[:, i_spec] * A_sg  # [kg of species k / s]
        
        if PartParams['multi_part'] == 1:
            for i_p in range(SurfParams['n_p']):
                mdotk_g_gen[:, i_spec] = mdotk_g_gen[:, i_spec] + (SurfParams['a_cat'] * (1 - SurfParams['phi']) * sdot_g[:, i_spec, i_p] * gas['Wk'][i_spec] * SurfParams['Vcell'][i_p]) * (BedParams['dVol']*(1-phi_bg)/SurfParams['Vpart'])
        
        if PartParams['simple_part'] == 1:
            mdotk_g_gen[:, i_spec] = mdotk_g_gen[:, i_spec] + (SurfParams['a_cat'] * (1 - SurfParams['phi']) * sdot_g[:, i_spec] * gas['Wk'][i_spec] * SurfParams['Vpart']) * (BedParams['dVol']*(1-phi_bg)/SurfParams['Vpart'])
        
    #Set mdotk_pg at first cell to be 0
    #mdotk_pg[0,:] = 0
    #mdotk_g_gen[0,:] = 0
    
    mdotk_pg[0,:] = mdotk_pg[1,:]
    
    # Gas momentum generation
    mvdot_gen = np.zeros((BedParams['n_y']))  
    mvdot_gen = np.sum(mdotk_pg, axis = 1) * v_bg
    
    # Gas enthalpy generation/destruction, and carried by boundary flux
    mhdot_gen   = np.zeros((BedParams['n_y']))
    jk_b_hk_b   = np.zeros((BedParams['n_y'], gas['kspec']))
    
    for i_y in range(BedParams['n_y']):
        for i_spec in range(gas['kspec']):
            jk_b_hk_b[i_y,i_spec] = mdotk_pg[i_y,i_spec] * ( hk_sg[i_y,i_spec] - hk_g[i_y,i_spec] ) 
    
    source_energy = 1
    if source_energy == 1:
        mhdot_gen = np.sum(mdotk_g_gen * hk_sg, axis = 1)
    elif source_energy == 2:
        mhdot_gen = np.sum(mdotk_pg * hk_sg, axis = 1)
    
    #%% Calculate pressure and void fraction derivatives in and out of cell vertical boundaries (in kg m s^-2)
    P_bg_interp = interp1d(BedParams['y'], P_bg, kind='linear', fill_value="extrapolate")
    P_bg_bnd    = np.concatenate(( [GasParams['P_in']], P_bg_interp(BedParams['y_bnd'][1:])))
    

    dP_bg       = P_bg_bnd[1:] - P_bg_bnd[:-1]          # pressure drop across cell height [Pa]
    dphi_bg     = phi_bg_bnd[1:] - phi_bg_bnd[:-1]     # solid volume fraction change across cell height [-]
    
    #%%
    if BedParams['energy'] == 1:
        """
        #%% Set gas temperature residuals based on conservation of gas thermal energy [W]
        # 1st term gas enthalpy in/out/gen, 2nd conduction, 3rd gas-particle convection, 4th wall flux, 5 species source term
        res[ind['start'] + ind['T_bg']]	= mhdot_g_in - mhdot_g_out \
                                        + q_cond_y_g \
                                        + A_sg * h_part_g * (T_bs - T_bg) \
                                        + q_wb * phi_bg \
                                        + np.sum(jk_b_hk_b, axis = 1)
                            
        #%% Set particle temperature residuals based on conservation of particle thermal energy [W]
        # 1st term solid enthalpy flow, 2nd solid conduction, 3rd interphasial heat transfer, 4th wall flux
        res[ind['start'] + ind['T_bs']] = mhdot_s_in - mhdot_s_out \
                                        + q_cond_y_s \
                                        - A_sg * h_part_g * (T_bs - T_bg) \
                                        + q_wb * phi_bs \
                                        - mhdot_gen \
                                        - np.sum(jk_b_hk_b, axis = 1)
        """
        # 1st term gas enthalpy in/out/gen, 2nd conduction, 3rd gas-particle convection, 4th wall flux, 5 species source term
        res[ind['start'] + ind['T_bg']]	= mhdot_g_in - mhdot_g_out \
                                        + q_cond_y_g \
                                        + A_sg * h_part_g * (T_bs - T_bg) \
                                        #+ np.sum(jk_b_hk_b, axis = 1)
                                        #- frac_g * mhdot_gen
                                            
        # 1st term solid enthalpy flow, 2nd solid conduction, 3rd interphasial heat transfer, 4th wall flux
        res[ind['start'] + ind['T_bs']] = mhdot_s_in - mhdot_s_out \
                                        + q_cond_y_s \
                                        - A_sg * h_part_g * (T_bs - T_bg) \
                                        + q_wb \
                                        - mhdot_gen \
                                        #- np.sum(jk_b_hk_b, axis = 1)
                    
        #%% Add particle-wall heat transfer flux to the internal wall residual
        res[ind['start'] + ind['T_wb']]     = - q_wb    # heat loss from the bed wall to the particles [W]
    
    #%% Set particle volume fraction residuals based on conservation of particle momentum [kg m s^-2]
    # 1st term particle momentum, 2nd gravity body force, 3rd particle-gas drag, 4th pressure forces
    # 5th solid phase elastic modulus, 6th gas momentum sink from particle model, 7th particle-wall drag
    if BedParams['solid_momentum'] == 1:
        if BedParams['geo'] == 1:
            res[ind['start'] + ind['phi_bs']]	    = (mvdot_s_in - mvdot_s_out) \
                            + BedParams['dVol'] * (-rho_s * phi_bs * g \
                            + beta_drag * (v_bg - v_bs)) \
                            + BedParams['Ay'] * ( - phi_bs * dP_bg \
                            + Gphi * np.maximum(0.15, dphi_bg) ) \
                            - mvdot_gen \
                            - (BedParams['Az_out'] + BedParams['Az_in']) * wb_drag * v_bs * BedParams['dy']
        elif BedParams['geo'] == 2:
            res[ind['start'] + ind['phi_bs']]	    = (mvdot_s_in - mvdot_s_out) \
                            + BedParams['dVol'] * (-rho_s * phi_bs * g \
                            + beta_drag * (v_bg - v_bs)) \
                            + BedParams['Ay'] * ( - phi_bs * dP_bg  \
                            + Gphi * np.maximum(0.15, dphi_bg) ) \
                            - mvdot_gen \
                            - (BedParams['Ar_out'] + BedParams['Ar_in']) * wb_drag * v_bs * BedParams['dy']
        
    #%% Set gas pressure residuals based on conservation of gas momentum (in kg m s^-2)
    # 1st term gas advection, 2nd gravity, 3rd gas-solid drag, 4th pressure, 5 species momentum source term
    if BedParams['gas_momentum'] == 1:
        res[ind['start']+ind['P_bg']]	= (mvdot_g_in - mvdot_g_out) \
                        + BedParams['dVol'] * (- rho_g * phi_bg * g \
                        - beta_drag*(v_bg - v_bs)) \
                        - BedParams['Ay'] * phi_bg * dP_bg \
                        + mvdot_gen
    
    #%% Set gas mass fractions residuals based on species equation
    source = 2 # 1:mdotk_g_gen, 2: mdotk_pg
    if source == 1:
        spec_source = mdotk_g_gen
    elif source == 2:
        spec_source = mdotk_pg
    
    for i_spec in range(gas['kspec'] ):
        if SurfParams['chem'] == 1:
            if BedParams['species_dispersion'] == 1:
                res[ind['start'] + ind['Yk_bg'][i_spec]] = mdotk_g_in[:, i_spec] - mdotk_g_out[:, i_spec] \
                                                         + j_dis_y_g_in[:, i_spec] - j_dis_y_g_out[:, i_spec] \
                                                         + spec_source[:, i_spec]
            elif BedParams['species_dispersion'] == 2:
                res[ind['start'] + ind['Yk_bg'][i_spec]] = mdotk_g_in[:, i_spec] - mdotk_g_out[:, i_spec] \
                                                         + spec_source[:, i_spec]
            
    #%% Boundary Condition for residual equations
    res[ind['start'][0] + ind['Yk_bg']]     = 1e2*(Yk_bg[0,:] - GasParams['Y_in'][:])  # inlet mass fraction
    #res[ind['start'][0] + ind['phi_bs']]    = 1e2*(phi_bs[1] - phi_bs[0])    # inlet solid volume fraction
    #res[ind['start'][0] + ind['P_bg']]      = 1e4*(P_bg[0] - GasParams['P_in'])    # inlet gas pressure
    res[ind['start'][0] + ind['T_bg']]      = 1e2*(T_bg[0] - GasParams['T_in'])
    #res[ind['start'][1] + ind['T_bg']]      = 1e4*(T_bg[1] - T_bg[0])
    
    #%% Reshape residuals for easier inspection
    res_reshape = res.reshape(BedParams['n_y'], ind['vars'])
    
    return res