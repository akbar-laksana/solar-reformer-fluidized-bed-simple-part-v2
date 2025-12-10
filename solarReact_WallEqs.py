import numpy as np
import sys
import os

functions_folder = os.path.join(".", 'Functions')
sys.path.append(functions_folder)

from Thermal_Cond import Thermal_Cond

#%% Define and calculate residual equations pertaining to wall temperature
def solarReact_WallEqs(wall, env, BedParams, WallParams, PartParams, EnvParams, ind, T_we, T_wb):
    # Constants
    g = 9.81            # Gravitational constant [m/s^2]
    sigma = 5.67e-8     # Stefan-Boltzmann constant [W/m^2-K^4]
    
    # Initialize variables
    n_y = WallParams['n_y']
    res = np.zeros(ind['tot'])
    
    q_rad_in    = np.zeros(n_y)
    q_rad_out   = np.zeros(n_y)
    
    #%% Calculate solar radiation absorbed and thermal radiation emitted by wall
    if BedParams['geo'] == 1:
        q_rad_in[BedParams['index_b']]  = WallParams['Az_out_b'] * WallParams['abs'] * EnvParams['q_sol'] * EnvParams['q_sol_dist']
        q_rad_out[BedParams['index_b']] = EnvParams['f'] * sigma * WallParams['Az_out_b'] * WallParams['emis'] * (T_we[BedParams['index_b']]**4 - EnvParams['T']**4)

    elif BedParams['geo'] == 2:
        q_rad_in[BedParams['index_b']]  = WallParams['Ar_out_b'] * WallParams['abs'] * EnvParams['q_sol'] * EnvParams['q_sol_dist']
        q_rad_out[BedParams['index_b']] = EnvParams['f'] * sigma * WallParams['Ar_out_b'] * WallParams['emis'] * (T_we[BedParams['index_b']]**4 - EnvParams['T']**4)
    
    #%% Conductive Heat Transfer from external to internal wall
    k_w_z = Thermal_Cond(WallParams['id'], 0.5 * (T_we + T_wb))
    
    if BedParams['geo'] == 1:
        q_cond_z = WallParams['Az'] * k_w_z * (T_we - T_wb) / WallParams['dz']
    elif BedParams['geo'] == 2:
        q_cond_r = 2 * np.pi * WallParams['dy'] * k_w_z * (T_we - T_wb) / np.log(WallParams['d_out']/WallParams['d_in'])
    
    #%% Find the external convective heat transfer on the external wall modeled 
    #   Use a vertical 1-D wall heat transfer correlation
    T_film = (T_we + EnvParams['T']) / 2    # Film Temperature [K]
    beta = 1.0 / T_film                     # Coefficient of volume expansion [1/K]
    
    # Initialize variables
    k_inf   = np.zeros(n_y)
    Nu_ext  = np.zeros(n_y)
    q_conv_ext = np.zeros(n_y)
    
    for i_y in range(n_y):
        # Retrieve gas properties
        env['obj'].TP   = T_film[i_y], EnvParams['P']
        rho_inf         = env['obj'].density
        mu_inf          = env['obj'].viscosity
        cp_inf          = env['obj'].cp_mass
        k_inf[i_y]      = env['obj'].thermal_conductivity
        
        alpha_inf   = k_inf[i_y] / (rho_inf * cp_inf)
        nu_inf      = mu_inf / rho_inf
        Pr          = nu_inf / alpha_inf 
        
        # Calculate local Rayleigh number and natural convection vertical plate Nusselt number
        Ra_y = g * beta[i_y] * abs(T_we[i_y] - EnvParams['T']) * WallParams['y'][i_y]**3 / (alpha_inf * nu_inf)

        # Nusselt Number calculation
        if Ra_y < 1e9:
            Nu_ext[i_y] = 0.68 + (0.67*Ra_y**0.25)/((1 + (0.492/Pr)**0.5625)**0.4444)
        else:
            Nu_ext[i_y] = (0.892 + (0.387*Ra_y**0.1667)/((1 + (0.492/Pr)**0.5625)**0.2963))**2

    # Calculate the external wall convection heat loss
    h_ext = Nu_ext * k_inf / WallParams['y']    # % external wall heat transfer coefficient [W/m^2-K]
    h_ext[0] = h_ext[1]
    
    if BedParams['geo'] == 1:
        q_conv_ext[BedParams['index_b']] = h_ext[BedParams['index_b']] * WallParams['Az_out_b'] * (T_we[BedParams['index_b']] - EnvParams['T'])   # [W]

    elif BedParams['geo'] == 2:
        q_conv_ext[BedParams['index_b']] = h_ext[BedParams['index_b']] * WallParams['Ar_out_b'] * (T_we[BedParams['index_b']] - EnvParams['T'])   # [W]
    
    #%% Find the vertical heat conduction in the internal and external walls
    #   Set temperature at the bottom of the bed to the particle outlet
    T_we_vec    = np.concatenate(([T_we[0]], T_we, [T_we[-1]]))
    T_wb_vec    = np.concatenate(([T_wb[0]], T_wb, [T_wb[-1]]))
    
    k_we        = Thermal_Cond(WallParams['id'], T_we_vec)
    k_wb        = Thermal_Cond(WallParams['id'], T_wb_vec)
    
    k_we_avg    = 0.5 * (k_we[:-1] + k_we[1:])
    k_wb_avg    = 0.5 * (k_wb[:-1] + k_wb[1:])

    q_cond_y_we = 0.5 * WallParams['Ay_cond'] * k_we_avg * (T_we_vec[:-1] - T_we_vec[1:]) / WallParams['dy_cond']
    q_cond_y_wb = 0.5 * WallParams['Ay_cond'] * k_wb_avg * (T_wb_vec[:-1] - T_wb_vec[1:]) / WallParams['dy_cond']

    #%% Energy Balance & Residuals
    #   The internal wall residual does not include the heat transfer to the bed which is added through
    #    the fluidized bed calclations
    if BedParams['geo'] == 1:
        res[ind['start'] + ind['T_we']] = (q_rad_in - q_rad_out - q_conv_ext - q_cond_z \
                                       + q_cond_y_we[:-1] - q_cond_y_we[1:])
        res[ind['start'] + ind['T_wb']] = (q_cond_z + q_cond_y_wb[:-1] - q_cond_y_wb[1:])
    elif BedParams['geo'] == 2:
        res[ind['start'] + ind['T_we']] = (q_rad_in - q_rad_out - q_conv_ext - q_cond_r \
                                       + q_cond_y_we[:-1] - q_cond_y_we[1:])
        res[ind['start'] + ind['T_wb']] = (q_cond_r + q_cond_y_wb[:-1] - q_cond_y_wb[1:])
    
    #%% Reshape residuals for easier inspection
    res_reshape = res.reshape(BedParams['n_y'], ind['vars'])
    
    return res

#%% #%% Define and calculate residual equations pertaining to wall temperature for a isothermal case
def solarReact_WallEqs_Isothermal(env, BedParams, WallParams, EnvParams, ind, scale, T_we, T_wb, T_bs, T_bg):
    #%% Initialize residual and property vectors calculated from Cantera objects
    res             = np.zeros(ind['tot'])      # initialized residual
    
    # Constants
    g = 9.81            # Gravitational constant [m/s^2]
    sigma = 5.67e-8     # Stefan-Boltzmann constant [W/m^2-K^4]
    
    n_y = WallParams['n_y']
    
    q_abs_in    = np.zeros(n_y)
    q_rad_out   = np.zeros(n_y)
    
    #%% Calculate solar radiation absorbed and thermal radiation emitted by wall
    if BedParams['geo'] == 1:
        q_abs_in[BedParams['index_b']]  = WallParams['Az_out_b'] * WallParams['abs'] * EnvParams['q_sol']
        q_rad_out[BedParams['index_b']] = EnvParams['f'] * sigma * WallParams['Az_out_b'] * WallParams['emis'] * (T_we[BedParams['index_b']]**4 - EnvParams['T']**4)
        q_abs_in_tot                    = np.sum(q_abs_in)
        q_rad_out_tot                   = np.sum(q_rad_out)
    elif BedParams['geo'] == 2:
        q_abs_in[BedParams['index_b']]  = WallParams['Ar_out_b'] * WallParams['abs'] * EnvParams['q_sol']
        q_rad_out[BedParams['index_b']] = EnvParams['f'] * sigma * WallParams['Ar_out_b'] * WallParams['emis'] * (T_we[BedParams['index_b']]**4 - EnvParams['T']**4)
        q_abs_in_tot                    = np.sum(q_abs_in)
        q_rad_out_tot                   = np.sum(q_rad_out)
    
    #%% Conductive Heat Transfer from external to internal wall
    k_w_z = Thermal_Cond(WallParams['id'], 0.5 * (T_we + T_wb))
    
    if BedParams['geo'] == 1:
        q_cond_z        = WallParams['Az'] * k_w_z * (T_we - T_wb) / WallParams['dz']
        q_cond_w_tot    = np.sum(q_cond_z)
    elif BedParams['geo'] == 2:
        q_cond_r = 2 * np.pi * WallParams['dy'] * k_w_z * (T_we - T_wb) / np.log(WallParams['d_out']/WallParams['d_in'])
        q_cond_w_tot   = np.sum(q_cond_r)
    
    #%% Find the external convective heat transfer on the external wall  
    #   Modeled with a vertical 1-D wall heat transfer correlation for flat plate
    T_film = (T_we + EnvParams['T']) / 2    # Film Temperature [K]
    beta = 1.0 / T_film                     # Coefficient of volume expansion [1/K]
    
    # Initialize variables
    k_inf   = np.zeros(n_y)
    Nu_ext  = np.zeros(n_y)
    q_conv_out = np.zeros(n_y)
    
    for i_y in range(n_y):
        # Retrieve gas properties
        env['obj'].TP   = T_film[i_y], EnvParams['P']
        rho_inf         = env['obj'].density
        mu_inf          = env['obj'].viscosity
        cp_inf          = env['obj'].cp_mass
        k_inf[i_y]      = env['obj'].thermal_conductivity
        
        alpha_inf   = k_inf[i_y] / (rho_inf * cp_inf)
        nu_inf      = mu_inf / rho_inf
        Pr          = nu_inf / alpha_inf 
        
        # Calculate local Rayleigh number and natural convection vertical plate Nusselt number
        Ra_y = g * beta[i_y] * abs(T_we[i_y] - EnvParams['T']) * WallParams['y'][i_y]**3 / (alpha_inf * nu_inf)

        # Nusselt Number calculation
        if Ra_y < 1e9:
            Nu_ext[i_y] = 0.68 + (0.67*Ra_y**0.25)/((1 + (0.492/Pr)**0.5625)**0.4444)
        else:
            Nu_ext[i_y] = (0.892 + (0.387*Ra_y**0.1667)/((1 + (0.492/Pr)**0.5625)**0.2963))**2

    # Calculate the external wall convection heat loss
    h_ext = Nu_ext * k_inf / WallParams['y']    # % external wall heat transfer coefficient [W/m^2-K]
    h_ext[0] = h_ext[1]
    
    if BedParams['geo'] == 1:
        q_conv_out[BedParams['index_b']] = h_ext[BedParams['index_b']] * WallParams['Az_out_b'] * (T_we[BedParams['index_b']] - EnvParams['T'])   # [W]
    elif BedParams['geo'] == 2:
        q_conv_out[BedParams['index_b']] = h_ext[BedParams['index_b']] * WallParams['Ar_out_b'] * (T_we[BedParams['index_b']] - EnvParams['T'])   # [W]
    
    q_conv_out_tot = np.sum(q_conv_out)
    
    #%% Set residual equations
    res[ind['start'][0] + ind['T_we']] = q_abs_in_tot - q_rad_out_tot - q_conv_out_tot - q_cond_w_tot
    res[ind['start'][1:] + ind['T_we']] = 0
    
    res[ind['start'][0] + ind['T_wb']] = q_cond_w_tot
    res[ind['start'][1:] + ind['T_wb']] = 0
    
    #%% Reshape residuals for easier inspection
    res_reshape = res.reshape(BedParams['n_y'], ind['vars'])   
    
    return res