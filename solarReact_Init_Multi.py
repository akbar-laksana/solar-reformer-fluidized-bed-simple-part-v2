import numpy as np
import sys
import os
import matplotlib.pyplot as plt

from scipy.stats import beta

functions_folder = os.path.join(".", 'Functions')
sys.path.append(functions_folder)

from Thermal_Cond import Thermal_Cond

#%% Initialization function
def solarReact_Init_Multi(gas, gas_surf, part, wall, surf, GasParams, PartParams, BedParams, WallParams, SurfParams, EnvParams):
    #%% Set up state variable indices
    ind = {}
    
    #%% Reactor Model Variables that are resolved spatially in the bed height direction
    if BedParams['energy'] == 1:
        if BedParams['gas_momentum'] == 1 and BedParams['solid_momentum'] == 1:
            ind['reactormod_varnames'] = ['T_we', 'T_wb', 'T_bs', 'T_bg', 'phi_bs', 'P_bg', 'Yk_bg']
        elif BedParams['gas_momentum'] != 1 and BedParams['solid_momentum'] == 1:
            ind['reactormod_varnames'] = ['T_we', 'T_wb', 'T_bs', 'T_bg', 'phi_bs', 'Yk_bg']
        elif BedParams['gas_momentum'] != 1 and BedParams['solid_momentum'] != 1:
            ind['reactormod_varnames'] = ['T_we', 'T_wb', 'T_bs', 'T_bg', 'Yk_bg']
    elif BedParams['energy'] == 2:
        if BedParams['gas_momentum'] == 1 and BedParams['solid_momentum'] == 1:
            ind['reactormod_varnames'] = ['phi_bs', 'P_bg', 'Yk_bg']
        elif BedParams['gas_momentum'] != 1 and BedParams['solid_momentum'] == 1:
            ind['reactormod_varnames'] = ['phi_bs', 'Yk_bg']
        elif BedParams['gas_momentum'] != 1 and BedParams['solid_momentum'] != 1:
            ind['reactormod_varnames'] = ['Yk_bg']
            
    # Total number of reactormod variables (minus all gas species)
    ind['reactormod_vars'] = len(ind['reactormod_varnames'])
    
    # Set array of Yk_bg indices
    ind['Yk_bg'] = np.zeros((gas['kspec']))
    
    # Setting of variable indices
    for i_var in range(ind['reactormod_vars']):
       if ind['reactormod_varnames'][i_var] == 'Yk_bg':
           ind['Yk_bg'] = i_var + np.arange(gas['kspec'] )
       else:
           ind[ind['reactormod_varnames'][i_var]] = i_var
    
    # Add N-1 species to the total number of reactormod variables       
    ind['reactormod_vars'] += (gas['kspec'] - 1)
    
    #%% Particle Model Variables: Only solve for Yk in the particle 
    # Since dT/dr and dP/dr = 0, particle model variables only consists of Yk_p
    ind['partmod_varnames'] = ['Yk_p']
    ind['partmod_vars'] = len(ind['partmod_varnames'])
    partmod_vars_start  = ind['reactormod_vars'] - 1
    
    # Set array of Yk_p indices
    ind['Yk_p'] = np.zeros(( gas['kspec'] * SurfParams['n_p'] ))
    
    # Setting of indices depends on whether the model is multi cell or single cell particle
    if SurfParams['n_p'] > 1:
        for i_var in range(ind['partmod_vars']):
            if ind['partmod_varnames'][i_var] == 'Yk_p':
                for i_p in range(SurfParams['n_p']):
                    ind['Yk_p'][( (i_p * gas['kspec']) + ( np.arange(gas['kspec']) ) )] = partmod_vars_start + i_var + 1 + np.arange(gas['kspec'] )
                
                    partmod_vars_start += ( gas['kspec'] )
    
    else:
        for i_var in range(ind['partmod_vars']):
            if ind['partmod_varnames'][i_var] == 'Yk_p':
                ind['Yk_p'] = partmod_vars_start + i_var + 1 + np.arange(gas['kspec'] )
    
    ind['partmod_vars'] = (ind['partmod_vars'] + (gas['kspec'] - 1)) * SurfParams['n_p']
    
    #%% Tabulate total variables for spatially resolved reactor and particle
    ind['vars']     = ind['reactormod_vars'] + ind['partmod_vars']
    
    #%% Tabulate all variables and start indices for spatially resolved variables
    # Defines the start indices for each control volume at the bed zone
    ind['start_b']  = np.arange(BedParams['n_y_b'])  * ind['vars'] 
    # Since there is no freeboard zone, all start indices only comprises of the bed 
    ind['start']    = ind['start_b']
    
    # Total size of variables
    ind['tot']      = BedParams['n_y'] * ind['vars']
    
    #%% Initialize SV_guess and estimate variables
    SV_guess = np.zeros((ind['tot']))    # Initialize the SV solution vector
    
    scale = {}
    scale['res'] = np.ones((ind['tot']))
    scale['res_2'] = np.ones((ind['tot']))
    scale['var'] = np.ones((ind['tot']))
    
    #%% Estimated performance metrics
    hT_guess    = 2200                           # [W/m^2-K]
    eta_th      = 0.70                           # [-]
    
    #%% Calculate absorbed solar flux
    q_abs       = EnvParams['q_sol_dist'] * BedParams['n_w'] * EnvParams['q_sol']             # [W/m^2]
    if BedParams['geo'] == 1:
        Q_abs       = np.sum( q_abs * WallParams['Az_out_b'] )
    elif BedParams['geo'] == 2:
        Q_abs       = np.sum( q_abs * WallParams['Ar_out_b'])
    
    #%% Estimated linear gas pressure
    P_bg        = np.zeros((BedParams['n_y']))
     
    dP_g        = BedParams['P_drop']*(BedParams['y_b'][-1] - BedParams['y_b'][0]) # Estimated pressure drop of ~15-20 kPa/m
    P_bg_out    = GasParams['P_in'] - dP_g                                      # Estimated outlet pressure of fluidizing gas [Pa]
     
    P_bg[BedParams['index_b']] = np.linspace(GasParams['P_in'], P_bg_out, BedParams['n_y_b'])   # [Pa]
    
    #%% Estimate particle volume fraction with piecewise parabola
    initialize_phi_bs = 'piecewise'
    if initialize_phi_bs == 'piecewise':
        # Prescribed points
        Mid_fraction = 0.2
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
    
    elif initialize_phi_bs == 'linear':
        phi_bs      = np.zeros((BedParams['n_y']))
        phi_bs_0    = PartParams['phi_bs_max'] - 0.1
        phi_bs_top  = PartParams['phi_bs_max'] - 0.25
        
        phi_bs[BedParams['index_b']] = np.linspace(phi_bs_0, phi_bs_top, BedParams['n_y_b'])
        
    # Gas volume fraction
    phi_bg = 1 - phi_bs
    
    #%% Estimated mass fraction of species and bed temperature
    Yk_bg = np.zeros((BedParams['n_y'], gas['kspec']))
    T_bg  = np.zeros((BedParams['n_y']))
    T_bs  = np.zeros((BedParams['n_y']))
    
    # Estimate fudge factor to calculate actual outlet gas species mass fraction compared to equilibrated value
    epsilon = 1.0
    
    # Set temperature profile for gas phase along the bed
    Deltah_g    = (Q_abs*eta_th)/GasParams['mdot_in']
    h_out_g     = GasParams['h_in'] + Deltah_g
    
    # Inlet gas condition
    gas['obj'].HPY  = GasParams['h_in'], GasParams['P_in'], GasParams['Y_in']
    Xk_bg_in        = gas['obj'].X
    
    # Set gas condition for Equlibrium calculation
    gas['obj'].HPY  = h_out_g, P_bg_out, GasParams['Y_in']
    gas['obj'].equilibrate('HP')
    Yk_bg_out_eq   = gas['obj'].Y
    Xk_bg_out_eq   = gas['obj'].X
    T_bg_out_eq    = gas['obj'].T
    
    # Modify initialization with 'extent of reaction' factor, values unchanged if kappa = 1
    Yk_bg_out       = GasParams['Y_in'] + epsilon * (Yk_bg_out_eq - GasParams['Y_in'])
    gas['obj'].HPY  = h_out_g, P_bg_out, Yk_bg_out
    Xk_bg_out       = gas['obj'].X
    T_bg_out        = gas['obj'].T
    
    # Limit to prevent properties eval to fall outside of range
    T_bg_out        = np.minimum(T_bg_out, 1400)
    
    initialize_T = 'constant'
    # Estimated gas temperature profile
    if initialize_T == 'constant':
        T_bg = np.ones((BedParams['n_y'])) * (GasParams['T_in'] + T_bg_out)/2 #GasParams['T_in'] #
    elif initialize_T == 'linear':
        T_bg[:] = np.linspace(GasParams['T_in'], T_bg_out, BedParams['n_y'])
    elif initialize_T == 'exponential':
        T_bg[:] = GasParams['T_in'] + (T_bg_out - GasParams['T_in']) * (1 - np.exp(-10 * BedParams['y'])) / (1 - np.exp(-10))
    
    # Estimated solid temperature profile
    T_bs = T_bg
    
    initialize_Y = 'linear'
    # Set the species profile of gas species along the bed height, two selection: linear and exponential
    for i_spec in range(gas['kspec']):
        if initialize_Y == 'linear':
            Yk_bg[:, i_spec] = np.linspace(GasParams['Y_in'][i_spec], Yk_bg_out[i_spec], BedParams['n_y'])
        elif initialize_Y == 'exponential':
            Yk_bg[:, i_spec] = GasParams['Y_in'][i_spec] + (Yk_bg_out[i_spec] - GasParams['Y_in'][i_spec]) * (1 - np.exp(-10 * BedParams['y'])) / (1 - np.exp(-10))
            
    # Override if negative
    for i_y in range(BedParams['n_y']):
        for i_spec in range(gas['kspec']):        
            if Yk_bg[i_y, i_spec] < 0:
                Yk_bg[i_y, i_spec] = 1e-10
    
    Xk_bg   = np.zeros((BedParams['n_y'], gas['kspec']))
    for i_y in range(BedParams['n_y']):
        # Set gas object
        gas['obj'].TPY = T_bg[i_y], P_bg[i_y], Yk_bg[i_y, :]
        Xk_bg[i_y, :]  = gas['obj'].X
    
    #%% Estimate wall temperatures
    T_wb = np.zeros((BedParams['n_y']))
    T_we = np.zeros((BedParams['n_y']))
    
    T_wb[BedParams['index_b']]  = T_bs[BedParams['index_b']]  + eta_th * q_abs / hT_guess
    
    k_w     = Thermal_Cond(WallParams['id'], T_wb)
    
    # estimated external-wall temperature [K]
    if BedParams['geo'] == 1:
        T_we[BedParams['index_b']] = T_wb[BedParams['index_b']] + eta_th * q_abs * WallParams['dz_b']/k_w[BedParams['index_b']]
    elif BedParams['geo'] == 2:
        T_we[BedParams['index_b']] = T_wb[BedParams['index_b']] + eta_th * q_abs * (WallParams['d_out_b']*np.log(WallParams['d_out_b']/WallParams['d_in_b'])/(2*k_w[BedParams['index_b']]))
        
    #%% Estimate gas and particle velocity
    rho_g = np.zeros((BedParams['n_y']))
    mu_g = np.zeros((BedParams['n_y']))
    k_g = np.zeros((BedParams['n_y']))
    rho_s = np.zeros((BedParams['n_y']))
    cp_s = np.zeros((BedParams['n_y']))
   
    for i_y in range(BedParams['n_y']):
        # Set gas state to get properties
        gas['obj'].TPY  = T_bg[i_y], P_bg[i_y], Yk_bg[i_y, :]
        rho_g[i_y]      = gas['obj'].density
        mu_g[i_y]       = gas['obj'].viscosity
        k_g[i_y]        = gas['obj'].thermal_conductivity
        
        # Set particle state to get properties
        part['obj'].TP  = T_bs[i_y], P_bg[i_y]                   
        rho_s[i_y]      = part['obj'].density*(1-SurfParams['phi'])
        cp_s[i_y]       = part['obj'].cp_mass        
    
    # Gas and particle velocity        
    v_bg = GasParams['mdot_in'] / (rho_g * BedParams['Ay'] * (1 - phi_bs))  # [m/s]
    v_bs = PartParams['mdot_in'] / (rho_s * BedParams['Ay'] * phi_bs)  # [m/s]
    
    #%% Check U_hat
    dp              = PartParams['dp']                                            
    phi_mf          = PartParams['phi_bs_max']
    g               = 9.81                                    
                                                                                                           
    # Calculate minimum fluidization, superficial, and dimensionless excess gas velocities
    a               = 1.75*phi_mf*rho_g/(dp*(1-phi_mf)**3)                   
    b               = 150*phi_mf**2*mu_g/(dp**2*(1-phi_mf)**3)              
    c               = -(phi_mf*rho_s - (1-phi_mf)*rho_g)*g                  

    U_mf            = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)                     
    Ug              = v_bg * phi_bg                                                                                 
    U_hat           = (rho_s*cp_s/(g*k_g))**(1/3)*(Ug - U_mf)
    U_hat_avg       = np.average(U_hat)
    
    #%% Estimate gas concentration within the particle'
    
    # Fudge factor of conversion within the particle
    epsilon_p = 1.0
    
    if SurfParams['n_p'] > 1:
        Yk_p    = np.zeros((BedParams['n_y'], gas['kspec'], SurfParams['n_p']))
        Xk_p    = np.zeros((BedParams['n_y'], gas['kspec'], SurfParams['n_p']))
        Yk_p_eq = np.zeros((BedParams['n_y'], gas['kspec'], SurfParams['n_p']))
        Xk_p_eq = np.zeros((BedParams['n_y'], gas['kspec'], SurfParams['n_p']))
        SurfParams['Zk_p_init'] = np.zeros((BedParams['n_y'], surf['kspec'], SurfParams['n_p']))
    
        for i_p in range(SurfParams['n_p']):    
            for i_y in range(BedParams['n_y']):
                # Set gas object
                gas['obj'].TPY = T_bg[i_y], P_bg[i_y], Yk_bg[i_y, :]
                gas['obj'].equilibrate('TP')
                Yk_p_eq[i_y, :, i_p] = gas['obj'].Y
                Xk_p_eq[i_y, :, i_p] = gas['obj'].X
                
                # Estimate particle concentration based on fudge factor "kappa"
                Yk_p[i_y, :, i_p]   = Yk_bg[i_y, :] + epsilon_p * (Yk_p_eq[i_y, :, i_p] - Yk_bg[i_y, :])
                gas['obj'].TPY      = T_bg[i_y], P_bg[i_y], Yk_p[i_y, :, i_p]
                Xk_p[i_y, :, i_p]   = gas['obj'].X
                
                # Set surface object and integrate
                gas_surf['obj'].TPY = T_bg[i_y], P_bg[i_y], np.concatenate((Yk_p[i_y, :, i_p], [0]))
                
                surf['obj'].TP = T_bs[i_y], P_bg[i_y]
                surf['obj'].coverages = SurfParams['Zk_p_0'][:]
                if BedParams['kinetics'] == 2:
                    surf['obj'].advance_coverages(1.5e2, 1e-7, 1e-14, 1e-0, 5e8, 20)
                SurfParams['Zk_p_init'][i_y, :, i_p] = surf['obj'].coverages
    
    else:
        Yk_p    = np.zeros((BedParams['n_y'], gas['kspec']))
        Xk_p    = np.zeros((BedParams['n_y'], gas['kspec']))
        Yk_p_eq = np.zeros((BedParams['n_y'], gas['kspec']))
        Xk_p_eq = np.zeros((BedParams['n_y'], gas['kspec']))
        SurfParams['Zk_p_init'] = np.zeros((BedParams['n_y'], surf['kspec']))
    
        for i_y in range(BedParams['n_y']):
            # Set gas object
            gas['obj'].TPY = T_bg[i_y], P_bg[i_y], Yk_bg[i_y, :]
            gas['obj'].equilibrate('TP')
            Yk_p_eq[i_y, :] = gas['obj'].Y
            Xk_p_eq[i_y, :] = gas['obj'].X
        
            # Estimate particle concentration based on fudge factor "kappa"
            Yk_p[i_y, :]    = Yk_bg[i_y, :] + epsilon_p * (Yk_p_eq[i_y, :] - Yk_bg[i_y, :])
            gas['obj'].TPY  = T_bg[i_y], P_bg[i_y], Yk_p[i_y, :]
            Xk_p[i_y, :]    = gas['obj'].X
            
            # Set surface object and integrate
            gas_surf['obj'].TPY = T_bg[i_y], P_bg[i_y], np.concatenate((Yk_p[i_y, :], [0]))
            
            surf['obj'].TP = T_bs[i_y], P_bg[i_y]
            surf['obj'].coverages = SurfParams['Zk_p_0'][:]
            if BedParams['kinetics'] == 2:
                surf['obj'].advance_coverages(1.5e2, 1e-7, 1e-14, 1e-0, 5e8, 20)
            SurfParams['Zk_p_init'][i_y, :] = surf['obj'].coverages
            
    #%% Lsqnonlin Bounds
    # Set generic bounds of 0, inf
    boundsLow =  np.zeros((ind['tot']))
    boundsUp = np.ones((ind['tot']))*np.inf
    
    #%% Construct SV_guess
    #%% For spatially resolved reactor variables
    for i_var in range(ind['reactormod_vars'] - (gas['kspec'] - 1)):
        var_name = ind['reactormod_varnames'][i_var]
        
        # External Receiver Wall Temperature
        if var_name == 'T_we':  
            scale['var'][ind['start'] + ind['T_we']]    = 1e3
            scale['res_T_we']                           = 1e0
            scale['res'][ind['start'] + ind['T_we']]    = scale['res_T_we'] * abs(GasParams['mdot_in'] * GasParams['cp_in'])
            SV_guess[ind['start'] + ind['T_we']]        = T_we / scale['var'][ind['start'] + ind['T_we']]
            boundsLow[ind['start'] + ind['T_we']]       = (GasParams['T_in'] - 750) / scale['var'][ind['start'] + ind['T_we']]
            boundsUp[ind['start'] + ind['T_we']]        = 3000 / scale['var'][ind['start'] + ind['T_we']]
            
            # res_2 denotes scaling when restart is activated
            scale['res_2_T_we']                         = 1e0
            scale['res_2'][ind['start'] + ind['T_we']]  = scale['res_2_T_we'] * abs(GasParams['mdot_in'] * GasParams['cp_in'])
        
        # Fluidized Bed Side Wall Temperature    
        elif var_name == 'T_wb':  
            scale['var'][ind['start'] + ind['T_wb']]    = 1e3
            scale['res_T_wb']                           = 1e0
            scale['res'][ind['start'] + ind['T_wb']]    = scale['res_T_wb'] * abs(GasParams['mdot_in'] * GasParams['cp_in'])
            SV_guess[ind['start'] + ind['T_wb']]        = T_wb / scale['var'][ind['start'] + ind['T_wb']]
            boundsLow[ind['start'] + ind['T_wb']]       = (GasParams['T_in'] - 750) / scale['var'][ind['start'] + ind['T_wb']]
            boundsUp[ind['start'] + ind['T_wb']]        = 3000 / scale['var'][ind['start'] + ind['T_wb']]
            
            # res_2 denotes scaling when restart is activated
            scale['res_2_T_wb']                         = 1e0
            scale['res_2'][ind['start'] + ind['T_wb']]  = scale['res_2_T_wb'] * abs(GasParams['mdot_in'] * GasParams['cp_in'])
        
        # Particle Solid Temperature    
        elif var_name == 'T_bs':  
            scale['var'][ind['start'] + ind['T_bs']]   = 1e3
            scale['res_T_bs']                          = 1e0
            scale['res'][ind['start'] + ind['T_bs']]   = scale['res_T_bs'] * abs(GasParams['mdot_in'] * GasParams['cp_in'])
            SV_guess[ind['start'] + ind['T_bs']]       = T_bs / scale['var'][ind['start'] + ind['T_bs']]
            boundsLow[ind['start'] + ind['T_bs']]      = (GasParams['T_in'] - 750) / scale['var'][ind['start'] + ind['T_bs']]
            boundsUp[ind['start'] + ind['T_bs']]       = 3000 / scale['var'][ind['start'] + ind['T_bs']]
            
            # res_2 denotes scaling when restart is activated
            scale['res_2_T_bs']                        = 1e0
            scale['res_2'][ind['start'] + ind['T_bs']] = scale['res_2_T_bs'] * abs(GasParams['mdot_in'] * GasParams['cp_in'])
            
        # Fluidizing Gas Temperature    
        elif var_name == 'T_bg':  
            scale['var'][ind['start'] + ind['T_bg']]    = 1e3
            scale['res_T_bg']                           = 1e-0
            scale['res'][ind['start'] + ind['T_bg']]    = scale['res_T_bg'] * abs(GasParams['mdot_in'] * GasParams['cp_in'])    
            SV_guess[ind['start'] + ind['T_bg']]        = T_bg / scale['var'][ind['start'] + ind['T_bg']]
            boundsLow[ind['start'] + ind['T_bg']]       = (GasParams['T_in'] - 750) / scale['var'][ind['start'] + ind['T_bg']]
            boundsUp[ind['start'] + ind['T_bg']]        = 3000 / scale['var'][ind['start'] + ind['T_bg']]
            
            # res_2 denotes scaling when restart is activated
            scale['res_2_T_bg']                         = 1e-0
            scale['res_2'][ind['start'] + ind['T_bg']]  = scale['res_2_T_bg'] * abs(GasParams['mdot_in'] * GasParams['cp_in']) 
        
        # Particle Solid Volume Fraction    
        elif var_name == 'phi_bs':  
            scale['var'][ind['start'] + ind['phi_bs']]  = 1e0
            scale['res_phi_bs']                         = 1e0
            scale['res'][ind['start'] + ind['phi_bs']]  = scale['res_phi_bs'] * abs(GasParams['mdot_in'] )  #* v_bg[0]
            SV_guess[ind['start'] + ind['phi_bs']]        = phi_bs / scale['var'][ind['start'] + ind['phi_bs']]
            boundsLow[ind['start'] + ind['phi_bs']]     = 0.2 / scale['var'][ind['start'] + ind['phi_bs']]
            boundsUp[ind['start'] + ind['phi_bs']]      = 0.8 / scale['var'][ind['start'] + ind['phi_bs']]
            
            # res_2 denotes scaling when restart is activated
            scale['res_2_phi_bs']                         = 1e-0
            scale['res_2'][ind['start'] + ind['phi_bs']]  = scale['res_2_phi_bs'] * abs(GasParams['mdot_in']  ) #* v_bg[0]
            
        # Particle Solid Volume Fraction
        elif var_name == 'P_bg':  
            scale['var'][ind['start'] + ind['P_bg']]    = 1e5
            scale['res_P_bg']                           = 1e0
            scale['res'][ind['start'] + ind['P_bg']]    = scale['res_P_bg'] * abs(GasParams['mdot_in']  ) #* v_bg[0]
            SV_guess[ind['start'] + ind['P_bg']]        = P_bg / scale['var'][ind['start'] + ind['P_bg']]
            boundsLow[ind['start'] + ind['P_bg']]       = (0.5*P_bg_out) / scale['var'][ind['start'] + ind['P_bg']]
            boundsUp[ind['start'] + ind['P_bg']]        = (2*P_bg_out) / scale['var'][ind['start'] + ind['P_bg']]
            
            # res_2 denotes scaling when restart is activated
            scale['res_2_P_bg']                         = 1e-0
            scale['res_2'][ind['start'] + ind['P_bg']]  = scale['res_2_P_bg'] * abs(GasParams['mdot_in']  ) #* v_bg[0]
            
        # Gas Mass Fraction at the bed
        elif var_name == 'Yk_bg':  
            for i_spec in range(gas['kspec'] ):
                scale['var'][ind['start'] + ind['Yk_bg'][i_spec]]   = 1e0
                if i_spec == gas['kH2']:
                    scale['res_Yk_bg']                              = 1e-2
                    scale['res_2_Yk_bg']                            = 1e-2
                else:
                    scale['res_Yk_bg']                              = 1e-2
                    scale['res_2_Yk_bg']                            = 1e-2
                scale['res'][ind['start'] + ind['Yk_bg'][i_spec]]   = scale['res_Yk_bg'] * abs(GasParams['mdot_in']) 
                SV_guess[ind['start'] + ind['Yk_bg'][i_spec]]       = Yk_bg[:, i_spec] / scale['var'][ind['start'] + ind['Yk_bg'][i_spec]]
                boundsLow[ind['start'] + ind['Yk_bg'][i_spec]]      =  0 / scale['var'][ind['start'] + ind['Yk_bg'][i_spec]]
                boundsUp[ind['start'] + ind['Yk_bg'][i_spec]]       =  1.0 / scale['var'][ind['start'] + ind['Yk_bg'][i_spec]]
                
                # res_2 denotes scaling when restart is activated
                scale['res_2'][ind['start'] + ind['Yk_bg'][i_spec]] = scale['res_2_Yk_bg'] * abs(GasParams['mdot_in']) 
    
    #%% For spatially resolved particle variables in the bed height and/or radius direction
    for i_var in range( len(ind['partmod_varnames']) ):
        var_name = ind['partmod_varnames'][i_var]
        
        if var_name == 'Yk_p':  # Gas Mass Fraction within the Particle
            
            # Setting SV  depends on whether the model is multi cell or single cell particle
            if SurfParams['n_p'] > 1:
                for i_p in range(SurfParams['n_p']):
                    for i_spec in range(gas['kspec'] ):
                        scale['var'][ind['start'] + int(ind['Yk_p'][(i_spec + gas['kspec'] * i_p)]) ] = 1
                        scale['res_Yk_p']                                = 1e0
                        scale['res'][ind['start'] + int(ind['Yk_p'][(i_spec + gas['kspec'] * i_p)]) ] = scale['res_Yk_p']
                        SV_guess[ind['start'] + int(ind['Yk_p'][(i_spec + gas['kspec'] * i_p)]) ] =  Yk_p[:, i_spec, i_p] / scale['var'][ind['start'] + int(ind['Yk_p'][(i_spec + gas['kspec'] * i_p)]) ]
                        boundsLow[ind['start'] + int(ind['Yk_p'][(i_spec + gas['kspec'] * i_p)]) ] =  0 / scale['var'][ind['start'] + int(ind['Yk_p'][(i_spec + gas['kspec'] * i_p)]) ]
                        boundsUp[ind['start'] + int(ind['Yk_p'][(i_spec + gas['kspec'] * i_p)]) ] =  1.0 / scale['var'][ind['start'] + int(ind['Yk_p'][(i_spec + gas['kspec'] * i_p)]) ]
            else:
                for i_spec in range(gas['kspec'] ):
                    scale['var'][ind['start'] + ind['Yk_p'][i_spec]] = 1
                    scale['res_Yk_p']                                = 1e-0
                    scale['res'][ind['start'] + ind['Yk_p'][i_spec]] = scale['res_Yk_p'] * abs(GasParams['mdot_in'])
                    SV_guess[ind['start'] + ind['Yk_p'][i_spec]] =  Yk_p[:, i_spec] / scale['var'][ind['start'] + ind['Yk_p'][i_spec]]
                    boundsLow[ind['start'] + ind['Yk_p'][i_spec]] =  0 / scale['var'][ind['start'] + ind['Yk_p'][i_spec]]
                    boundsUp[ind['start'] + ind['Yk_p'][i_spec]] =  1.0 / scale['var'][ind['start'] + ind['Yk_p'][i_spec]]
    
    
    #%% Set Bounds Vector
    varBounds = ( boundsLow, boundsUp ) 
    
    #%% Reshape SV for diagnostic
    SV_guess_reshape = SV_guess.reshape(BedParams['n_y'], ind['vars'])
    
    return SV_guess, ind, scale, varBounds

#%% Initialization function for particle model only at single height
def solarReact_Init_Part_y(gas, gas_surf, part, wall, surf, GasParams, PartParams, BedParams, WallParams, SurfParams, EnvParams, Result_y):
    
    ind = {
        'vars': ((gas['kspec'] ) * SurfParams['n_p']),
        'tot': ((gas['kspec'] ) * SurfParams['n_p']) * 1,
        'start': np.arange(1) * ((gas['kspec'] ) * SurfParams['n_p'])
    }
    
    # Define Yk_p indices
    ind['Yk_p'] = np.zeros((SurfParams['n_p']*(gas['kspec'])))
    partmod_vars_start = 0
    for i_p in range(SurfParams['n_p']):
        for i_spec in range(gas['kspec']):
                ind['Yk_p'][i_spec + (i_p * (gas['kspec'] ))] =  partmod_vars_start + i_spec
        partmod_vars_start += (gas['kspec'] )
    
    boundsLow =  np.zeros((ind['tot']))
    boundsUp = np.ones((ind['tot']))

    Bounds = ( boundsLow, boundsUp ) 
    
    Result_part = {}
    Result_part['eta_cat_k'] = np.zeros((BedParams['n_y'], gas['kspec']))
    
    # Initialize species arrays
    Yk_p_0 = np.zeros((1, gas['kspec']))
    Xk_p_0 = np.zeros((1, gas['kspec']))
    Yk_p_eq = np.zeros((1, gas['kspec']))
    Xk_p_eq = np.zeros((1, gas['kspec']))
    
    for i_p in range(SurfParams['n_p']):
        # Equilibrium concentration
        gas['obj'].TPY = Result_y['T_p'], Result_y['P_p'], Result_y['Yk_bg']
        gas['obj'].equilibrate('TP')
        Yk_p_0[0,:] = gas['obj'].Y
        Xk_p_0[0,:] = gas['obj'].X
        Yk_p_eq[0,:] = gas['obj'].Y
        Xk_p_eq[0,:] = gas['obj'].X

        Yk_p_0_lin = np.zeros((SurfParams['n_p'], gas['kspec']))  
        for i_spec in range(gas['kspec']):
            Yk_p_0_lin[:, i_spec] = np.linspace(Yk_p_eq[0,i_spec], Result_y['Yk_bg'][0, i_spec], SurfParams['n_p'])
        
        # Initialize state vector (SV_0)
        SV_guess = np.zeros(ind['tot'])
        for i_p in range(SurfParams['n_p']):
            for i_spec in range(gas['kspec'] ):
                SV_guess[ind['start'] + int(ind['Yk_p'][i_spec] + (i_p) * (ind['vars'] / SurfParams['n_p']))] = Yk_p_0[:, i_spec]
    
    return SV_guess, ind, Bounds, Xk_p_eq