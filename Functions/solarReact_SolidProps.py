import numpy as np

#%% Function to calculate gas properties
def solarReact_SolidProps(part, SurfParams, n_y, P_bg, T_b):

    rho_s           = np.zeros(n_y)             # density of porous particles [kg/m^3 total in particle]
    h_s             = np.zeros(n_y)             # enthalpy of solid particles [J/kg] 
    cp_s            = np.zeros(n_y)             # specific heat capacity of solid particles [J/kg-K] 
    
    #%% Calculate gas thermophysical and transport properties along height of channel
    for i_y in range(n_y):
        # Particle Properties
        part['obj'].TP = T_b[i_y], P_bg[i_y]                                   # Define particle thermodynamic state
        rho_s[i_y]      = part['obj'].density*(1-SurfParams['phi'])       # Bulk density of solid particles [kg of solid/m^3 total particle volume]
        cp_s[i_y]       = part['obj'].cp_mass                             # Specific heat capacity of solid [J/kg-K]
        h_s[i_y]        = part['obj'].enthalpy_mass                       # Enthalpies of solid [kJ/kg of k]
        
    return rho_s, h_s, cp_s
