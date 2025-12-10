import numpy as np

#%% Function to calculate gas properties
def solarReact_GasProps(gas, n_y, P_bg, T_bg, Yk_bg, T_bs):

    rho_g   = np.zeros(n_y)             # density of fluidizing gas [kg/m^3]
    h_g     = np.zeros(n_y)             # enthalpy of fluidizing gas [J/kg]
    mu_g    = np.zeros(n_y)             # viscosity of fluidizing gas [Pa-s]
    cp_g    = np.zeros(n_y)             # specific heat capacity of fluidizing gas [J/kg-K]
    k_g     = np.zeros(n_y)             # thermal conductivity of fluidizing gas [W/m-K]
    
    Dk_mix          = np.zeros((n_y, gas['kspec'])) # diffusivity of gas phase species [m^2/s]
    hk_g            = np.zeros((n_y, gas['kspec'])) # species specific enthalpies per unit mass at bulk gas conditions [kJ/kg of species k]
    hk_sg           = np.zeros((n_y, gas['kspec'])) # species specific enthalpies per unit mass at particle gas conditions [kJ/kg of species k]
    
    #%% Calculate gas thermophysical and transport properties along height of channel
    for i_y in range(n_y):
        # Get gas properties
        gas['obj'].TPY  = T_bg[i_y], P_bg[i_y], Yk_bg[i_y,:]         # Define gas thermodynamic state
        rho_g[i_y]      = gas['obj'].density                        # Density of fluidizing gas [kg/m^3]
        k_g[i_y]        = gas['obj'].thermal_conductivity           # Thermal conductiviy of fluidizing gas [W/m-K]
        mu_g[i_y]       = gas['obj'].viscosity                      # Viscosity of fluidizing gas [Pa-s]
        cp_g[i_y]       = gas['obj'].cp_mass                        # Specific heat capacity of fluidizing gas [J/kg-K]
        h_g[i_y]        = gas['obj'].enthalpy_mass
        hk_g[i_y,:]     = gas['obj'].partial_molar_enthalpies/gas['Wk']
        Dk_mix[i_y,:]   = gas['obj'].mix_diff_coeffs
    
        # Set gas phaseenthalpy at the particle condition
        gas['obj'].TPY  = T_bs[i_y], P_bg[i_y], Yk_bg[i_y,:]        # Define gas thermodynamic state
        hk_sg[i_y,:]    = gas['obj'].partial_molar_enthalpies/gas['Wk'] # Enthalpies of masses leaving at solid temp
    
    return rho_g, h_g, mu_g, k_g, cp_g, Dk_mix, hk_g, hk_sg