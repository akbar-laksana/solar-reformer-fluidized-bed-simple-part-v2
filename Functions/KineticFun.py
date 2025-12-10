import numpy as np

#%% 4 Main reaction considered:
"""
DMR:
CH4 + CO2 ↔ 2CO + 2H2

SMR1:
CH4 + H2O ↔ CO + 3H2
	​
SMR2:
CH4 + 2H2O ↔ CO2 + 4H2
	​
WGS:
CO + H2O ↔ CO2 + H2

"""
#%% Define function that calculate the kinetics
#   Kinetic data from the work of Park et al, Fuel 115 (2014) 357–365
#   https://doi.org/10.1016/j.fuel.2013.07.034
#   and
#   Kinetic data from the work of Jun et al, J of Natrl Gas Chem (2011) Vol. 20 No. 1 
#   https://doi.org/10.1016/S1003-9953(10)60148-X

#%% Kinetic Function
def KineticFun(ind, gas, Xk_p, T_p, P_p, SurfParams):
    R       = 8.314     # [J/mol K]
    T_ref   = 1123.15   # [K] 
    
    # Number of reactions j and indices
    n_reac  = gas['jreac']
    j_DMR   = gas['jDMR']
    j_SMR1  = gas['jSMR1'] 
    j_SMR2  = gas['jSMR2'] 
    j_WGS   = gas['jWGS']
    
    kinetics = 2   # 1: Park, 2: Jun
    # NOTE: Kinetics 2 by Jun et al works/converges better as of now
    
    if kinetics == 1:
        # Calculate partial pressure of gas species
        P_CH4   = P_p * Xk_p[gas['kCH4']]
        P_CO2   = P_p * Xk_p[gas['kCO2']]
        P_CO    = P_p * Xk_p[gas['kCO']]
        P_H2    = P_p * Xk_p[gas['kH2']]
        P_H2O   = P_p * Xk_p[gas['kH2O']]
        
        # Reaction rate constant
        k_SMR1  = 4.72e6 * np.exp((-232477/R)*(1/T_p - 1/T_ref))    # [Mol Pa^0.5 / g_cat h]
        k_SMR2  = 1.89e3 * np.exp((-267760/R)*(1/T_p - 1/T_ref))    # [Mol Pa^0.5 / g_cat h]
        k_DMR   = 2.91e-7 * np.exp((-234851/R)*(1/T_p - 1/T_ref))    # [Mol / g_cat h Pa^2]
        k_WGS   = 1.06e-3 * np.exp((-71537/R)*(1/T_p - 1/T_ref))   # [Mol / g_cat h Pa]
    
        # Adsorption equilibrium constant
        K_CO2   = 5.97e-7 * np.exp(52670 / (R*T_p))     # [Pa^-1]
        K_CO    = 8.23e-10 * np.exp(70650 / (R*T_p))    # [Pa^-1]
        K_H2    = 6.12e-14 * np.exp(82900 / (R*T_p))    # [Pa^-1]
        K_CH4   = 6.65e-9 * np.exp(38280 / (R*T_p))     # [Pa^-1]
        K_H2O   = 1.77e5 * np.exp(-88680 / (R*T_p))     # [-]
    
        # Reaction equilibrium constant (K_p)
        K_p_WGS     = np.exp(-12.11 + 5318.69/T_p + 1.01*np.log(T_p) + 1.14e-4*T_p)
        K_p_SMR1    = np.exp(2.48 - 22920.6/T_p + 7.19*np.log(T_p) - 2.95e-3*T_p)
        K_p_DMR     = K_p_SMR1/K_p_WGS
        K_p_SMR2    = K_p_SMR1*K_p_WGS
    
        # Reaction rate (r) of the global reactions considered
        r_DMR = ( k_DMR*(P_CH4 * P_CO2 - P_H2**2 * P_CO**2 / K_p_DMR) ) / \
            ( (1 + K_CH4*P_CH4 + K_CO*P_CO) * (1 + K_CO2*P_CO2) )
    
        r_SMR1 = ( k_SMR1 * (P_CH4 * P_H2O - P_H2**3 * P_CO / K_p_SMR1)/P_H2**2.5 ) / \
            ( (1 + K_CO*P_CO + K_H2*P_H2 + K_CH4*P_CH4 + K_H2O*(P_H2O/P_H2))**2 )
    
        r_SMR2 = ( k_SMR2 * (P_CH4*P_H2O**2 - P_H2**4 * P_CO2 / K_p_SMR2)/P_H2**3.5 ) / \
            ( (1 + K_CO*P_CO + K_H2*P_H2 + K_CH4*P_CH4 + K_H2O*(P_H2O/P_H2))**2 )
        
        r_WGS = ( k_WGS * (P_CO*P_H2O - P_H2 * P_CO2 / K_p_WGS)/P_H2 ) / \
            ( (1 + K_CO*P_CO + K_H2*P_H2 + K_CH4*P_CH4 + K_H2O*(P_H2O/P_H2))**2 )
    
    elif kinetics == 2:
        # Calculate partial pressure of gas species
        P_CH4   = P_p * Xk_p[gas['kCH4']] / 1e5
        P_CO2   = P_p * Xk_p[gas['kCO2']] / 1e5
        P_CO    = P_p * Xk_p[gas['kCO']] / 1e5
        P_H2    = P_p * Xk_p[gas['kH2']] / 1e5
        P_H2O   = P_p * Xk_p[gas['kH2O']] / 1e5
        
        # Reaction rate constant
        k_SMR1  = 1.5e3 * np.exp((-237506.27/R)*(1/T_p - 1/T_ref))      # [Mol / g_cat h bar]
        k_SMR2  = 2.07e2 * np.exp((-266073.80/R)*(1/T_p - 1/T_ref))     # [Mol bar ^0.5 / g_cat h]
        k_DMR   = 2.79e3 * np.exp((-225613.58/R)*(1/T_p - 1/T_ref))     # [Mol / g_cat h atm^2]
        k_WGS   = 5.71e0 * np.exp((-81394.60/R)*(1/T_p - 1/T_ref))      # [Mol bar ^0.5 / g_cat h]
        
        # Adsorption equilibrium constant
        K_1     = 0.5   # [bar^-1]
        K_2     = 9.71  # [bar^-1]
        K_3     = 26.21 # [bar^-1]
        K_CO    = 8.23e-5 * np.exp(-70650 / (R*T_p))    # [bar^-1]
        K_H2    = 6.12e-9 * np.exp(-82900 / (R*T_p))    # [bar^-1]
        K_CH4   = 6.65e-4 * np.exp(-38280 / (R*T_p))    # [bar^-1]
        K_H2O   = 1.77e-5 * np.exp(-88680 / (R*T_p))    # [-]
        
        # Reaction equilibrium constant (K_p)
        K_p_SMR1    = np.exp(29.71 - 2.62e4/T_p)
        K_p_SMR2    = np.exp(27.40 - 2.34e4/T_p)
        K_p_WGS     = K_p_SMR2/K_p_SMR1
        K_p_DMR     = K_p_SMR1/K_p_WGS
        
        # Reaction rate (r) of the global reactions considered
        r_DMR = ( k_DMR*(P_CH4 * P_CO2 - P_H2**2 * P_CO**2 / K_p_DMR) ) / \
            ( (1 + K_1*P_CH4 + K_2*P_CO) * (1 + K_3*P_CO2) )
    
        r_SMR1 = ( k_SMR1 * (P_CH4 * P_H2O - P_H2**3 * P_CO / K_p_SMR1)/P_H2**2.5) / \
            ( (1 + K_CO*P_CO + K_H2*P_H2 + K_CH4*P_CH4 + K_H2O*(P_H2O/P_H2))**2 )
    
        r_SMR2 = ( k_SMR2 * (P_CH4 * P_H2O**2 - P_H2**4 * P_CO2 / K_p_SMR2)/P_H2**3.5) / \
            ( (1 + K_CO*P_CO + K_H2*P_H2 + K_CH4*P_CH4 + K_H2O*(P_H2O/P_H2))**2 )
        
        r_WGS = ( k_WGS * (P_CO * P_H2O - P_H2 * P_CO2 / K_p_WGS)/P_H2) / \
             ( (1 + K_CO*P_CO + K_H2*P_H2 + K_CH4*P_CH4 + K_H2O*(P_H2O/P_H2))**2 )
    
    # Production rate of species [mol/g_cat/h], coefficient refers to reaction above
    R_CH4 = -r_DMR - r_SMR1 - r_SMR2
    R_CO2 = -r_DMR + r_WGS + r_SMR2
    R_CO  = +2*r_DMR + r_SMR1 - r_WGS
    R_H2  = +2*r_DMR + 3*r_SMR1 + 4*r_SMR2 + r_WGS
    R_H2O = -r_SMR1 - 2*r_SMR2 - r_WGS
    
    # Put into array in gas species order
    R_g = np.zeros(gas['kspec'])
    R_g[gas['kCH4']] = R_CH4
    R_g[gas['kCO2']] = R_CO2
    R_g[gas['kCO']]  = R_CO
    R_g[gas['kH2']]  = R_H2
    R_g[gas['kH2O']] = R_H2O

    # Convert to kmol/m^2/s
    conv = 1 / ( 1000.0 * 3600)  # mol/g_cat-h → kmol/g_cat-s
    sdot_g = R_g * conv * SurfParams['mcat_per_area'] # mcat_per_area = [g_cat / m^2]
    
    # Scaling factor for the production rate since the global mechanism is lower by O(2) compared to surface chem
    sdot_g = 1e-0 * sdot_g
    
    # Reaction rates [Mol / g_cat h]
    R_j = np.zeros(n_reac)
    R_j[j_DMR]  = r_DMR * conv * SurfParams['mcat_per_area']
    R_j[j_SMR1] = r_SMR1 * conv * SurfParams['mcat_per_area']
    R_j[j_SMR2] = r_SMR2 * conv * SurfParams['mcat_per_area']
    R_j[j_WGS]  = r_WGS * conv * SurfParams['mcat_per_area']
    
    return sdot_g, R_j