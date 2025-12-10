import numpy as np
import cantera as ct
#from cantera import Solution

# Define function to calculate fluxes at the outer particle boundary (integrated with Reactor Model)
def ParticleModel_Boundary_flux(gas, PartParams, P_1, Yk_1, T_1, P_2, Yk_2, T_2, U_inf, phi_bg):
    # Average properties at the boundary
    b = {
        'P_b': 0.5 * (P_1 + P_2),
        'Yk_b': 0.5 * (Yk_1 + Yk_2),
        'T_b': 0.5 * (T_1 + T_2),
    }
    
    # Set the gas properties at boundary conditions
    gas['obj'].TPY = b['T_b'], b['P_b'], b['Yk_b']
    b['rho_b'] = gas['obj'].density
    mu_b = gas['obj'].viscosity                      # Gas viscosity [Pa s]
    D_k_b = gas['obj'].mix_diff_coeffs               # Diffusion coefficient [m^2/s]
    c_p_b = gas['obj'].cp_mass                       # Heat capacity [J/kg-K]
    k_b   = gas['obj'].thermal_conductivity
    b['Xk_b'] = gas['obj'].X
    
    # Reynolds number
    b['Re'] = b['rho_b'] * U_inf * PartParams['dp'] / mu_b

    # Schmidt number
    b['Sc'] = (mu_b / (b['rho_b'] * D_k_b))
    
    # Sherwood number correlation
    if PartParams['Sherwood'] == 'Zhang':
        # Zhang 2024
        b['Sh_k'] = 2 + 0.6 * b['Re']**0.5 * b['Sc']**(1/3)
    elif PartParams['Sherwood'] == 'Frossling':
        # Frossling 1938
        b['Sh_k'] = 2 + 0.69 * b['Re']**0.5 * b['Sc']**(1/3)
    elif PartParams['Sherwood'] == 'Gunn':
        # Gunn 1978
        b['Sh_k'] = (7 - 10*phi_bg + 5*phi_bg**2)*(1 + 0.7*b['Re']**0.2 * b['Sc']**(1/3)) \
        + (1.33 - 2.4*phi_bg + 1.2*phi_bg**2)*(b['Re']**0.7 * b['Sc']**(1/3))
    
    # Give multiplication factor to Sherwood, or override
    b['Sh_k'] = 1e0 * b['Sh_k']
    
    # Prandtl number
    b['Pr'] = mu_b * c_p_b / k_b
    
    # Mass flux at the boundary
    jk_b = b['rho_b'] * b['Sh_k'] * D_k_b * (Yk_1 - Yk_2) / PartParams['dp']
    
    # Constant Sh of 3.5 (DeCaluwe et al)
    #jk_b = b['rho_b'] * 3.5 * D_k_b * (Yk_1 - Yk_2) / PartParams['dp']
    
    # Corrective flux scaling to Yk_b
    jk_corr = np.sum(jk_b) * b['Yk_b']
    #jk_corr = np.sum(jk_b)/gas['kspec'] * np.ones((gas['kspec']))
    jk_b -= jk_corr
    
    # Do checking on sum of jk_b to see how mass neutral it is
    sumjk_b = np.sum(jk_b)
    
    # Calculate sum of species fluxes weighted by specific heat capacities
    sumjkcp_b = np.sum(jk_b * c_p_b)

    return jk_b, sumjkcp_b, b