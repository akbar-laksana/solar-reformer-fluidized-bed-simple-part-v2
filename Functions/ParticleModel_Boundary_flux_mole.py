import numpy as np
import cantera as ct
#from cantera import Solution

# Define function to calculate fluxes at the outer particle boundary (integrated with Reactor Model)
def ParticleModel_Boundary_flux_mole(gas, PartParams, P_1, Yk_1, T_1, P_2, Yk_2, T_2, U_inf):
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
    b['Xk_b'] = gas['obj'].X

    # Reynolds number
    Re = b['rho_b'] * U_inf * PartParams['dp'] / mu_b

    # Schmidt number
    Sc = (mu_b / (b['rho_b'] * D_k_b))
    
    # Sherwood number correlation
    Sh_k = 2 + 0.6 * Re**0.5 * Sc**(1/3)
    
    """
    # Alternative Sherwood correlation following Nu_p form
    Nu_p = 2 + (0.4 * Re**0.5 + 0.06 * Re**0.667)
    Sh_k = Nu_p
    """

    # Mass flux at the boundary
    jk_b = b['rho_b'] * Sh_k * D_k_b * (Yk_1 - Yk_2) / PartParams['dp']

    # Corrective flux scaling to Yk_b
    jk_corr = np.sum(jk_b) * b['Yk_b']
    #jk_corr = np.sum(jk_b)/gas['kspec'] * np.ones((gas['kspec']))
    jk_b -= jk_corr

    # Calculate sum of species fluxes weighted by specific heat capacities
    sumjkcp_b = np.sum(jk_b * c_p_b)

    return jk_b, sumjkcp_b, b