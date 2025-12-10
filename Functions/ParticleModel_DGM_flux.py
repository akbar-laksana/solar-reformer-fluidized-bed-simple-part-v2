import numpy as np
import cantera as ct

# Define Dusty Gas Model flux function (integrated with Reactor Model)
def ParticleModel_DGM_flux(gas, SurfParams, rface, r_1, P_1, Xk_1, T_1, r_2, P_2, Xk_2, T_2):
    # Find midpoint values at r_face with linear interpolation
    wt_1 = (r_2 - rface) / (r_2 - r_1)
    wt_2 = (rface - r_1) / (r_2 - r_1)
    P_face = P_1 * wt_1 + P_2 * wt_2
    Xk_face = Xk_1 * wt_1 + Xk_2 * wt_2
    T_face = T_1 * wt_1 + T_2 * wt_2
    gas['obj'].TPX = T_face, P_face, Xk_face
    mu_face = gas['obj'].viscosity  # gas viscosity [Pa s]
    
    # Molar concentrations
    C_1 = (P_1 * Xk_1) / (ct.gas_constant * T_1)   # [kmol/m^3]
    C_2 = (P_2 * Xk_2) / (ct.gas_constant * T_2)   # [kmol/m^3]

    # Gas binary diffusion coefficients
    DiffBin = gas['obj'].binary_diff_coeffs * SurfParams['phi'] / SurfParams['tau']   # [m^2/s]
    
    # Kozeny-Carman permeability
    perm = SurfParams['B_g']    # permeability [m^2]
    
    # Effective Knudsen diffusion coefficient [m^2/s]
    DiffKnu = (2 / 3) * (SurfParams['phi'] / SurfParams['tau']) * SurfParams['Rpore'] * \
              np.sqrt(8 * ct.gas_constant * T_face / (np.pi * gas['Wk']))
    
    # Initialize DiffDGM matrix
    DiffDGM = np.zeros((gas['kspec'], gas['kspec']))
    
    # Constructing the h matrix for Dusty Gas Model
    for i in range(gas['kspec']):
        DiffDGM[i, i] = 1 / DiffKnu[i]
        for j in range(gas['kspec']):
            if i != j:
                DiffDGM[i, j] = -Xk_face[i] / DiffBin[i, j]
                DiffDGM[i, i] += Xk_face[j] / DiffBin[i, j]
    
    # Invert DiffDGM matrix
    DiffDGM = np.linalg.inv(DiffDGM)
    
    # Calculate flux from concentration term
    FluxConc = -np.dot(DiffDGM, (C_2 - C_1) / (r_2 - r_1))

    # Calculate flux from pressure term
    Velocity = -(perm / mu_face) * (P_2 - P_1) / (r_2 - r_1)
    CU = C_1 if Velocity > 0 else C_2
    FluxPres = np.sum(DiffDGM * CU / DiffKnu[:, np.newaxis], axis=1) * Velocity

    # Total species flux J_k [kmol/m^2 s]
    J_k = FluxConc + FluxPres
    
    # Convert to mass flux [kg/m^2 s]
    jk_face = gas['Wk'] * J_k

    # Heat flux term (sumjkcp_face) [J/m^2 Ks]
    sumjkcp_face = np.sum(jk_face * gas['obj'].cp_mass)
    
    return jk_face, sumjkcp_face