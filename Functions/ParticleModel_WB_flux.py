import numpy as np
import cantera as ct

def ParticleModel_WB_flux(gas, SurfParams, rface, r_1, P_1, Xk_1, T_1, r_2, P_2, Xk_2, T_2):
    # Linear interpolation for face conditions
    wt_1 = (r_2 - rface) / (r_2 - r_1)
    wt_2 = (rface - r_1) / (r_2 - r_1)
    P_face = P_1 * wt_1 + P_2 * wt_2
    Xk_face = Xk_1 * wt_1 + Xk_2 * wt_2
    T_face = T_1 * wt_1 + T_2 * wt_2
    gas['obj'].TPX = T_face, P_face, Xk_face

    # Total concentration at faces [kmol/m^3]
    C_1 = P_1 / (ct.gas_constant * T_1)
    C_2 = P_2 / (ct.gas_constant * T_2)
    
    # Binary diffusivities [m^2/s] with porosity/tortuosity correction
    DiffBin = gas['obj'].binary_diff_coeffs
    DiffBin_eff = SurfParams['phi']/SurfParams['tau'] * DiffBin

    # Knudsen diffusivity for each species [m^2/s]
    DiffKnu = (2/3) * (SurfParams['phi']/SurfParams['tau']) * SurfParams['Rpore'] * \
              np.sqrt(8*ct.gas_constant*T_face/(np.pi*gas['Wk']))

    # Wilke–Bosanquet effective diffusivity for each species
    DiffWB = np.zeros(gas['kspec'])
    for i in range(gas['kspec']):
        # Effective binary diffusivity for species i
        denom = 0.0
        for j in range(gas['kspec']):
            if i != j:
                denom += Xk_face[j]/DiffBin_eff[i,j]
        Dbin_i = 1.0/denom
        DiffWB[i] = 1.0 / (1.0/Dbin_i + 1.0/DiffKnu[i])

    # Fluxes [kmol/m2/s], Fick’s law with effective diffusivity
    dXi = Xk_2 - Xk_1
    dC = C_2 - C_1
    J_k = -(DiffWB * ( (C_2*Xk_2 - C_1*Xk_1) ))/(r_2 - r_1)

    # Convert to mass flux [kg/m2/s]
    jk_face = gas['Wk'] * J_k

    # Heat flux (approx cp_mass-weighted convective flux)
    sumjkcp_face = np.sum(jk_face * gas['obj'].cp_mass)

    return jk_face, sumjkcp_face