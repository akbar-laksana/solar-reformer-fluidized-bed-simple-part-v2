import numpy as np

#%% Calculates the drag coefficient (beta_drag) based on different drag models: 
#    Syamlal-O'Brien, Gidaspow-Ergun-Wen-Yu, BVK, HKL.
def DragModel(PartParams, BedParams, phi_bs, phi_bg, rho_g, mu_g, v_bs, v_bg, U_hat, U_mf):
    
    # Constant
    g = 9.81     # [m/s^2]
    
    # Calculate Reynolds number
    Re_p = PartParams['dp'] * abs(v_bs - v_bg) * rho_g / mu_g

    # Initialize drag coefficient array
    beta_drag = np.zeros(len(phi_bs))
    H2 = np.zeros(len(phi_bs))
    
    # Calculate uncorrected beta_drag
    # Syamlal-O'Brien Drag Model
    if PartParams['drag_model'] == 'Syamlal-OBrien':
        A_term                  = phi_bg**4.14
        B_term                  = 0.8 * phi_bg**1.28
        B_term[phi_bg > 0.85]   = phi_bg[phi_bg > 0.85]**2.65
        v_term                  = (A_term - 0.06 * Re_p + np.sqrt((0.06 * Re_p)**2 + 0.12 * Re_p * (2 * B_term - A_term) + A_term**2)) / 2 # terminal velocity
        C_drag                  = (0.63 + 4.8 / np.sqrt(Re_p / v_term))**2 # drag coefficient
        beta_drag_Syamlal       = (3 * phi_bs * phi_bg * rho_g / (4 * v_term**2 * PartParams['dp'])) * C_drag * abs(v_bs - v_bg)
        beta_drag_Ergun         = (150 * ((phi_bs**2) * mu_g) / (phi_bg*PartParams['dp']**2)) + 1.75 * (phi_bs * rho_g) * (abs(v_bs - v_bg)) / (PartParams['dp']) # dense regime
        
        # Smooth blending function between two beta_drag models
        psi_drag                = 0.5 + ((np.arctan(150*1.75*(0.55 - phi_bs)))/np.pi)
        beta_drag               = (1 - psi_drag) * (beta_drag_Ergun) + psi_drag * (beta_drag_Syamlal)
    
    elif PartParams['drag_model'] == 'Ergun': # packed-bed expression, valid for phi_bg < 0.8
        beta_drag = 150 * mu_g *(phi_bs**2) / (phi_bg * PartParams['dp']**2) + 1.75 * rho_g *phi_bs * abs(v_bs - v_bg) / PartParams['dp']
    
    # Gidaspow-Ergun-Wen-Yu Drag Model
    elif PartParams['drag_model'] == 'Gidaspow':
        C_drag2 = np.full_like(Re_p, 0.44)
        mask = Re_p < 1000
        C_drag2[mask] = (24 / (Re_p[mask] * phi_bg[mask])) * (1 + 0.15 * (phi_bg[mask] * Re_p[mask]) ** 0.687)
        
        beta_drag_Ergun = np.zeros_like(phi_bs)
        i_beta = phi_bg < 0.8
        beta_drag_Ergun[i_beta] = (150 * (phi_bs[i_beta] ** 2) * mu_g[i_beta] / (phi_bg[i_beta] * PartParams['dp'] ** 2)) + \
                                  1.75 * (phi_bs[i_beta] * rho_g[i_beta]) * (np.abs(v_bs[i_beta] - v_bg[i_beta])) / PartParams['dp']
        
        beta_drag_Wen_Yu = np.zeros_like(phi_bs)
        i_beta = phi_bg >= 0.8
        beta_drag_Wen_Yu[i_beta] = 0.75 * C_drag2[i_beta] * (phi_bs[i_beta] * phi_bg[i_beta] * rho_g[i_beta]) * \
                                   np.abs(v_bs[i_beta] - v_bg[i_beta]) * (phi_bg[i_beta] ** -2.65)
        
        psi_drag = 0.5 + (np.arctan(150 * 1.75 * (0.2 - phi_bs)) / np.pi)
        beta_drag = (1 - psi_drag) * beta_drag_Ergun + psi_drag * beta_drag_Wen_Yu
    
    # BVK Drag Model
    elif PartParams['drag_model'] == 'BVK':       
        d_avg = PartParams['dp'] # Assuming monodisperse system
        y_m = PartParams['dp'] / d_avg # Size ratio (monodisperse = 1)

        F_phi_bs_Re = (10 * phi_bs / (1 - phi_bs)**2) + ((1 - phi_bs)**2 * (1 + 1.5 / np.sqrt(phi_bs))) + \
            (0.413 * Re_p / 24 * (1 - phi_bs**2)) * ((1 / phi_bg) + 3 * phi_bg * phi_bs + \
            8.4 * Re_p**-0.343 / (1 + 10**3 * phi_bs * Re_p**-0.5) ** 2)

        F_gm = (18 * mu_g * phi_bg / PartParams['dp']**2) * ((1 - phi_bs) * y_m + phi_bs * y_m**2) + \
            0.064 * (1 - phi_bs) * y_m**3 * F_phi_bs_Re
        beta_drag = F_gm * rho_g / phi_bg
    
    # Tenneti et al. Model
    elif PartParams['drag_model'] == 'Tenneti':     
        K1 = 1 + 0.15 * Re_p**0.687
        K2 = 5.81 * phi_bs / phi_bg**3 + 0.48 * (phi_bs**0.3 / phi_bg**4)
        K3 = phi_bs**3 * Re_p * (0.95 + 0.61 * phi_bs**3 / phi_bg**2)
        F0 = K1 / phi_bg**3 + K2 + K3
        F0 = np.real(F0)
        beta_drag = (18 * mu_g * phi_bg**2 * phi_bs * F0) / (PartParams['dp']**2)
    
    
    #%% Calculate beta_drag correction factor H2
    # phi star
    if PartParams['drag_corrector'] == 'phi_star':
        H2 = (PartParams['phi_bs_max'] - phi_bs) / PartParams['phi_bs_max']     # F_bs -> beta when not fluidized, F_bs -> 0 with lots of bubbles
    
    # Sarkar et al., Chem. Eng. Sci. (2016)
    elif PartParams['drag_corrector'] == 'sarkar':
        # Fit for a specific case in Sarkar 2016 for del_f_st {1.36-6.74}
        u_slip_st = abs((v_bs - v_bg) / v_term)                                 # [-] dimensionless slip velocity
        del_f_st = g * BedParams['dVol']**(1/3) / U_mf**2
        Hval = (0.9506 + 0.1708 / u_slip_st) * phi_bs**(0.049 * (1 / del_f_st - 1) + 0.3358 / u_slip_st)
        H2 = min(Hval, 0.97)
    
    # Milioli et al., Powder Technol. (2021)
    elif PartParams['drag_corrector'] == 'milioli':
        # Specific case in Milioli 2013 for del_f_st {1.36-6.74}
        u_slip_st = abs((v_bs - v_bg) / U_mf)                                   # [-] dimensionless slip velocity
        f_inf = 0.882 * (2.145 - 7.8 * u_slip_st**1.8 / (7.746 * u_slip_st**1.8 + 0.5586))
        h_1 = (1.6 * u_slip_st + 4) / (7.9 * u_slip_st + 0.08) * phi_bs + 0.9394 - 0.22 / (0.6 * u_slip_st + 0.01)
        h_lin = h_1 * f_inf
        h_lin[h_1 < 0] = 0  # Set negative h_lin values to zero

        h_env = 0.8428 + 0.6393 * phi_bs - 0.6743 * phi_bs**2
        mask = phi_bs > 0.54
        h_env[mask] = 0.4099 * (0.65 - phi_bs[mask])**0.25 / (phi_bs[mask]**(-0.25) - 0.9281)

        H2 = np.minimum(h_env, h_lin)

    elif PartParams['drag_corrector'] == 'none':
        H2[:] = 0

    H2[U_hat <0] = 0
    
    #  Add in correction factor for drag coefficient
    beta_drag = (1 - H2)*beta_drag
    
    return beta_drag