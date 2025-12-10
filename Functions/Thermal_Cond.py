import numpy as np

#%% Calculate the solid thermal conductivity
def Thermal_Cond(species, T):

    if species in ['SiO2', 'Silica']:
        # 3rd order inverse polynomial coefficients for silica/quartz glass thermal conductivity
        a = np.array([2.1704E+00, -3.0025E+02, 2.0412E+04, -5.2263E+05])
        lambda_ = a[0] + a[1] * T**(-1) + a[2] * T**(-2) + a[3] * T**(-3)  # [W/m-K]

    elif species in ['AlSiFeOx', 'CARBO_HSP']:
        # Constant thermal conductivity for alumina-silica particles
        a = np.array([2.0000E+00])
        lambda_ = a[0] + 0 * T  # [W/m-K]

    #elif species == 'CARBO_HSP_BULK' or species == 'CARBO_CP':
    elif species in['CARBO_HSP_BULK', 'CARBO_CP', 'Alumina'] :
        # Thermal conductivity with linear temperature dependence
        a = np.array([0.0003824, 0.29])
        lambda_ = a[0] * T + a[1]  # [W/m-K]

    elif species == 'SS_bulk':
        # 2nd order polynomial for 304SS plate thermal conductivity
        a = np.array([7.9318E+00, 2.3051E-02, -6.4166E-06])
        lambda_ = a[0] + a[1] * T + a[2] * T**2  # [W/m-K]

    elif species == 'Inco_718':
        # 4th order polynomial for Inco 718 thermal conductivity
        a = np.array([6.9028E+00, 1.5000E-02, 1.6357E-16, 7.2302E-21, -6.6174E-24])
        lambda_ = a[0] + a[1] * T + a[2] * T**2 + a[3] * T**3 + a[4] * T**4  # [W/m-K]
    
    elif species == 'Inco470H':
        # 4th order polynomial for Inco 718 thermal conductivity
        a = np.array([6.9028E+00, 1.5000E-02, 1.6357E-16, 7.2302E-21, -6.6174E-24])
        lambda_ = a[0] + a[1] * T + a[2] * T**2 + a[3] * T**3 + a[4] * T**4  # [W/m-K]
    
    elif species == 'Regolith':
        # Polynomial for regolith thermal conductivity, with constant T2=400K
        T2 = np.full_like(T, 400)
        a = np.array([1.43E-03, 0.197E-10])
        lambda_ = a[0] + a[1] * T2**3  # [W/m-K]
    
    elif species == 'silicon-carbide':   # R.P. Joshi, P.G. Neudeck, and C. Fazi, J. Appl. Phys., 88(1) 265-269
        a = 4.517E+05
        n = -1.29
        T_corr = np.maximum(300*np.ones(len(T)), T)  # temperatures [K]
        lambda_  = a * T_corr**n  

    return lambda_
