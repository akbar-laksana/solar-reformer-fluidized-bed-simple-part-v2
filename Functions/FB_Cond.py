import numpy as np

#%% Set function for calculating effective solids conductivity (Syamlal and Gidaspow 1985) %%
def FB_Cond(phi_bg, R):
    contact     = 7.26e-3
    B           = 1.25*((1 - phi_bg)/phi_bg)**(1.1111)
    k_ratio     = ( (B/R)*(R - 1)*(1 - B/R)**(-2)*np.log(R/B) - (B - 1)/(1 - B/R) - 0.5*(B + 1) )*2/(1 - B/R)
    cond_ratio  = (1 - np.sqrt(1 - phi_bg)) + np.sqrt(1 - phi_bg)*( contact*R + (1 - contact)*k_ratio)
    return cond_ratio
