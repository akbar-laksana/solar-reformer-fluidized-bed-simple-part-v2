import numpy as np

#%% Calculates the fluidized bed-wall convective heat transfer coefficient
def hT_wb_conv(U_hat, Ar_lam, Pr_gs, dp, k_g):   
    # Initialize the laminar Archimedes number heat transfer function
    f_Ar_lam = np.zeros(len(Ar_lam))
    
    # Compute f_Ar_lam based on Ar_lam values
    for j_y in range(len(Ar_lam)):
        if Ar_lam[j_y] < 1500:
            f_Ar_lam[j_y] = 0.129 * Ar_lam[j_y]**0.594
        else:
            f_Ar_lam[j_y] = 2.089 * Ar_lam[j_y]**0.1743
    
    # U_hat function for particle-wall heat transfer
    f_U_hat = 0.241 + 0.043 * np.maximum(0, U_hat)**0.905 * np.exp(-U_hat / 71.673)
    
    # Calculate Nusselt number for particle-wall heat transfer
    Nu_d = (f_U_hat * f_Ar_lam) / (1 + Pr_gs)
    
    # Calculate particle-wall heat transfer coefficient
    hT_wb_c = Nu_d * k_g / dp
    
    return hT_wb_c
