import numpy as np

#%% Calculates the fluidized bed-wall convective heat transfer coefficient
def hT_wb_conv_v2(U_hat, Ar_lam, Pr_gs, dp, k_g):   

    # This function calculates fluidized bed-wall convective heat transfer coefficeint

    # Calculate bed-wall convective Nusselt number correlation based on laminar Archimedes number
    f_Ar_lam    = np.zeros(len(Ar_lam))       # laminar Archemedes number heat transfer function [--]
    i_Ar_1      = np.argwhere(Ar_lam < 1480)
    i_Ar_m      = np.argwhere((Ar_lam >= 1480) & (Ar_lam <= 1520))
    i_Ar_2      = np.argwhere(Ar_lam > 1520)
    f_Ar_lam_1  = 0.129 * Ar_lam**0.594
    f_Ar_lam_2  = 2.089 * Ar_lam**0.1743
    f_Ar_lam[i_Ar_1] = f_Ar_lam_1[i_Ar_1]
    f_Ar_lam[i_Ar_m] = f_Ar_lam_1[i_Ar_m] * (1520 - Ar_lam[i_Ar_m]) / 40 \
        + f_Ar_lam_2[i_Ar_m]*(Ar_lam[i_Ar_m] - 1480)/40
    f_Ar_lam[i_Ar_2] = f_Ar_lam_2[i_Ar_2]

    f_U_hat   = 0.241 + 0.043* np.maximum(0, U_hat)**0.905 * np.exp(-U_hat / 71.673) # [-], Uhat function for particle-wall heat transfer
    Nu_d      = ( f_U_hat * f_Ar_lam ) / (1 + Pr_gs)                     # [-], Molerus form Nusselt number correlation for particle-wall heat transfer

    # Calculate particle-wall heat transfer coefficient with convective and radiative components %%
    hT_wb_c  = Nu_d * k_g / dp              

    return hT_wb_c