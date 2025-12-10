import numpy as np
import os
import sys

functions_folder = os.path.join(".", 'Functions')
sys.path.append(functions_folder)

from Thermal_Cond import Thermal_Cond

#%%  This function calcultions the fin effectiveness eta and the array
# Efficiency eta_o for an a fin with an adiabatic boundary
# Single Fin Effectiveness (assuming rectangular fin with adiabatic tip

def FinCalcs(T, hT_wb , FinParams): 

    k_w     =   Thermal_Cond(FinParams['mat'], T)
    m       =   np.sqrt( hT_wb * FinParams['Perim']  / (k_w * FinParams['Az_cond'] ) )
    ml      =   m * FinParams['l_cond']
    eta     =   np.tanh(ml)/ml

    # Fin Array Effectiveness
    eta_o   =   1 - (1 - eta) * FinParams['A_fin_o_A_tot']
    F_fin   =   eta_o * FinParams['A_tot_o_A_base']

    return eta, eta_o, F_fin