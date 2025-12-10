import numpy as np

def D_T(ug, umf, D_h, species):
    u_excess = ug - umf
    dt = D_h / umf

    if species == 'CARBO_HSP':
        Pe = 3.92                   # calibrated from batch mode CARBO tests
        D_t = u_excess * D_h / Pe
        D_yy_s = np.maximum(D_t, 0)           # zeroes out negatives

    #elif species == 'CARBO_CP':
    elif species in ['CARBO_CP', 'Alumina']:
        Pe = 3.92
        D_t = u_excess * D_h / Pe
        D_yy_s = np.maximum(D_t, 0)

    elif species == 'Olivine':
        D_t = u_excess / 2
        D_yy_s = np.maximum(D_t, 1)

    elif species == 'Silica':
        Pe = 13.02                   # calibrated from batch mode CARBO tests
        D_t = u_excess * D_h / Pe
        D_yy_s = np.maximum(D_t, 0)           # zeroes out negatives

    theta = D_yy_s / dt * 2

    return D_yy_s, theta
