import os
import json
from datetime import datetime 

def solarReact_Saver(gas, gas_surf, part, wall, surf, env, GasParams, PartParams, BedParams, WallParams, FinParams, SurfParams, EnvParams, Result, SolFilePath):
    #%% Output file
    # Convert all numpy array to list to make it json serializable
    if BedParams['geo'] == 1:
        BedParams['dz_b']   = BedParams['dz_b'].tolist()    
        BedParams['dx_b']   = BedParams['dx_b'].tolist()  
        BedParams['dz']     = BedParams['dz'].tolist()
        
        WallParams['dz_b']  = WallParams['dz_b'].tolist()
        WallParams['dx_b']  = WallParams['dx_b'].tolist() 
        WallParams['dz']    = WallParams['dz'].tolist()
    
        BedParams['Az_in_b']    = BedParams['Az_in_b'].tolist()
        BedParams['Az_out_b']   = BedParams['Az_out_b'].tolist()
        BedParams['Az_in']      = BedParams['Az_in'].tolist()
        BedParams['Az_out']     = BedParams['Az_out'].tolist()
        BedParams['Az_out_tot_b'] = BedParams['Az_out_tot_b'].tolist()
        
        WallParams['Az_in_b']   = WallParams['Az_in_b'].tolist()
        WallParams['Az_out_b']  = WallParams['Az_out_b'].tolist()
        WallParams['Az']        = WallParams['Az'].tolist()
        
    elif BedParams['geo'] == 2:
        BedParams['d_in_b']     = BedParams['d_in_b'].tolist()
        BedParams['d_out_b']    = BedParams['d_out_b'].tolist()
        BedParams['d_in']       = BedParams['d_in'].tolist()
        BedParams['d_out']      = BedParams['d_out'].tolist()
        
        WallParams['d_in_b']    = WallParams['d_in_b'].tolist()
        WallParams['d_out_b']   = WallParams['d_out_b'].tolist()
        WallParams['d_in']      = WallParams['d_in'].tolist()
        WallParams['d_out']     = WallParams['d_out'].tolist()
    
        BedParams['Ar_in_b']    = BedParams['Ar_in_b'].tolist()
        BedParams['Ar_out_b']   = BedParams['Ar_out_b'].tolist()
        BedParams['Ar_in']      = BedParams['Ar_in'].tolist()
        BedParams['Ar_out']     = BedParams['Ar_out'].tolist()
        BedParams['Ar_out_tot_b'] = BedParams['Ar_out_tot_b'].tolist()
        
        WallParams['Ar_in_b']   = WallParams['Ar_in_b'].tolist()
        WallParams['Ar_out_b']  = WallParams['Ar_out_b'].tolist()
    
    BedParams['Ay']         = BedParams['Ay'].tolist()
    BedParams['Ay_b']       = BedParams['Ay_b'].tolist()
    BedParams['Ay_cond']    = BedParams['Ay_cond'].tolist()
    
    WallParams['Ay']        = WallParams['Ay'].tolist()
    WallParams['Ay_b']      = WallParams['Ay_b'].tolist()
    WallParams['Ay_cond']   = WallParams['Ay_cond'].tolist()
    
    BedParams['D_hyd']      = BedParams['D_hyd'].tolist()
    BedParams['D_hyd_b']    = BedParams['D_hyd_b'].tolist()
    
    BedParams['dVol']       = BedParams['dVol'].tolist()
    BedParams['dVol_b']     = BedParams['dVol_b'].tolist()
    
    WallParams['dVol']      = WallParams['dVol'].tolist()
    WallParams['dVol_b']    = WallParams['dVol_b'].tolist()
    
    BedParams['dy']         = BedParams['dy'].tolist()
    BedParams['dy_b']       = BedParams['dy_b'].tolist()
    BedParams['dy_cond']    = BedParams['dy_cond'].tolist()
    BedParams['index_b']    = BedParams['index_b'].tolist()
    BedParams['y']          = BedParams['y'].tolist()
    BedParams['y_b']        = BedParams['y_b'].tolist()
    BedParams['y_bnd']      = BedParams['y_bnd'].tolist()
    
    WallParams['dy']        = WallParams['dy'].tolist()
    WallParams['dy_b']      = WallParams['dy_b'].tolist()
    WallParams['dy_cond']   = WallParams['dy_cond'].tolist()
    WallParams['y']         = WallParams['y'].tolist()
    WallParams['y_b']       = WallParams['y_b'].tolist()
    
    gas['Wk']               = gas['Wk'].tolist()
    GasParams['X_in']       = GasParams['X_in'].tolist()
    GasParams['Xk_out']     = GasParams['Xk_out'].tolist()
    GasParams['Y_in']       = GasParams['Y_in'].tolist()
    GasParams['Yk_out']     = GasParams['Yk_out'].tolist()
    GasParams['hk_in']      = GasParams['hk_in'].tolist()
    
    if SurfParams['n_p'] > 1:
        SurfParams['Aface'] = SurfParams['Aface'].tolist()
        SurfParams['deltar'] = SurfParams['deltar'].tolist()
        SurfParams['rcell'] = SurfParams['rcell'].tolist()
        SurfParams['rface'] = SurfParams['rface'].tolist()
        SurfParams['rRmax'] = SurfParams['rRmax'].tolist()
        SurfParams['Vcell'] = SurfParams['Vcell'].tolist()
    
    SurfParams['Zk_p_init']     = SurfParams['Zk_p_init'].tolist()
    SurfParams['Zk_p_0']        = SurfParams['Zk_p_0'].tolist()    
    
    env['Wk']               = env['Wk'].tolist()
    
    EnvParams['q_sol_dist'] = EnvParams['q_sol_dist'].tolist()
    
    gas.pop('obj', None)
    gas_surf.pop('obj', None)
    part.pop('obj', None)
    wall.pop('obj', None)
    surf.pop('obj', None)
    env.pop('obj', None)
    
    Result['hT_wb']         = Result['hT_wb'].tolist()
    Result['jk_b']          = Result['jk_b'].tolist()
    Result['Jk_b']          = Result['Jk_b'].tolist()
    Result['P_bg']          = Result['P_bg'].tolist()
    Result['P_p']           = Result['P_p'].tolist()
    Result['phi_bg']        = Result['phi_bg'].tolist()
    Result['phi_bs']        = Result['phi_bs'].tolist()
    Result['T_bg']          = Result['T_bg'].tolist()
    Result['T_bs']          = Result['T_bs'].tolist()
    Result['T_p']           = Result['T_p'].tolist()
    Result['T_wb']          = Result['T_wb'].tolist()
    Result['T_we']          = Result['T_we'].tolist()
    Result['U_hat']         = Result['U_hat'].tolist()
    Result['v_bg']          = Result['v_bg'].tolist()
    Result['v_bs']          = Result['v_bs'].tolist()
    Result['Xk_bg']         = Result['Xk_bg'].tolist()
    Result['Xk_p']          = Result['Xk_p'].tolist()
    Result['Yk_bg']         = Result['Yk_bg'].tolist()
    Result['Yk_p']          = Result['Yk_p'].tolist()
    
    if PartParams['simple_part'] == 1:
        Result['Xk_p_int']      = Result['Xk_p_int'].tolist()
        Result['Yk_p_int']      = Result['Yk_p_int'].tolist()
    elif PartParams['multi_part'] == 1:
        Result['Xk_p_bound']      = Result['Xk_p_bound'].tolist()
        Result['Yk_p_bound']      = Result['Yk_p_bound'].tolist()
    
    #%% Save output file that collects "Result" dict
    save = {
        "gas"           : gas,
        "gas_surf"      : gas_surf,
        "part"          : part,
        "wall"          : wall,
        "surf"          : surf,
        "env"           : env,
        "GasParams"     : GasParams,
        "PartParams"    : PartParams,
        "BedParams"     : BedParams,
        "WallParams"    : WallParams,
        "FinParams"     : FinParams,
        "SurfParams"    : SurfParams,
        "EnvParams"     : EnvParams,
        "Result"        : Result
        }
    
    # Improve file naming to include case condition
    """
    file_name_to_save = PartParams['id'] + '_' + BedParams['geo_name'] + '-' + \
        f"{BedParams['geo_size']:.3f}" + '_' + 'y' + '-' + f"{BedParams['y'][-1]:.1f}" + '_' + 'dp' + '-' + \
        f"{1e6*PartParams['dp']:.0f}" + '_'  + 'Tin' + '-' + f"{GasParams['T_in']-273.15:.0f}" + '_' + \
        'Pin' + '-' + f"{GasParams['P_in']/1e5:.0f}" + '_' + 'mfluxin' + '-' + f"{GasParams['mflux_in']:.2f}" + '_' + \
        'qsol_ap' + '-' + f"{EnvParams['q_sol_aper']/1e3:.0f}" + '_' + 'qabs_avg' + '-' + f"{EnvParams['q_abs_avg']/1e3:.0f}"
   
   """
    
    file_name_to_save = 'OD' + '-' + f"{BedParams['geo_size']:.2f}" + '_' + 'y' + '-' + f"{BedParams['y'][-1]:.1f}" + '_'  \
        +'dp' + '-' + f"{1e6*PartParams['dp']:.0f}" + '_' + 'Tin' + '-' + f"{GasParams['T_in']-273.15:.0f}" + '_' \
        + 'Pin' + '-' + f"{GasParams['P_in']/1e5:.0f}" + '_' + 'mfluxin' + '-' + f"{GasParams['mflux_in']:.2f}" + '_' \
        + 'qsol_ap' + '-' + f"{EnvParams['q_sol_aper']/1e3:.0f}" + '_' + 'qabs_avg' + '-' + f"{EnvParams['q_abs_avg']/1e3:.0f}" \
        + BedParams['custom_name']
   
    with open(os.path.join(SolFilePath, "Output_" + file_name_to_save + ".json"), 'w') as file_out:
        json.dump(save, file_out)      