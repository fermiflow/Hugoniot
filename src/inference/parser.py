import re 

def _parse(cast, pattern, f):
    res = re.search(pattern, f)
    return None if (res is None) else cast(res.group(1))
    
def parse_filename(f):
    f_strlist = f.split("/")

    n = _parse(int, 'n_([0-9]*)', f_strlist[-2])
    dim = _parse(int, '_dim_([0-9]*)', f_strlist[-2])
    rs = _parse(float, '_rs_([0-9]*\.?[0-9]*)', f_strlist[-2])
    T = _parse(float, '_T_([0-9]*\.?[0-9]*)', f_strlist[-2])

    # flow
    flow_steps = _parse(int, '_fs_([0-9]*)', f_strlist[-2])
    flow_depth = _parse(int, '_fd_([0-9]*)', f_strlist[-2])
    flow_h1size = _parse(int, '_fh1_([0-9]*)', f_strlist[-2])
    flow_h2size = _parse(int, '_fh2_([0-9]*)', f_strlist[-2])

    # transformer
    nlayers = _parse(int, '_l_([0-9]*)', f_strlist[-2])
    modelsize = _parse(int, '_m_([0-9]*)', f_strlist[-2])
    nheads = _parse(int, '_he_([0-9]*)', f_strlist[-2])
    nhidden = _parse(int, '_hi_([0-9]*)', f_strlist[-2])

    # wavefunction
    wfn_depth =  _parse(int, '_wd_([0-9]*)', f_strlist[-2])
    wfn_h1size = _parse(int, '_wh1_([0-9]*)', f_strlist[-2])
    wfn_h2size = _parse(int, '_wh2_([0-9]*)', f_strlist[-2])
    Nf = _parse(int, '_Nf_([0-9]*)', f_strlist[-2])

    Gmax = _parse(int, '_G_([0-9]*)', f_strlist[-2])
    kappa = _parse(int, '_kp_([0-9]*)', f_strlist[-2])

    # mc
    mc_therm = _parse(int, '_mt_([0-9]*)', f_strlist[-2])
    mc_electron_steps = _parse(int, '_mp_([0-9]*)', f_strlist[-2])
    mc_electron_width = _parse(int, '_mw_([0-9]*)', f_strlist[-2])
    mc_therm = _parse(int, '_mt_([0-9]*)', f_strlist[-2])

    lr = _parse(float, '_lr_([0-9]*\.?[0-9]*)', f_strlist[-2])
    decay = _parse(float, '_decay_([0-9]*\.?[0-9]*)', f_strlist[-2])
    damping = _parse(float, '_damping_([0-9]*\.?[0-9]*)', f_strlist[-2])
    maxnorm = _parse(float, 'norm_([0-9]*\.?[0-9]*)', f_strlist[-2])

    clip_factor = _parse(float, '_cl_([0-9]*\.?[0-9]*)', f_strlist[-2])
    alpha = _parse(float, '_al_([0-9]*\.?[0-9]*)', f_strlist[-2])

    batchsize = _parse(int, '_bs_([0-9]*)', f_strlist[-2])
    acc_steps= _parse(int, '_ap_([0-9]*)', f_strlist[-2])

    params = {"n": n, "dim": dim, "rs": rs, "T": T, 
              "flow_steps": flow_steps, "flow_depth": flow_depth, "flow_h1size": flow_h1size, "flow_h2size": flow_h2size, "Nf": Nf, 
              "nlayers": nlayers, "modelsize": modelsize, "nheads": nheads, "nhidden": nhidden,
              "wfn_depth": wfn_depth, "wfn_h1size": wfn_h1size, "wfn_h2size": wfn_h2size,
              "Gmax": Gmax, "kappa": kappa, 
              "mc_therm": mc_therm, "mc_electron_steps": mc_electron_steps, "mc_electron_width": mc_electron_width, "mc_therm": mc_therm, 
              "lr": lr, "decay": decay, "damping": damping, "maxnorm":maxnorm, 
              "clip_factor": clip_factor, "alpha": alpha, 
              "batchsize": batchsize, "acc_steps": acc_steps}

    orbital_strlist = f_strlist[-3].split("_")
    params_strlist = f_strlist[-2].split("_")

    if 'sm' in params_strlist:
        params["smearing"] = True
    else:
        params["smearing"] = False

    if 'pw' in f_strlist or 'pw' in orbital_strlist or 'pw' in params_strlist:
        params["orbital"] = 'pw'
        params["Emax"] = _parse(int, '_Em_([0-9]*)', f)
 
    elif 'hf' in orbital_strlist or 'hf' in params_strlist:
        params["orbital"] = 'hf'
        if 'hf' in params_strlist:
            params["basis"] = params_strlist[params_strlist.index('ao')+1]
        else:
             params["basis"] = orbital_strlist[orbital_strlist.index('hf')+1]

    elif 'hf0' in orbital_strlist or 'hf0' in params_strlist:
        params["orbital"] = 'hf0'
        if 'hf0' in params_strlist:
            params["basis"] = params_strlist[params_strlist.index('ao')+1]
        else:
             params["basis"] = orbital_strlist[orbital_strlist.index('hf0')+1]

    elif 'dft' in orbital_strlist or 'dft' in params_strlist:
        params["orbital"] = 'dft'
        # params["basis"] = params_strlist[params_strlist.index('ao')+1]
        # params["xc"] = params_strlist[params_strlist.index('xc')+1]
        
    else:
        params["orbital"] = 'pw'
        # raise Exception("orbital not fond in", f)

    return params

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

if __name__=='__main__':

    fname = '/data/wanglei/hydrogenT/hf_gth-dzv/n_16_dim_3_rs_1.86_T_10000_Em_5_l_2_m_16_he_4_hi_32_fs_1_fd_3_fh1_32_fh2_16_wd_2_wh1_16_wh2_16_Nf_5_G_15_kp_10_mt_5_mp_10_50_mw_0.03_0.03_lr_1_1_1_decay_0.01_dp_0.001_0.001_0.001_nm_0.001_0.001_0.001_cl_5_al_0.1_bs_512_ap_1/data.txt'
    r = parse_filename(fname)

    print (r)
