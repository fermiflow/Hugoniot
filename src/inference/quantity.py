import csv
import math

from src.checkpoint import load_data
from src.inference.utils import *
from src.inference.find_ckpt_sample import find_ckpt_sample
from src.inference.parser import parse_filename

def quantity(obs: str,
             df: pd.DataFrame,
             n: int,
             rs: float,
             T: float,
             isotope: str='Deuterium',
    ):
    """
        Compute the quantity of interest for a given observable and dataset.
    Inputs:
        obs: str, observable of interest, 
            'f', 'etot', 'k', 'vtot', 'vep', 'vee', 'vpp', 
            'p', 's', 'se', 'sp', 'acc_s', 'acc_x'
        df: pd.DataFrame, dataset
        n: int, principal quantum number
        rs: float, Wigner-Seitz radius
        T: float, temperature in Kelvin
        isotope: str, isotope of the dataset
    Outputs:
        y: np.ndarray, quantity of interest
        yerr: np.ndarray, uncertainty of the quantity of interest
    Note:
        all energy quantities are in Rydberg.
        pressure is in GPa.
    """
    
    if isotope == "Deuterium" or isotope == "D":
        mp = 3697.9
    elif isotope == "Hydrogen" or isotope == "H":
        mp = 1836
    else:
        raise ValueError(f"Isotope {isotope} not recognized.")

    L = (4/3*jnp.pi*n)**(1/3)
    Kp = 1.5*T/157887.66 # proton kinetic energy in Ry
    ds = -jax.scipy.special.gammaln(n+1)/n + 1.5*(1+np.log(mp*rs**2/(4*np.pi)*T/157887.66))

    if obs == 'f':  
        y = df['f'].values + Kp - ds*T/157887.66
        yerr = df['f_err'].values
    elif obs == 'vtot':
        y = df['vpp'].values + df['vep'].values + df['vee'].values
        yerr = np.sqrt(df['vpp_err'].values**2 + df['vep_err'].values**2 + df['vee_err'].values**2)
    elif obs == 'etot':
        y = df['e'].values + df['vpp'].values + Kp
        yerr = np.sqrt(df['e_err'].values**2 + df['vpp_err'].values**2)
    elif obs in ['acc_s', 'acc_x']:
        y = df[obs].values
        yerr = np.zeros_like(y)
    elif obs == 'p':
        y = df['p'].values + (2*Kp)/(3*(L*rs)**3)*14710.513242194795*n # GPa
        yerr = df[obs+'_err'].values
    elif obs == 's':
        y, yerr = (df['se'] + df['sp'] + ds), np.sqrt(df['se_err']**2 + df['sp_err']**2)
    elif obs == 'sp':
        y, yerr = df[obs].values + ds, df[obs+'_err'].values
    else:
        y, yerr = df[obs].values, df[obs+'_err'].values

    return y, yerr

def quantity_inf(obs: str,
                 ckpt_file: str,
                 isotope: str='Deuterium',
                 auto_load: bool=True,
                 silent_mode: bool=False,
    ):
    """
        Compute the quantity of interest for a given observable and inference dataset.
    Inputs:
        obs: str, observable of interest, 
            'f', 'etot', 'k', 'vtot', 'vep', 'vee', 'vpp', 
            'p', 's', 'se', 'sp', 'acc_s', 'acc_x', 'epcov'
        ckpt_file: str, path to the checkpoint or inference dataset.
                   remember to run sample_x.py at this checkpoint file first.
        isotope: str, isotope of the dataset.
        auto_load: bool, if True, automatically load the biggest checkpoint sample file.
        silent_mode: bool, if True, do not print the loading message.
    Outputs:
        y: np.ndarray, quantity of interest
        yerr: np.ndarray, uncertainty of the quantity of interest
    Note:
        all energy quantities are in Rydberg.
        pressure is in GPa.
    """

    if isotope == "Deuterium" or isotope == "D":
        mp = 3697.9
    elif isotope == "Hydrogen" or isotope == "H":
        mp = 1836
    else:
        raise ValueError(f"Isotope {isotope} not recognized.")

    directory = os.path.dirname(ckpt_file)
    cfg_filename = os.path.join(directory, 'config.yaml')
    if os.path.isfile(cfg_filename):
        cfg = OmegaConf.load(cfg_filename)
        rs = cfg.rs
        n = cfg.num
        T = cfg.T
    else:
        params = parse_filename(ckpt_file)
        rs = params["rs"]
        n = params["n"]
        T = params["T"]

    L = (4/3*jnp.pi*n)**(1/3)
    Kp = 1.5*T/157887.66 # proton kinetic energy in Ry
    ds = -jax.scipy.special.gammaln(n+1)/n + 1.5*(1+np.log(mp*rs**2/(4*np.pi)*T/157887.66))

    if auto_load:
        auto_find_f, auto_find_f_batch = find_ckpt_sample(ckpt_file, mode="sx", silent_mode=True)
        if auto_find_f is not None:
            ckpt_file = auto_find_f
            if not silent_mode:
                print(f"{GREEN}auto load the biggest ckeckpoint sample file:\n{RESET}File:", ckpt_file)
        else:
            raise FileNotFoundError(f"{RED}no ckeckpoint sample file found, please check the path{RESET}")
    else:
        if not silent_mode:
            print(f"{GREEN}load checkpoint file:\n{RESET}File:", ckpt_file)

    data = load_data(ckpt_file)

    if obs == 'f':
        F_data = data['F'] + Kp - ds*T/157887.66
        y, yerr = F_data.mean(), F_data.std()/jnp.sqrt(F_data.shape[0])
    elif obs == 'etot':
        Etot_data = data['E'] + data['Vpp'] + Kp
        y, yerr = Etot_data.mean(), Etot_data.std()/jnp.sqrt(Etot_data.shape[0])
    elif obs == 'vtot':
        Vtot_data = data['Vpp'] + data['Vep'] + data['Vee']
        y, yerr = Vtot_data.mean(), Vtot_data.std()/jnp.sqrt(Vtot_data.shape[0])
    elif obs == 'vep':
        Vep_data = data['Vep']
        y, yerr = Vep_data.mean(), Vep_data.std()/jnp.sqrt(Vep_data.shape[0])
    elif obs == 'vee':
        Vee_data = data['Vee']
        y, yerr = Vee_data.mean(), Vee_data.std()/jnp.sqrt(Vee_data.shape[0])
    elif obs == 'vpp':
        Vpp_data = data['Vpp']
        y, yerr = Vpp_data.mean(), Vpp_data.std()/jnp.sqrt(Vpp_data.shape[0])
    elif obs == 'k':
        K_data = data['K']
        y, yerr = K_data.mean(), K_data.std()/jnp.sqrt(K_data.shape[0])
    elif obs == 'p':
        P_data = data['P'] + (2*Kp)/(3*(L*rs)**3)*14710.513242194795*n
        y, yerr = P_data.mean(), P_data.std()/jnp.sqrt(P_data.shape[0])
    elif obs == 's':
        S_data = data['Se'] + data['Sp'] + ds
        y, yerr = S_data.mean(), S_data.std()/jnp.sqrt(S_data.shape[0])
    elif obs == 'se':
        Se_data = data['Se']
        y, yerr = Se_data.mean(), Se_data.std()/jnp.sqrt(Se_data.shape[0])
    elif obs == 'sp':
        Sp_data = data['Sp'] + ds
        y, yerr = Sp_data.mean(), Sp_data.std()/jnp.sqrt(Sp_data.shape[0])
    elif obs == 'epcov':
        E_data = data['E'] + data['Vpp']
        P_data = data['P']
        EP_data = (data['E'] + data['Vpp']) * data['P']
        y = (EP_data.mean() - E_data.mean() * P_data.mean()) / E_data.shape[0]
        yerr = 0

    return y, yerr

def format_uncertainty(unc, sig_figs=2):
    formatted_unc = ("{:."+str(sig_figs-1)+"e}").format(unc)
    mantissa, exponent = formatted_unc.split("e")
    mantissa = ("{:."+str(sig_figs-1)+"f}").format(float(mantissa))
    formatted_unc = f"{mantissa}e{exponent}" if exponent != "e+00" else mantissa
    return formatted_unc

def format_measurement(error, sig_figs=2):
    err_str = format_uncertainty(error, sig_figs)
    if 'e' in err_str:
        err_mantissa, err_exponent = err_str.split('e')
        err_decimal_place = len(err_mantissa.split('.')[1]) if '.' in err_mantissa else 0
        err_decimal_place -= int(err_exponent)
    else:
        if '.' in err_str:
            err_decimal_place = len(err_str.split('.')[1])
        else:
            err_decimal_place = 0
    return err_decimal_place

def eos(ckpt_files, isotope: str='Deuterium', unit='Ry', 
        obs_list=['f', 'etot', 'vtot', 'vep', 'vee', 'vpp', 'k', 'p', 's', 'se', 'sp', 'epcov'],
        tex_list=['f', 'etot', 'vep', 'vee', 'vpp', 'k', 'p', 'se', 'sp'],
        save_csv_filename = None, 
        save_textable_filename = None, 
        err_sig_figs=2):
    """
        Compute the equation of state for a given sampled checkpoint file.
    Inputs:
        ckpt_files: list of str, path to the checkpoint files.
        isotope: str, isotope of the dataset.
        unit: str, unit of the output, 'Ha', 'Ry', 'eV'.
        obs_list: list, list of observables to compute.
        tex_list: list, list of observables to save in the latex table.
        save_csv_filename: str, path to save the output csv file.
        save_textable_filename: str, path to save the output latex table file.
        err_sig_figs: str, 
    """

    if unit == 'Ha':
        unit_conv = 0.5 # 1 Ry = 2 Ha
    elif unit == 'Ry':
        unit_conv = 1
    elif unit == "eV":
        unit_conv = 13.6056980659

    if save_csv_filename is not None:
        csv_file = open(save_csv_filename, mode='w', newline='')
        writer = csv.writer(csv_file)
        content_list = ['n', 'T', 'rs']
        if 'f' in obs_list:
            content_list.extend(['f', 'f_err'])
        if 'etot' in obs_list:
            content_list.extend(['etot', 'etot_err'])
        if 'vtot' in obs_list:
            content_list.extend(['vtot', 'vtot_err'])
        if 'vep' in obs_list:
            content_list.extend(['vep', 'vep_err'])
        if 'vee' in obs_list:
            content_list.extend(['vee', 'vee_err'])
        if 'vpp' in obs_list:
            content_list.extend(['vpp', 'vpp_err'])
        if 'k' in obs_list:
            content_list.extend(['k', 'k_err'])
        if 'p' in obs_list:
            content_list.extend(['p', 'p_err'])
        if 's' in obs_list:
            content_list.extend(['s', 's_err'])
        if 'se' in obs_list:
            content_list.extend(['se', 'se_err'])
        if 'sp' in obs_list:
            content_list.extend(['sp', 'sp_err'])
        if 'epcov' in obs_list:
            content_list.extend(['ep_cov'])
        writer.writerows([content_list])
        
    if save_textable_filename is not None:
        content_str = "$N$ & $r_s$ & $T$ & "
        if 'f' in tex_list:
            content_str += "$F$ & "
        if 'etot' in tex_list:
            content_str += "$E_{tot}$ & "
        if 'vtot' in tex_list:
            content_str += "$V_{tot}$ & "
        if 'vep' in tex_list:
            content_str += "$V_{ep}$ & "
        if 'vee' in tex_list:
            content_str += "$V_{ee}$ & "
        if 'vpp' in tex_list:
            content_str += "$V_{pp}$ & "
        if 'k' in tex_list:
            content_str += "$K$ & "
        if 'p' in tex_list:
            content_str += "$P$ & "
        if 's' in tex_list:
            content_str += "$S$ & "
        if 'se' in tex_list:
            content_str += "$S_e$ & "
        if 'sp' in tex_list:
            content_str += "$S_p$ & "
        content_str = content_str[:-2] + "\\\\\n"
        with open(save_textable_filename, 'w') as f_textable:
            f_textable.write(content_str)

    for ckpt_file in ckpt_files:
        
        directory = os.path.dirname(ckpt_file)
        cfg_filename = os.path.join(directory, 'config.yaml')
        if os.path.isfile(cfg_filename):
            cfg = OmegaConf.load(cfg_filename)
            rs = cfg.rs
            n = cfg.num
            T = cfg.T
        else:
            params = parse_filename(ckpt_file)
            rs = params["rs"]
            n = params["n"]
            T = params["T"]

        if 'sample_sx' in ckpt_file:
            auto_load = False
            print(f"{GREEN}load checkpoint sample file:\n{RESET}File:", ckpt_file)
        else:
            auto_find_f, auto_find_f_batch = find_ckpt_sample(ckpt_file, mode="sx", silent_mode=True)
            if auto_find_f is not None:
                auto_load = True
                print(f"{GREEN}auto load the biggest ckeckpoint sample file:\n{RESET}File:", auto_find_f)
            else:
                raise FileNotFoundError(f"{RED}no ckeckpoint sample file found, please check the path{RESET}")

        print(f"{MAGENTA}-------- system info --------{RESET}")
        print(f"{MAGENTA}isotope:{RESET} %s" % isotope)
        print(f"{MAGENTA}n:{RESET} %d" % n)
        print(f"{MAGENTA}rs:{RESET} %.2f" % rs)
        print(f"{MAGENTA}T:{RESET} %d" % T)
        print(f"{MAGENTA}energy unit:{RESET} %s" % unit)
        print(f"{MAGENTA}pressure unit:{RESET} GPa")
        print(f"{MAGENTA}entropy unit:{RESET} kB")
        print(f"{MAGENTA}batch:{RESET} %s" % auto_find_f_batch)

        print(f"{BLUE}-------- observables --------{RESET}")
        if 'f' in obs_list or 'f' in tex_list:
            F, F_err = quantity_inf('f', ckpt_file, isotope, silent_mode=True, auto_load=auto_load) * unit_conv
            F_err_decimal_place = format_measurement(F_err, err_sig_figs)
            F_value_str = "{:.{}f}".format(F, F_err_decimal_place)
            F_err_str = "{:.{}f}".format(F_err, F_err_decimal_place)
            print(f"{BLUE}F:{RESET}" + f"{F_value_str} ± {F_err_str}")
        if 'etot' in obs_list or 'etot' in tex_list:
            Etot, Etot_err = quantity_inf('etot', ckpt_file, isotope, silent_mode=True, auto_load=auto_load) * unit_conv
            Etot_err_decimal_place = format_measurement(Etot_err, err_sig_figs)
            Etot_value_str = "{:.{}f}".format(Etot, Etot_err_decimal_place)
            Etot_err_str = "{:.{}f}".format(Etot_err, Etot_err_decimal_place)
            print(f"{BLUE}Etot:{RESET}" + f"{Etot_value_str} ± {Etot_err_str}")
        if 'vtot' in obs_list or 'vtot' in tex_list:
            Vtot, Vtot_err = quantity_inf('vtot', ckpt_file, isotope, silent_mode=True, auto_load=auto_load) * unit_conv
            Vtot_err_decimal_place = format_measurement(Vtot_err, err_sig_figs)
            Vtot_value_str = "{:.{}f}".format(Vtot, Vtot_err_decimal_place)
            Vtot_err_str = "{:.{}f}".format(Vtot_err, Vtot_err_decimal_place)
            print(f"{BLUE}Vtot:{RESET}" + f"{Vtot_value_str} ± {Vtot_err_str}")
        if 'vep' in obs_list or 'vep' in tex_list:
            Vep, Vep_err = quantity_inf('vep', ckpt_file, isotope, silent_mode=True, auto_load=auto_load) * unit_conv
            Vep_err_decimal_place = format_measurement(Vep_err, err_sig_figs)
            Vep_value_str = "{:.{}f}".format(Vep, Vep_err_decimal_place)
            Vep_err_str = "{:.{}f}".format(Vep_err, Vep_err_decimal_place)
            print(f"{BLUE}Vep:{RESET}" + f"{Vep_value_str} ± {Vep_err_str}")
        if 'vee' in obs_list or 'vee' in tex_list:
            Vee, Vee_err = quantity_inf('vee', ckpt_file, isotope, silent_mode=True, auto_load=auto_load) * unit_conv
            Vee_err_decimal_place = format_measurement(Vee_err, err_sig_figs)
            Vee_value_str = "{:.{}f}".format(Vee, Vee_err_decimal_place)
            Vee_err_str = "{:.{}f}".format(Vee_err, Vee_err_decimal_place)
            print(f"{BLUE}Vee:{RESET}" + f"{Vee_value_str} ± {Vee_err_str}")
        if 'vpp' in obs_list or 'vpp' in tex_list:
            Vpp, Vpp_err = quantity_inf('vpp', ckpt_file, isotope, silent_mode=True, auto_load=auto_load) * unit_conv
            Vpp_err_decimal_place = format_measurement(Vpp_err, err_sig_figs)
            Vpp_value_str = "{:.{}f}".format(Vpp, Vpp_err_decimal_place)
            Vpp_err_str = "{:.{}f}".format(Vpp_err, Vpp_err_decimal_place)
            print(f"{BLUE}Vpp:{RESET}" + f"{Vpp_value_str} ± {Vpp_err_str}")
        if 'k' in obs_list or 'k' in tex_list:
            K, K_err = quantity_inf('k', ckpt_file, isotope, silent_mode=True, auto_load=auto_load) * unit_conv
            K_err_decimal_place = format_measurement(K_err, err_sig_figs)
            K_value_str = "{:.{}f}".format(K, K_err_decimal_place)
            K_err_str = "{:.{}f}".format(K_err, K_err_decimal_place)
            print(f"{BLUE}K:{RESET}" + f"{K_value_str} ± {K_err_str}")
        if 'p' in obs_list or 'p' in tex_list:
            P, P_err = quantity_inf('p', ckpt_file, isotope, silent_mode=True, auto_load=auto_load) * unit_conv
            P_err_decimal_place = format_measurement(P_err, err_sig_figs)
            P_value_str = "{:.{}f}".format(P, P_err_decimal_place)
            P_err_str = "{:.{}f}".format(P_err, P_err_decimal_place)
            print(f"{BLUE}P:{RESET}" + f"{P_value_str} ± {P_err_str}")
        if 's' in obs_list or 's' in tex_list:
            S, S_err = quantity_inf('s', ckpt_file, isotope, silent_mode=True, auto_load=auto_load) * unit_conv
            S_err_decimal_place = format_measurement(S_err, err_sig_figs)
            S_value_str = "{:.{}f}".format(S, S_err_decimal_place)
            S_err_str = "{:.{}f}".format(S_err, S_err_decimal_place)
            print(f"{BLUE}S:{RESET}" + f"{S_value_str} ± {S_err_str}")
        if 'se' in obs_list or 'se' in tex_list:
            Se, Se_err = quantity_inf('se', ckpt_file, isotope, silent_mode=True, auto_load=auto_load) * unit_conv
            Se_err_decimal_place = format_measurement(Se_err, err_sig_figs)
            Se_value_str = "{:.{}f}".format(Se, Se_err_decimal_place)
            Se_err_str =  "{:.{}f}".format(Se_err, Se_err_decimal_place)
            print(f"{BLUE}Se:{RESET}" + f"{Se_value_str} ± {Se_err_str}")
        if 'sp' in obs_list or 'sp' in tex_list:
            Sp, Sp_err = quantity_inf('sp', ckpt_file, isotope, silent_mode=True, auto_load=auto_load) * unit_conv
            Sp_err_decimal_place = format_measurement(Sp_err, err_sig_figs)
            Sp_value_str = "{:.{}f}".format(Sp, Sp_err_decimal_place)
            Sp_err_str = "{:.{}f}".format(Sp_err, Sp_err_decimal_place)
            print(f"{BLUE}Sp:{RESET}" + f"{Sp_value_str} ± {Sp_err_str}")
        if 'epcov' in obs_list:
            EPcov = quantity_inf('epcov', ckpt_file, isotope, silent_mode=True, auto_load=auto_load)[0] * unit_conv
            EPcov_str = "{:.{}f}".format(EPcov, 6)
            print(f"{BLUE}EPcov:{RESET}" + f"{EPcov_str}")

        if save_csv_filename is not None:
            data_list = [n, T, format(rs, '.2f')]
            if 'f' in obs_list:
                data_list.extend([F_value_str, F_err_str])
            if 'etot' in obs_list:
                data_list.extend([Etot_value_str, Etot_err_str])
            if 'vtot' in obs_list:
                data_list.extend([Vtot_value_str, Vtot_err_str])
            if 'vep' in obs_list:
                data_list.extend([Vep_value_str, Vep_err_str])
            if 'vee' in obs_list:
                data_list.extend([Vee_value_str, Vee_err_str])
            if 'vpp' in obs_list:
                data_list.extend([Vpp_value_str, Vpp_err_str])
            if 'k' in obs_list:
                data_list.extend([K_value_str, K_err_str])
            if 'p' in obs_list:
                data_list.extend([P_value_str, P_err_str])
            if 's' in obs_list:
                data_list.extend([S_value_str, S_err_str])
            if 'se' in obs_list:
                data_list.extend([Se_value_str, Se_err_str])
            if 'sp' in obs_list:
                data_list.extend([Sp_value_str, Sp_err_str])
            if 'epcov' in obs_list:
                data_list.append(EPcov_str)
            writer.writerows([data_list])
            
        if save_textable_filename is not None:
            textable_str = ''
            textable_str += f"{n} & {T} & {format(rs, '.2f')} & "
            if 'f' in tex_list:
                F_str = f"{F_value_str}({round(F_err*10**F_err_decimal_place)})"
                textable_str += F_str + " & "
            if 'etot' in tex_list:
                Etot_str = f"{Etot_value_str}({round(Etot_err*10**Etot_err_decimal_place)})"
                textable_str += Etot_str + " & "
            if 'vtot' in tex_list:
                Vtot_str = f"{Vtot_value_str}({round(Vtot_err*10**Vtot_err_decimal_place)})"
                textable_str += Vtot_str + " & "
            if 'vep' in tex_list:
                Vep_str = f"{Vep_value_str}({round(Vep_err*10**Vep_err_decimal_place)})"
                textable_str += Vep_str + " & "
            if 'vee' in tex_list:
                Vee_str = f"{Vee_value_str}({round(Vee_err*10**Vee_err_decimal_place)})"
                textable_str += Vee_str + " & "
            if 'vpp' in tex_list:
                Vpp_str = f"{Vpp_value_str}({round(Vpp_err*10**Vpp_err_decimal_place)})"
                textable_str += Vpp_str + " & "
            if 'k' in tex_list:
                K_str = f"{K_value_str}({round(K_err*10**K_err_decimal_place)})"
                textable_str += K_str + " & "
            if 'p' in tex_list:
                P_str = f"{P_value_str}({round(P_err*10**P_err_decimal_place)})"
                textable_str += P_str + " & "
            if 's' in tex_list:
                S_str = f"{S_value_str}({round(S_err*10**S_err_decimal_place)})"
                textable_str += S_str + " & "
            if 'se' in tex_list:
                Se_str = f"{Se_value_str}({round(Se_err*10**Se_err_decimal_place)})"
                textable_str += Se_str + " & "
            if 'sp' in tex_list:
                Sp_str = f"{Sp_value_str}({round(Sp_err*10**Sp_err_decimal_place)})"
                textable_str += Sp_str + " & "
            
            textable_str = textable_str[:-2] + "\\\\\n"
            with open(save_textable_filename, 'a') as f_textable:
                f_textable.write(textable_str)
            
    if save_csv_filename is not None:
        print(f"{GREEN}save csv file:{RESET}", save_csv_filename)  
    if save_textable_filename is not None:
        print(f"{GREEN}save textable file:{RESET}", save_textable_filename)
