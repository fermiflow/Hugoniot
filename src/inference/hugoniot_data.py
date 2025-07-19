import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)

from src.inference.utils import *
from src.inference.parser import parse_filename
from src.inference.colors import default_colors
from src.inference.markers import markers
from src.checkpoint import load_data
from src.inference.hugoniot import eval_rho_ratio, eval_H, eval_H_std, eval_rho_hugoniot, eval_P_hugoniot
from src.inference.quantity import quantity

# E0 = -0.58380 # Ha/atom
# rho0 = 0.167 # g/cm^3

# start point of Deuterium Hugoniot curve
E0 = -15.886 / 27.211386245981 # Ha/atom, Ceperley 2000 prl
rho0 = 0.171 # g/cm^3, Ceperley 2000 prl

Bohr = 5.2917724900001E-11 # m
J2Ha = 2.2937104486906E17 # Ha/J
rhoDunit = 1.75313**3/rho0
rs0 = rhoDunit**(1/3)
Vunit = 4./3*jnp.pi*(Bohr)**3*J2Ha*1E9
V0 = Vunit*rs0**3

def eval_data_hugoniot(data_filename, average_epoch=10, epoch_end=-1, isotope='Deuterium', axes=None):
    params = parse_filename(data_filename)
    n = params["n"]
    rs = params["rs"]
    T = params["T"]
    batchsize = params["batchsize"]
    acc_steps = params["acc_steps"]
    rho = eval_rho_ratio(rs)

    df = pd.read_csv(data_filename, delimiter=r"\s+")
    f_ave, f_std = quantity('f', df, n, rs, T, isotope=isotope)
    etot_ave, etot_std = quantity('etot', df, n, rs, T, isotope=isotope)
    p_ave, p_std = quantity('p', df, n, rs, T, isotope=isotope)
    eptot_cov = df['ep_cov'].values

    f_ave = f_ave[epoch_end-average_epoch:epoch_end]
    f_std = f_std[epoch_end-average_epoch:epoch_end]
    etot_ave = etot_ave[epoch_end-average_epoch:epoch_end]
    etot_std = etot_std[epoch_end-average_epoch:epoch_end]
    p_ave = p_ave[epoch_end-average_epoch:epoch_end]
    p_std = p_std[epoch_end-average_epoch:epoch_end]
    eptot_cov = eptot_cov[epoch_end-average_epoch:epoch_end]

    F_ave = np.mean(f_ave)
    F_std = np.sqrt((np.mean(f_std**2 + f_ave**2)-np.mean(f_ave)**2)/average_epoch)
    E_ave = np.mean(etot_ave)
    E_std = np.sqrt((np.mean(etot_std**2 + etot_ave**2)-np.mean(etot_ave)**2)/average_epoch)
    P_ave = np.mean(p_ave)
    P_std = np.sqrt((np.mean(p_std**2 + p_ave**2)-np.mean(p_ave)**2)/average_epoch)
    ep_ave = etot_ave * p_ave + eptot_cov * batchsize * acc_steps
    EP_cov = (np.mean(ep_ave) - np.mean(etot_ave) * np.mean(p_ave))/batchsize/acc_steps/average_epoch

    print(f"{MAGENTA}E (Ry):{RESET} {E_ave} ± {E_std}")
    print(f"{MAGENTA}P (GPa):{RESET} {P_ave} ± {P_std}")

    Ry2Ha = 0.5
    F_ave *= Ry2Ha
    F_std *= Ry2Ha
    E_ave *= Ry2Ha
    E_std *= Ry2Ha
    EP_cov *= Ry2Ha

    H = eval_H(E_ave, P_ave, rs)
    H_std = eval_H_std(E_ave, E_std, P_ave, P_std, EP_cov, rs)
    print(f"{MAGENTA}H (Ha):{RESET} {H} ± {H_std}")
    
    if axes is not None:
        axes[0].errorbar(rho, P_ave, yerr=P_std, fmt='-o', capsize=2, markersize=4, linewidth=0.7, color=default_colors[0])
        axes[1].errorbar(rho, H, yerr=H_std, fmt='-o', capsize=4, markersize=4, color=default_colors[0])

    return H, H_std, E_ave, E_std, P_ave, P_std, EP_cov, n, T, rs, rho, F_ave, F_std

def hugoniot_point(files, 
                   axes = None,
                   index: int = 0,
                   dark_mode: bool = False, 
                   silent_mode: bool = False
    ):
    """
    Evaluate the Hugoniot point from the given files.
    Input:
        files: list of str
    Output:
        rho_hugoniot: float, rho/rho0 on Hugoniot curve.
        rho_std_hugoniot: float, standard deviation of rho/rho0 on Hugoniot curve.
        P_hugoniot: float, pressure on Hugoniot curve, unit: GPa
        P_std_hugoniot: float, standard deviation of pressure on Hugoniot curve, unit: GPa
    """
    if len(files) != 2:
        raise ValueError("The number of files should be 2.")

    H1, H1_std, E1_ave, E1_std, P1_ave, P1_std, EP1_cov, n1, T1, rs1, rho1, F1_ave, F1_std = eval_data_hugoniot(files[0])
    H2, H2_std, E2_ave, E2_std, P2_ave, P2_std, EP2_cov, n2, T2, rs2, rho2, F2_ave, F2_std = eval_data_hugoniot(files[1])

    if n1 != n2:
        raise ValueError("The number of atoms should be the same.")
    if T1 != T2:
        raise ValueError("The temperature should be the same.")

    if not silent_mode:
        print(f"{YELLOW}\n========= Calculate Hugoniot ========{RESET}")
        print(f"{MAGENTA}Loaded n:{RESET} {n1}")
        print(f"{MAGENTA}Loaded T:{RESET} {T1}")
        print(f"{MAGENTA}Loaded rs1:{RESET} {rs1}")
        print(f"{MAGENTA}Loaded rho1:{RESET} {rho1}")
        print(f"{MAGENTA}Loaded E1 (Ha):{RESET} {E1_ave} ± {E1_std}")
        print(f"{MAGENTA}Loaded P1 (GPa):{RESET} {P1_ave} ± {P1_std}")
        print(f"{MAGENTA}Loaded H1:{RESET} {H1} ± {H1_std}")
        print(f"{MAGENTA}Loaded rs2:{RESET} {rs2}")
        print(f"{MAGENTA}Loaded rho2:{RESET} {rho2}")
        print(f"{MAGENTA}Loaded E2 (Ha):{RESET} {E2_ave} ± {E2_std}")
        print(f"{MAGENTA}Loaded P2 (GPa):{RESET} {P2_ave} ± {P2_std}")
        print(f"{MAGENTA}Loaded H2:{RESET} {H2} ± {H2_std}")

    if H1 * H2 > 0:
        print(f"{RED}H1 and H2 have the same sign, extrapolation may cause Numerical Instability.{RESET}")
    
    rho_hugoniot, rho_std_hugoniot = eval_rho_hugoniot(E1_ave, E1_std, P1_ave, P1_std, EP1_cov, rs1, E2_ave, E2_std, P2_ave, P2_std, EP2_cov, rs2)
    P_hugoniot, P_std_hugoniot = eval_P_hugoniot(E1_ave, E1_std, P1_ave, P1_std, EP1_cov, rs1, E2_ave, E2_std, P2_ave, P2_std, EP2_cov, rs2)
    if not silent_mode:
        print(f"{MAGENTA}Hugoniot rho/rho0:{RESET} {rho_hugoniot} ± {rho_std_hugoniot}")
        print(f"{MAGENTA}Hugoniot P (GPa):{RESET} {P_hugoniot} ± {P_std_hugoniot}")
    
    if axes is not None:
        # dark mode
        if dark_mode:
            # facecolor = 'black'
            # facecolor = '#404040'
            facecolor = (0.2, 0.2, 0.2)
            fontcolor = 'white'
        else:
            facecolor = 'white'
            fontcolor = 'black'

        axes[0].errorbar([rho1, rho2], [P1_ave, P2_ave], yerr=[P1_std, P2_std], fmt='-o', capsize=2, markersize=4, linewidth=0.7, color=default_colors[index])
        axes[0].axvline(rho_hugoniot, color=fontcolor, linestyle='--', linewidth=0.5)

        axes[1].errorbar([rho1, rho2], [H1, H2], yerr=[H1_std, H2_std], fmt='-o', capsize=4, color=default_colors[index], markersize=4)
        axes[1].axvline(rho_hugoniot, color=fontcolor, linestyle='--', linewidth=0.5)

    return rho_hugoniot, rho_std_hugoniot, P_hugoniot, P_std_hugoniot, axes

def plot_hugoniot(files, savefigname: str = None, dark_mode: bool = False):
    if savefigname is not None:
        # dark mode
        if dark_mode:
            # facecolor = 'black'
            # facecolor = '#404040'
            facecolor = (0.2, 0.2, 0.2)
            fontcolor = 'white'
        else:
            facecolor = 'white'
            fontcolor = 'black'
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), dpi=300, facecolor=facecolor)

        if len(files) % 2 != 0:
            raise ValueError("The number of files should be even.")
        
        n_points = len(files)//2
        rho_hugoniot = np.zeros(n_points)
        rho_std_hugoniot = np.zeros(n_points)
        P_hugoniot = np.zeros(n_points)
        P_std_hugoniot = np.zeros(n_points)
        for i in range(n_points):
            rho_hugoniot[i], rho_std_hugoniot[i], P_hugoniot[i], P_std_hugoniot[i], axes = hugoniot_point(files[i*2:i*2+2], axes, index=i, dark_mode=dark_mode, silent_mode=True)

        ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs/prl2000_hugoniot.csv'
        ref_df = pd.read_csv(ref_filename, delimiter=r"\s+")
        ref_n = 4
        rho_ref = ref_df['rho'].to_numpy()[-ref_n:]/rho0 # rho/rho0
        rho_err_ref = ref_df['rho_err'].to_numpy()[-ref_n:]/rho0 # rho/rho0
        P_ref = ref_df['p'].to_numpy()[-ref_n:] * 100 # GPa
        P_err_ref = ref_df['p_err'].to_numpy()[-ref_n:] * 100 # GPa

        ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs/prl2000_hugoniot_exp1.csv'
        ref_df = pd.read_csv(ref_filename, delimiter=r"\s+")
        rho_ref_exp1 = ref_df['rho'].to_numpy()/rho0 # rho/rho0
        rho_err_ref_exp1 = ref_df['rho_err'].to_numpy()/rho0 # rho/rho0
        P_ref_exp1 = ref_df['p'].to_numpy() * 100 # GPa
        P_err_ref_exp1 = ref_df['p_err'].to_numpy() * 100 # GPa

        ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs/prl2000_hugoniot_exp2.csv'
        ref_df = pd.read_csv(ref_filename, delimiter=r"\s+")
        rho_ref_exp2 = ref_df['rho'].to_numpy()/rho0 # rho/rho0
        rho_err_ref_exp2 = ref_df['rho_err'].to_numpy()/rho0 # rho/rho0
        P_ref_exp2 = ref_df['p'].to_numpy() * 100 # GPa
        P_err_ref_exp2 = ref_df['p_err'].to_numpy() * 100 # GPa

        ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs/prl2000_hugoniot_dft.csv'
        ref_df = pd.read_csv(ref_filename, delimiter=r"\s+")
        rho_ref_dft = ref_df['rho'].to_numpy()/rho0 # rho/rho0
        P_ref_dft = ref_df['p'].to_numpy() * 100 # GPa

        axes[0].errorbar(rho_hugoniot, P_hugoniot, xerr=rho_std_hugoniot, yerr=P_std_hugoniot, fmt='-o', color='r', capsize=2, label='Present', linewidth=0.7, markersize=4)
        axes[1].errorbar(rho_hugoniot, np.zeros(n_points), xerr=rho_std_hugoniot, fmt='-o', color='r', capsize=2, linewidth=0.7, markersize=4)

        axes[0].errorbar(rho_ref, P_ref, xerr=rho_err_ref, yerr=P_err_ref, fmt='-^', color='g', capsize=2, label='prl2000', linewidth=0.7, markersize=4)
        axes[0].plot(rho_ref_dft, P_ref_dft, '-.', color='lime', label='DFT-MD', linewidth=0.7)
        axes[0].errorbar(rho_ref_exp1, P_ref_exp1, xerr=rho_err_ref_exp1, yerr=P_err_ref_exp1, fmt='D', color='b', capsize=2, label='experiment1', linewidth=0.7, markersize=4)
        axes[0].errorbar(rho_ref_exp2, P_ref_exp2, xerr=rho_err_ref_exp2, yerr=P_err_ref_exp2, fmt='s', color='cyan', capsize=2, label='experiment2', linewidth=0.7, markersize=4)

        axes[0].set_xlabel(r'$\rho/\rho_0$', color=fontcolor)
        axes[0].set_ylabel(r'$P$ (GPa)', color=fontcolor)
        axes[0].set_xlim(2.6, 8.1)
        axes[0].set_facecolor(facecolor) 
        axes[0].spines['bottom'].set_color(fontcolor)
        axes[0].spines['top'].set_color(fontcolor)
        axes[0].spines['left'].set_color(fontcolor)
        axes[0].spines['right'].set_color(fontcolor)
        axes[0].tick_params(axis='both', colors=fontcolor)

        axes[0].axvline(eval_rho_ratio(1.86), color="yellow", linestyle='--', linewidth=0.5)
        axes[0].axvline(eval_rho_ratio(2), color="yellow", linestyle='--', linewidth=0.5)
        axes[1].axvline(eval_rho_ratio(1.86), color="yellow", linestyle='--', linewidth=0.5)
        axes[1].axvline(eval_rho_ratio(2), color="yellow", linestyle='--', linewidth=0.5)

        axes[1].set_xlabel(r'$\rho/\rho_0$', color=fontcolor)
        axes[1].set_ylabel(r'$H$ (Ha)', color=fontcolor)
        axes[1].axhline(0, color=fontcolor, linestyle='--', linewidth=0.5)
        axes[1].set_xlim(2.6, 8.1)
        axes[1].set_facecolor(facecolor) 
        axes[1].spines['bottom'].set_color(fontcolor)
        axes[1].spines['top'].set_color(fontcolor)
        axes[1].spines['left'].set_color(fontcolor)
        axes[1].spines['right'].set_color(fontcolor)
        axes[1].tick_params(axis='both', colors=fontcolor)

        fig.legend(loc='center right', ncol=1, facecolor=facecolor, edgecolor=fontcolor, labelcolor=fontcolor)
        plt.tight_layout(rect=[0, 0, 0.75, 1]) 

        if savefigname is not None:
            plt.savefig(savefigname+'.png', dpi=300)
            print(f"{GREEN}Figure saved as {savefigname}.png{RESET}")
