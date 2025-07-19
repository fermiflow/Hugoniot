import os
import csv
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
jax.config.update("jax_enable_x64", True)

from src.inference.utils import *
from src.inference.colors import default_colors
from src.inference.markers import markers
from src.checkpoint import load_data
from src.inference.parser import parse_filename
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

    directory = os.path.dirname(data_filename)
    cfg_filename = os.path.join(directory, 'config.yaml')
    if os.path.isfile(cfg_filename):
        cfg = OmegaConf.load(cfg_filename)
        n = cfg.num
        rs = cfg.rs
        T = cfg.T
        batchsize = cfg.batchsize
        acc_steps = cfg.acc_steps
    else:
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

def eval_df_hugoniot(df, axes=None):

    n = df['n']
    T = df['T']
    rs = df['rs']

    rho = eval_rho_ratio(rs)

    E_ave = df['etot']
    E_std = df['etot_err']
    P_ave = df['p']
    P_std = df['p_err']
    if 'ep_cov' in df:
        EP_cov = df['ep_cov']
    else:
        EP_cov = 0

    print(f"{MAGENTA}E (Ry):{RESET} {E_ave} ± {E_std}")
    print(f"{MAGENTA}P (GPa):{RESET} {P_ave} ± {P_std}")

    Ry2Ha = 0.5
    E_ave *= Ry2Ha
    E_std *= Ry2Ha
    EP_cov *= Ry2Ha

    H = eval_H(E_ave, P_ave, rs)
    H_std = eval_H_std(E_ave, E_std, P_ave, P_std, EP_cov, rs)
    print(f"{MAGENTA}H (Ha):{RESET} {H} ± {H_std}")
    
    if axes is not None:
        axes[0].errorbar(rho, P_ave, yerr=P_std, fmt='-o', capsize=2, markersize=4, linewidth=0.7, color=default_colors[0])
        axes[1].errorbar(rho, H, yerr=H_std, fmt='-o', capsize=4, markersize=4, color=default_colors[0])

    return H, H_std, E_ave, E_std, P_ave, P_std, EP_cov, n, T, rs, rho

def eval_rho_ratio(rs):
    """
    Evaluate the ratio of the density to the reference density rho0 for a given rs.
    Input:
        rs: float
    Output:
        rho/rho0
    """
    return rhoDunit/rs**3

def eval_rs_rho(rho):
    """
    Evaluate rs from the ratio of the density
    Input:
        rho/rho0
    Output:
        rs
    """
    return (rhoDunit/rho)**(1/3)

def eval_H(E, P, rs):
    """
    Evaluate the enthalpy in Ha for a given total energy, pressure, and rs.
    Input:
        E: total energy, unit: Ha
        P: pressure, unit: GPa
        rs: Wigner-Seitz radius, unit: Bohr
    Output:
        H: Hugoniot, unit: Ha
    """
    return E-E0+0.5*P*(Vunit*rs**3-V0)

def eval_H_std(E, E_std, P, P_std, EP_cov, rs):
    """
    Evaluate the standard deviation of enthalpy in Ha for a given total energy, pressure, and rs.
    Input:
        E: total energy, unit: Ha
        E_std: standard deviation of total energy, unit: Ha
        P: pressure, unit: GPa
        P_std: standard deviation of pressure, unit: GPa
        EP_cov: covariance of E and P, unit: Ha*GPa)
        rs: Wigner-Seitz radius, unit: Bohr
    Output:
        H_std: standard deviation of Hugoniot, unit: Ha
    """
    p_H_E = jax.grad(eval_H, argnums=0)(E, P, rs)
    p_H_P = jax.grad(eval_H, argnums=1)(E, P, rs)
    H_std = jnp.sqrt(p_H_E**2*E_std**2 + p_H_P**2*P_std**2 + 2*p_H_E*p_H_P*EP_cov)
    return H_std

def eval_rho_hugoniot(E1, E1_std, P1, P1_std, EP1_co, rs1, E2, E2_std, P2, P2_std, EP2_co, rs2):
    """
        Evaluate the standard deviation of rho/rho0 on Hugoniot curve using linear interpolation.
        Input:
            E1: total energy at point rs1, unit: Ha
            E1_std: standard deviation of total energy at point rs1 (rho1), unit: Ha
            P1: Pressure at point rs1, unit: GPa
            P1_std: standard deviation of Pressure at point rs1, unit: GPa
            EP1_co: covariance of E1 and P1, unit: Ha*GPa
            rs1: Wigner-Seitz radius, unit: Bohr
            E2: total energy at point rs2, unit: Ha
            E2_std: standard deviation of total energy at point rs2 (rho2), unit: Ha
            P2: Pressure at point rs2, unit: GPa
            P2_std: standard deviation of Pressure at point rs2, unit: GPa
            EP2_co: covariance of E2 and P2, unit: Ha*GPa
            rs2: Wigner-Seitz radius, unit: Bohr
        Output:
            rho_hugoniot: rho/rho0 on Hugoniot curve.
            rho_std_hugoniot: standard deviation of rho/rho0 on Hugoniot curve.
    """

    def eval_rho(e1, p1, e2, p2):
        h1 = eval_H(e1, p1, rs1)
        h2 = eval_H(e2, p2, rs2)
        rho1 = eval_rho_ratio(rs1)
        rho2 = eval_rho_ratio(rs2)
        rho_hugoniot = (h2*rho1-h1*rho2)/(h2-h1)
        return rho_hugoniot

    rho_hugoniot = eval_rho(E1, P1, E2, P2)

    p_rho_E1 = jax.grad(eval_rho, argnums=0)(E1, P1, E2, P2)
    p_rho_P1 = jax.grad(eval_rho, argnums=1)(E1, P1, E2, P2)
    p_rho_E2 = jax.grad(eval_rho, argnums=2)(E1, P1, E2, P2)
    p_rho_P2 = jax.grad(eval_rho, argnums=3)(E1, P1, E2, P2)
    
    rho_std_hugoniot = jnp.sqrt(p_rho_E1**2*E1_std**2 + p_rho_P1**2*P1_std**2 + 2*p_rho_E1*p_rho_P1*EP1_co \
                              + p_rho_E2**2*E2_std**2 + p_rho_P2**2*P2_std**2 + 2*p_rho_E2*p_rho_P2*EP2_co)

    return rho_hugoniot, rho_std_hugoniot

def eval_P_hugoniot(E1, E1_std, P1, P1_std, EP1_co, rs1, E2, E2_std, P2, P2_std, EP2_co, rs2):
    """
        Evaluate the standard deviation of pressure on Hugoniot curve using linear interpolation.
        Input:
            E1: total energy at point rs1, unit: Ha
            E1_std: standard deviation of total energy at point rs1 (rho1), unit: Ha
            P1: Pressure at point rs1, unit: GPa
            P1_std: standard deviation of Pressure at point rs1, unit: GPa
            EP1_co: covariance of E1 and P1, unit: Ha*GPa
            rs1: Wigner-Seitz radius, unit: Bohr
            E2: total energy at point rs2, unit: Ha
            E2_std: standard deviation of total energy at point rs2 (rho2), unit: Ha
            P2: Pressure at point rs2, unit: GPa
            P2_std: standard deviation of Pressure at point rs2, unit: GPa
            EP2_co: covariance of E2 and P2, unit: Ha*GPa
            rs2: Wigner-Seitz radius, unit: Bohr
        Output:
            P_hugoniot: pressure on Hugoniot curve, unit: GPa
            P_std_hugoniot: standard deviation of pressure on Hugoniot curve, unit: GPa
    """
    def eval_P(e1, p1, e2, p2):
        h1 = eval_H(e1, p1, rs1)
        h2 = eval_H(e2, p2, rs2)
        P_hugoniot = (h2*p1-h1*p2)/(h2-h1)
        return P_hugoniot

    P_hugoniot = eval_P(E1, P1, E2, P2)

    p_P_E1 = jax.grad(eval_P, argnums=0)(E1, P1, E2, P2)
    p_P_P1 = jax.grad(eval_P, argnums=1)(E1, P1, E2, P2)
    p_P_E2 = jax.grad(eval_P, argnums=2)(E1, P1, E2, P2)
    p_P_P2 = jax.grad(eval_P, argnums=3)(E1, P1, E2, P2)

    P_std_hugoniot = jnp.sqrt(p_P_E1**2*E1_std**2 + p_P_P1**2*P1_std**2 + 2*p_P_E1*p_P_P1*EP1_co \
                            + p_P_E2**2*E2_std**2 + p_P_P2**2*P2_std**2 + 2*p_P_E2*p_P_P2*EP2_co)

    return P_hugoniot, P_std_hugoniot

def lagrange_polynomial(x, xk, yk):
    """
        Lagrange polynomial interpolation.
    Input:
        x: float, input of lagrange interpolation function.
        xk: array of float, x coordinates of lagrange interpolation data points.
        yk: array of float, y coordinates of lagrange interpolation data points.
    Output:
        y: float, lagrange interpolation result.
    """
    if len(xk) != len(yk):
        raise ValueError("xk and yk must have the same length!")
    
    if len(np.unique(xk)) != len(xk):
        raise ValueError("There cannot be duplicate value in xk!")
    
    n = len(xk)
    y = 0.0
    for i in range(n):
        li = 1.0
        for j in range(n):
            if j != i:
                li *= (x - xk[j]) / (xk[i] - xk[j])
        y += yk[i] * li
    return y

def newton(func, x_init, tol=1e-7, maxiter=100):
    """
        Newton method to find the root of the function.
    Input:
        func: function, the function to find the root.
        x_init: float, initial guess of the root.
        tol: float, tolerance of the root.
        maxiter: int, maximum number of iterations.
    Output:
        x: float, the root of the function.
    """
    func1 = jax.grad(lambda x: (func(x))**2)
    func2 = jax.grad(func1)
    Newton_iter_func = lambda x: x - func1(x)/jnp.abs(func2(x))

    x = x_init
    for i in range(maxiter):
        x = Newton_iter_func(x)
        if jnp.abs(func(x)) < tol:
            break
    return x

def bisect(func, x_init, x_lo=0, x_hi=0, tol=1e-7, maxiter=100):
    """
        Bisection method to find the root of the function.
    Input:
        func: function, the function to find the root.
        x_init: float, initial guess of the root.
        tol: float, tolerance of the root.
        maxiter: int, maximum number of iterations.
    Output:
        x: float, the root of the function.
    """
    x = x_init
    if x_lo == 0:
        x_lo = x - 1.0
    if x_hi == 0:
        x_hi = x + 1.0

    for i in range(maxiter):
        x = (x_lo + x_hi) / 2
        if func(x) * func(x_lo) < 0:
            x_hi = x
        else:
            x_lo = x
        if jnp.abs(func(x)) < tol:
            break
    return x

def eval_rho_P_hugoniot_lagrange_interpolation(E, E_std, P, P_std, EP_cov, rs):
    """
        Evaluate rho and P on hugoniot curve by Lagrange interpolation (at one temperature).
    Input:
        E: array, energy.
        E_std: array, energy standard error.
        P: array, pressure.
        P_std: array, pressure standard error.
        EP_cov: array, EP covariance.
        rs: array, rs.
    Output:
        rho_hugoniot: float, rho of hugoniot curve (H=0).
        rho_std_hugoniot: float, rho standard error of hugoniot curve.
    """
    if len(E) != len(E_std):
        raise ValueError("E and E_std must have the same length!")
    if len(E) != len(P):
        raise ValueError("E and P must have the same length!")
    if len(E) != len(P_std):
        raise ValueError("E and P_std must have the same length!")
    if len(E) != len(EP_cov):
        raise ValueError("E and EP_cov must have the same length!")
    if len(E) != len(rs):
        raise ValueError("E and rs must have the same length!")

    rho = eval_rho_ratio(rs)
    H_lagrange_interpolation_func = lambda rho_x, rho_data, E_data, P_data: lagrange_polynomial(rho_x, rho_data, eval_H(E_data, P_data, rs))
    P_lagrange_interpolation_func = lambda rho_x, rho_data, P_data: lagrange_polynomial(rho_x, rho_data, P_data)
    H_rho_lagrange_interpolation_func = lambda rho_x: H_lagrange_interpolation_func(rho_x, rho, E, P)
    P_rho_lagrange_interpolation_func = lambda rho_x: P_lagrange_interpolation_func(rho_x, rho, P)

    # initial guess
    rho1 = jnp.min(rho)
    rho2 = jnp.max(rho)
    h1 = H_rho_lagrange_interpolation_func(rho1)
    h2 = H_rho_lagrange_interpolation_func(rho2)
    rho_init = (h2*rho1-h1*rho2)/(h2-h1)

    rho_hugoniot = newton(H_rho_lagrange_interpolation_func, rho_init)
    H_partial_rho = jax.grad(H_rho_lagrange_interpolation_func)(rho_hugoniot)
    H_partial_E = jax.jacrev(H_lagrange_interpolation_func, argnums=2)(rho_hugoniot, rho, E, P)
    H_partial_P = jax.jacrev(H_lagrange_interpolation_func, argnums=3)(rho_hugoniot, rho, E, P)
    rho_std_hugoniot = jnp.sqrt((jnp.dot(H_partial_E**2, E_std**2) + jnp.dot(H_partial_P**2, P_std**2) +
                                 2*jnp.dot(H_partial_E*H_partial_P, EP_cov)) / H_partial_rho**2)
    
    P_hugoniot = P_rho_lagrange_interpolation_func(rho_hugoniot)
    P_partial_rho = jax.grad(P_rho_lagrange_interpolation_func)(rho_hugoniot)
    P_partial_E = - P_partial_rho * H_partial_E / H_partial_rho
    P_partial_P = jax.jacrev(P_lagrange_interpolation_func, argnums=2)(rho_hugoniot, rho, P) - P_partial_rho * H_partial_P / H_partial_rho
    P_std_hugoniot = jnp.sqrt((jnp.dot(P_partial_E**2, E_std**2) + jnp.dot(P_partial_P**2, P_std**2) +
                               2*jnp.dot(P_partial_E*P_partial_P, EP_cov)))
    
    return rho_hugoniot, rho_std_hugoniot, P_hugoniot, P_std_hugoniot, H_rho_lagrange_interpolation_func, P_rho_lagrange_interpolation_func

def eval_H_file(file, silent_mode: bool = False):
    """
        Evaluate the Hugoniot from the given file.
    Input:
        file: str, file path
    Output:
        H: float, Hugoniot, unit: Ha
        H_std: float, standard deviation of Hugoniot, unit: Ha
        E_ave: float, average total energy, unit: Ha
        E_std: float, standard deviation of total energy, unit: Ha
        P_ave: float, average pressure, unit: GPa
        P_std: float, standard deviation of pressure, unit: GPa
        n: int, number of atoms
        T: float, temperature, unit: K
        rs: float, Wigner-Seitz radius, unit: Bohr
        rho: float, density ratio rho/rho0
    """

    # check if the files are existed
    index = file.index(".pkl")
    file_basename = os.path.basename(file)
    if "_quantity" not in file_basename:
        file = file[:index] + "_quantity" + ".pkl"
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File {file} not found.")
    else:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"File {file} not found.")
        
    if not silent_mode:
        print(f"{YELLOW}========= Load quantity data ========{RESET}")
    data = load_data(file)
    if not silent_mode:
        print(f"{MAGENTA}Loaded quantity data from:{RESET} {file}")
    
    n = data["n"]
    T = data["T"]
    rs = data["rs"]
    
    L = (4/3*jnp.pi*n)**(1/3)
    Kp = 1.5*T/157887.66 # proton kinetic energy in Ry
    rho = eval_rho_ratio(rs)

    E_ave, E_std = data["Etot_ave"], data["Etot_std"]
    P_ave, P_std = data["P_ave"], data["P_std"]
    EP_cov = data["EPtot_cov"]

    if not silent_mode:
        print(f"{MAGENTA}Loaded n:{RESET} {n}")
        print(f"{MAGENTA}Loaded T:{RESET} {T}")
        print(f"{MAGENTA}Loaded rs:{RESET} {rs}")
        print(f"{MAGENTA}Loaded rho:{RESET} {rho}")
        print(f"{MAGENTA}Loaded E (Ry):{RESET} {E_ave} ± {E_std}")
        print(f"{MAGENTA}Loaded P (GPa):{RESET} {P_ave} ± {P_std}")
        print(f"{MAGENTA}Loaded EP_cov (Ha*GPa):{RESET} {EP_cov}")

    if not silent_mode:
        print(f"{YELLOW}========= Add Ion Kinetic Energy ========{RESET}")
    E_ave += Kp
    P_ave += (2*Kp)/(3*(L*rs)**3)*14710.513242194795*n # GPa
    if not silent_mode:
        print(f"{MAGENTA}Added Ki (Ry):{RESET} {Kp}")
        print(f"{MAGENTA}Updated E (Ry):{RESET} {E_ave} ± {E_std}")
        print(f"{MAGENTA}Updated P (GPa):{RESET} {P_ave} ± {P_std}")

    if not silent_mode:
        print(f"{YELLOW}========= Convert Energy Unit ========{RESET}")
    Ry2Ha = 0.5
    E_ave *= Ry2Ha
    E_std *= Ry2Ha
    EP_cov *= Ry2Ha
    if not silent_mode:
        print(f"{MAGENTA}Converted E (Ha):{RESET} {E_ave} ± {E_std}")
    
    if not silent_mode:
        print(f"{YELLOW}========= Calculate Hugoniot ========{RESET}")
    H = eval_H(E_ave, P_ave, rs)
    H_std = eval_H_std(E_ave, E_std, P_ave, P_std, EP_cov, rs)
    if not silent_mode:
        print(f"{MAGENTA}H (Ha):{RESET} {H} ± {H_std}")

    return H, H_std, E_ave, E_std, P_ave, P_std, EP_cov, n, T, rs, rho

def hugoniot_point(files, 
                   savefigname: str = None, 
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

    H1, H1_std, E1_ave, E1_std, P1_ave, P1_std, EP1_cov, n1, T1, rs1, rho1 = eval_H_file(files[0], silent_mode=silent_mode)
    H2, H2_std, E2_ave, E2_std, P2_ave, P2_std, EP2_cov, n2, T2, rs2, rho2 = eval_H_file(files[1], silent_mode=silent_mode)

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
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 10), dpi=300, facecolor=facecolor)

        axes[0].errorbar([rho1, rho2], [P1_ave, P2_ave], yerr=[P1_std, P2_std], fmt='-o', capsize=2, markersize=4, linewidth=0.7, color=default_colors[1])
        axes[0].errorbar(rho_hugoniot, P_hugoniot, xerr=rho_std_hugoniot, yerr=P_std_hugoniot, fmt='-o', capsize=2, markersize=4, linewidth=0.7, color=default_colors[1])
        axes[0].set_xlabel(r'$\rho/\rho_0$', color=fontcolor)
        axes[0].set_ylabel(r'$P$ (GPa)', color=fontcolor)
        axes[0].axvline(rho_hugoniot, color=fontcolor, linestyle='--', linewidth=0.5)

        axes[0].set_xlim(2.6, 8.1)

        axes[0].set_facecolor(facecolor) 
        axes[0].spines['bottom'].set_color(fontcolor)
        axes[0].spines['top'].set_color(fontcolor)
        axes[0].spines['left'].set_color(fontcolor)
        axes[0].spines['right'].set_color(fontcolor)
        axes[0].tick_params(axis='both', colors=fontcolor)

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

        axes[0].errorbar(rho_ref, P_ref, xerr=rho_err_ref, yerr=P_err_ref, fmt='-^', color='g', capsize=2, label='prl2000', linewidth=0.7, markersize=4)
        axes[0].plot(rho_ref_dft, P_ref_dft, '-.', color='lime', label='DFT-MD', linewidth=0.7)
        axes[0].errorbar(rho_ref_exp1, P_ref_exp1, xerr=rho_err_ref_exp1, yerr=P_err_ref_exp1, fmt='D', color='b', capsize=2, label='experiment1', linewidth=0.7, markersize=4)
        axes[0].errorbar(rho_ref_exp2, P_ref_exp2, xerr=rho_err_ref_exp2, yerr=P_err_ref_exp2, fmt='s', color='cyan', capsize=2, label='experiment2', linewidth=0.7, markersize=4)

        axes[1].errorbar([rho1, rho2], [H1, H2], yerr=[H1_std, H2_std], fmt='-o', capsize=4, color=default_colors[1], markersize=4)
        axes[1].errorbar(rho_hugoniot, 0, xerr=rho_std_hugoniot, fmt='-o', capsize=4, color=default_colors[1], markersize=4)
        axes[1].set_xlabel(r'$\rho/\rho_0$', color=fontcolor)
        axes[1].set_ylabel(r'$H$ (Ha)', color=fontcolor)
        axes[1].axhline(0, color=fontcolor, linestyle='--', linewidth=0.5)
        axes[1].axvline(rho_hugoniot, color=fontcolor, linestyle='--', linewidth=0.5)

        axes[1].set_xlim(2.6, 8.1)

        axes[1].set_facecolor(facecolor) 
        axes[1].spines['bottom'].set_color(fontcolor)
        axes[1].spines['top'].set_color(fontcolor)
        axes[1].spines['left'].set_color(fontcolor)
        axes[1].spines['right'].set_color(fontcolor)
        axes[1].tick_params(axis='both', colors=fontcolor)
    
        plt.tight_layout()
        if savefigname is not None:
            plt.savefig(savefigname+'.png', dpi=300)
            print(f"{GREEN}Figure saved as {savefigname}.png{RESET}")

    return rho_hugoniot, rho_std_hugoniot, P_hugoniot, P_std_hugoniot, H1, H1_std, H2, H2_std, rho1, rho2, P1_ave, P1_std, P2_ave, P2_std

def hugoniot_point2(files, 
                    average_epoch: int = 10,
                    savefigname: str = None, 
                    dark_mode: bool = False, 
                    silent_mode: bool = False,
                    experiment: bool = True,
                    theory: bool = True,
    ):
    """
    Evaluate the Hugoniot point from the given files.
    Input:
        files: list of str
        average_epoch: int, number of epochs to average the data.
    Output:
        rho_hugoniot: float, rho/rho0 on Hugoniot curve.
        rho_std_hugoniot: float, standard deviation of rho/rho0 on Hugoniot curve.
        P_hugoniot: float, pressure on Hugoniot curve, unit: GPa
        P_std_hugoniot: float, standard deviation of pressure on Hugoniot curve, unit: GPa
    """
    n_file = len(files)
    if n_file < 2:
        raise ValueError("The number of files should be larger than 2.")

    H = np.zeros(n_file)
    H_std = np.zeros(n_file)
    E_ave = np.zeros(n_file)
    E_std = np.zeros(n_file)
    P_ave = np.zeros(n_file)
    P_std = np.zeros(n_file)
    EP_cov = np.zeros(n_file)
    n = np.zeros(n_file)
    T = np.zeros(n_file)
    rs = np.zeros(n_file)
    rho = np.zeros(n_file)

    for i_file in range(n_file):
        # H[i_file], H_std[i_file], E_ave[i_file], E_std[i_file], P_ave[i_file], P_std[i_file], EP_cov[i_file], \
        #     n[i_file], T[i_file], rs[i_file], rho[i_file] = eval_H_file(files[i_file], silent_mode=silent_mode)
        H[i_file], H_std[i_file], E_ave[i_file], E_std[i_file], P_ave[i_file], P_std[i_file], EP_cov[i_file], \
            n[i_file], T[i_file], rs[i_file], rho[i_file], _, _ = eval_data_hugoniot(files[i_file], average_epoch=average_epoch)

    if not np.all(n == n[0]):
        raise ValueError("The number of atoms should be the same.")
    if not np.all(T == T[0]):
        raise ValueError("The temperature should be the same.")

    if not silent_mode:
        print(f"{YELLOW}\n========= Calculate Hugoniot ========{RESET}")
        print(f"{MAGENTA}Loaded n:{RESET} {n[0]}")
        print(f"{MAGENTA}Loaded T:{RESET} {T[0]}")
        for i_file in range(n_file):
            print(f"{YELLOW}---------------------{RESET}")
            print(f"{MAGENTA}Loaded file:{RESET} {i_file}")
            print(f"{MAGENTA}Loaded rs{i_file+1}:{RESET} {rs[i_file]}")
            print(f"{MAGENTA}Loaded rho{i_file+1}:{RESET} {rho[i_file]}")
            print(f"{MAGENTA}Loaded E{i_file+1} (Ha):{RESET} {E_ave[i_file]} ± {E_std[i_file]}")
            print(f"{MAGENTA}Loaded P{i_file+1} (GPa):{RESET} {P_ave[i_file]} ± {P_std[i_file]}")
            print(f"{MAGENTA}Loaded H{i_file+1}:{RESET} {H[i_file]} ± {H_std[i_file]}")

    rho_hugoniot, rho_std_hugoniot, P_hugoniot, P_std_hugoniot, H_rho_lagrange_interpolation_func, P_rho_lagrange_interpolation_func \
        = eval_rho_P_hugoniot_lagrange_interpolation(E_ave, E_std, P_ave, P_std, EP_cov, rs)

    if not silent_mode:
        print(f"{YELLOW}----- Hugoniot Point -----{RESET}")
        print(f"{MAGENTA}Hugoniot rho/rho0:{RESET} {rho_hugoniot} ± {rho_std_hugoniot}")
        print(f"{MAGENTA}Hugoniot P (GPa):{RESET} {P_hugoniot} ± {P_std_hugoniot}")
    
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
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4.5, 7.2), dpi=300, sharex=True, facecolor=facecolor)

        if experiment:
            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Zmachine.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='D', markerfacecolor='none', markeredgecolor='sienna', color='sienna', capsize=2, label='Z-machine', linewidth=0.7, markersize=4, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Omega.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='X', markerfacecolor='none', markeredgecolor='darkslategray', color='darkslategray', capsize=2, label='Omega', linewidth=0.7, markersize=4, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Omega_reanalyzed.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='s', markerfacecolor='none', markeredgecolor='darkblue', color='darkblue', capsize=2, label='Omega, reanalyzed', linewidth=0.7, markersize=4, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Explosions.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='*', markerfacecolor='none', markeredgecolor='cadetblue', color='cadetblue', capsize=2, label='Explosions', linewidth=0.7, markersize=4, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Sano.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='o', markerfacecolor='none', markeredgecolor='darkgoldenrod', color='darkgoldenrod', capsize=2, label='Sano', linewidth=0.7, markersize=4, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Fernandez.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='P', markerfacecolor='none', markeredgecolor='purple', color='purple', capsize=2, label='Fernandez', linewidth=0.7, markersize=4, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Dick.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].plot(rho_ref_exp, P_ref_exp, marker='^', markerfacecolor='none', markeredgecolor='darkolivegreen', color='darkolivegreen', label='Dick', linewidth=0.7, markersize=4, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Nellis.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='1', markerfacecolor='none', markeredgecolor='darkmagenta', color='darkmagenta', capsize=2, label='Nellis', linewidth=0.7, markersize=4, alpha=0.4)

            ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Zmachine2017.csv'
            ref_df = pd.read_csv(ref_filename)
            rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            P_ref_exp = ref_df['p'].to_numpy() # GPa
            P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            zmachine = axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='D', markerfacecolor='none', markeredgecolor='slategray', color='slategray', capsize=2, label='Z-machine (Knudson2017)', linewidth=0.7, markersize=4, alpha=0.5)

            ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Fernandez.csv'
            ref_df = pd.read_csv(ref_filename)
            rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            P_ref_exp = ref_df['p'].to_numpy() # GPa
            P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            fernandez = axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='s', markerfacecolor='white', markeredgecolor='slategrey', color='slategrey', capsize=2, label='Laser (Fernandez2019)', linewidth=0.7, markersize=4, alpha=0.5)

        if theory:
            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs/prl2000_hugoniot_dft.csv'
            # ref_df = pd.read_csv(ref_filename, delimiter=r"\s+")
            # rho_ref_dft = ref_df['rho'].to_numpy()/rho0 # rho/rho0
            # P_ref_dft = ref_df['p'].to_numpy() * 100 # GPa
            # axes[0].plot(rho_ref_dft, P_ref_dft, '-.', color='darkgreen', label='DFT-MD', linewidth=0.7)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_SESAME.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_dft = ref_df['rho'].to_numpy() # rho/rho0
            # P_ref_dft = ref_df['p'].to_numpy() # GPa
            # axes[0].plot(rho_ref_dft, P_ref_dft, '-', color='BLACK', label='SESAME', linewidth=0.7, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_CEIMC.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_dft = ref_df['rho'].to_numpy() # rho/rho0
            # P_ref_dft = ref_df['p'].to_numpy() # GPa
            # axes[0].plot(rho_ref_dft, P_ref_dft, marker='o', markerfacecolor='none', markeredgecolor='darkred', linestyle='', label='CEIMC', linewidth=0.7)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs/prl2000_hugoniot.csv'
            # ref_df = pd.read_csv(ref_filename, delimiter=r"\s+")
            # ref_n = 4
            # rho_ref = ref_df['rho'].to_numpy()[-ref_n:]/rho0 # rho/rho0
            # rho_err_ref = ref_df['rho_err'].to_numpy()[-ref_n:]/rho0 # rho/rho0
            # P_ref = ref_df['p'].to_numpy()[-ref_n:] * 100 # GPa
            # P_err_ref = ref_df['p_err'].to_numpy()[-ref_n:] * 100 # GPa
            # axes[0].errorbar(rho_ref, P_ref, xerr=rho_err_ref, yerr=P_err_ref, marker='^', markerfacecolor='none', color='mediumblue', capsize=3, label='RPIMC', linewidth=0.7, markersize=4)

            ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_DFTMD.csv'
            ref_df = pd.read_csv(ref_filename)
            rho_ref_dftmd = ref_df['rho'].to_numpy() # rho/rho0
            P_ref_dftmd = ref_df['p'].to_numpy() # GPa
            dftmd, = axes[0].plot(rho_ref_dftmd, P_ref_dftmd, linestyle="-", markerfacecolor='none', color='darkblue', label='DFT-MD (Caillabet2011)', linewidth=0.8, markersize=4)

            ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_DFTFT_Knudson.csv'
            ref_df = pd.read_csv(ref_filename)
            rho_ref_dftft = ref_df['rho'].to_numpy() # rho/rho0
            P_ref_dftft = ref_df['p'].to_numpy() # GPa
            dftft, = axes[0].plot(rho_ref_dftft, P_ref_dftft, linestyle="-.", markerfacecolor='none', color='darkorchid', label='DFT-FT (Knudson2017)', linewidth=0.8, markersize=4)

            ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Filinov2005.csv'
            ref_df = pd.read_csv(ref_filename)
            rho_ref_pimc = ref_df['rho'].to_numpy() # rho/rho0
            P_ref_pimc = ref_df['p'].to_numpy() # GPa
            filinov2005, = axes[0].plot(rho_ref_pimc, P_ref_pimc, linestyle="-", marker="P", markeredgecolor='black', color='orchid', markeredgewidth=0.5, label='Direct PIMC (Filinov2005)', linewidth=0.8, markersize=4.5)

            ref_filename ='/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_CEIMC2.csv'
            ref_df = pd.read_csv(ref_filename)
            rho_ref = ref_df['rho'].to_numpy() # rho/rho0
            rho_err_ref = ref_df['rho_err'].to_numpy() # rho/rho0
            P_ref = ref_df['p'].to_numpy() # GPa
            P_err_ref = ref_df['p_err'].to_numpy() # GPa
            ceimc = axes[0].errorbar(rho_ref, P_ref, xerr=rho_err_ref, yerr=P_err_ref, marker='d', color='mediumaquamarine', markeredgecolor='black', markeredgewidth=0.5, capsize=3.5, label='CEIMC (Tubman2015)', linewidth=0.8, markersize=4.5)

            ref_filename ='/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_khairallah2011.csv'
            ref_df = pd.read_csv(ref_filename)
            rho_ref = ref_df['rho'].to_numpy() # rho/rho0
            rho_err_ref = ref_df['rho_err'].to_numpy() # rho/rho0
            P_ref = ref_df['p'].to_numpy() # GPa
            P_err_ref = ref_df['p_err'].to_numpy() # GPa
            rpimc = axes[0].errorbar(rho_ref, P_ref, xerr=rho_err_ref, yerr=P_err_ref, marker='v', linestyle='-', markeredgecolor='black', color='orange', markeredgewidth=0.5, capsize=3.5, label='RPIMC (Khairallah2011)', linewidth=0.8, markersize=4.5)

            ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs/prl2000_hugoniot.csv'
            ref_df = pd.read_csv(ref_filename, delimiter=r"\s+")
            ref_n = 5
            rho_ref = ref_df['rho'].to_numpy()[-ref_n:]/rho0 # rho/rho0
            rho_err_ref = ref_df['rho_err'].to_numpy()[-ref_n:]/rho0 # rho/rho0
            P_ref = ref_df['p'].to_numpy()[-ref_n:] * 100 # GPa
            P_err_ref = ref_df['p_err'].to_numpy()[-ref_n:] * 100 # GPa
            pimc = axes[0].errorbar(rho_ref, P_ref, xerr=rho_err_ref, yerr=P_err_ref, marker='^', color='cornflowerblue', markeredgecolor='black', markeredgewidth=0.5, capsize=3.5, label='RPIMC (Militzer2000)', linewidth=0.8, markersize=4.5)

        rho_min = np.min(rho)
        rho_max = np.max(rho)
        rho_mesh = np.linspace(rho_min, rho_max, 1000)
        H_rho = H_rho_lagrange_interpolation_func(rho_mesh)
        P_rho = P_rho_lagrange_interpolation_func(rho_mesh)

        vfe_raw0 = axes[0].errorbar(rho, P_ave, yerr=P_std, marker='s', color='salmon', markeredgecolor='black', markeredgewidth=0.5, capsize=3.5, label='Raw data', linestyle='', linewidth=1, markersize=4)
        axes[0].plot(rho_mesh, P_rho, linestyle='-', color='salmon')
        vfe_lag0 = axes[0].errorbar(rho_hugoniot, P_hugoniot, xerr=rho_std_hugoniot, yerr=P_std_hugoniot, marker='o', color='orangered', markeredgecolor='black', markeredgewidth=0.5, capsize=4.5, linewidth=1, markersize=5, label="Interpolation "+str(int(n_file))+" points")

        axes[0].set_xlabel(r'$\rho/\rho_0$', color=fontcolor)
        axes[0].set_ylabel(r'$P$ (GPa)', color=fontcolor)
        axes[0].axvline(rho_hugoniot, color=fontcolor, linestyle='--', linewidth=0.5)

        axes[0].set_facecolor(facecolor) 
        axes[0].spines['bottom'].set_color(fontcolor)
        axes[0].spines['top'].set_color(fontcolor)
        axes[0].spines['left'].set_color(fontcolor)
        axes[0].spines['right'].set_color(fontcolor)
        axes[0].tick_params(axis='both', colors=fontcolor)
        axes[0].tick_params(axis="y", direction="in", which='both', left=True, right=True)
        axes[0].tick_params(axis="x", direction="in", which='both', top=True, bottom=True)
        axes[0].set_ylim(0, 270)

        vfe_raw1 = axes[1].errorbar(rho, H, yerr=H_std, marker='s', color='salmon', markeredgecolor='black', markeredgewidth=0.5, capsize=3.5, label='Raw data', linestyle='', linewidth=1, markersize=4)
        axes[1].plot(rho_mesh, H_rho, linestyle='-', color='salmon')
        vfe_lag1 = axes[1].errorbar(rho_hugoniot, 0, xerr=rho_std_hugoniot, marker='o', color='orangered', markeredgecolor='black', markeredgewidth=0.5, capsize=4.5, linewidth=1, markersize=5, label="Interpolation "+str(int(n_file))+" points")
        axes[1].set_xlabel(r'$\rho/\rho_0$', color=fontcolor)
        axes[1].set_ylabel(r'$H$ (Ha)', color=fontcolor)
        axes[1].axhline(0, color=fontcolor, linestyle='--', linewidth=0.5)
        axes[1].axvline(rho_hugoniot, color=fontcolor, linestyle='--', linewidth=0.5)
        
        axes[1].set_facecolor(facecolor) 
        axes[1].spines['bottom'].set_color(fontcolor)
        axes[1].spines['top'].set_color(fontcolor)
        axes[1].spines['left'].set_color(fontcolor)
        axes[1].spines['right'].set_color(fontcolor)
        axes[1].tick_params(axis='both', colors=fontcolor)
        axes[1].tick_params(axis="y", direction="in", which='both', left=True, right=True)
        axes[1].tick_params(axis="x", direction="in", which='both', top=True, bottom=True)
        axes[1].set_xlim(2.4, 5.3)
        
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(left=0.18, right=0.94, top=0.99, bottom=0.07)
        if experiment:
            exp_legend = Legend(axes[0], 
                    [
                     zmachine, 
                     fernandez
                     ], 
                    [
                     'Z-machine (Knudson2017)', 
                     'Laser (Fernandez2019)'
                     ], 
                    alignment='left',
                    title='Experiment:', 
                    loc='upper left', 
                    bbox_to_anchor=(0.025, 0.98), 
                    title_fontsize=8,
                    fontsize=8,
                    frameon=False)
        if theory:
            calc_legend = Legend(axes[0], 
                     [dftmd, 
                      dftft, 
                      filinov2005,
                      rpimc,
                      ceimc, 
                      pimc], 
                     ['DFT-MD (Caillabet2011)', 
                      'DFT-FT (Knudson2017)', 
                      'Direct PIMC (Filinov2005)',
                      'RPIMC (Khairallah2011)',
                      'CEIMC (Tubman2015)', 
                      'RPIMC (Militzer2000)'], 
                     alignment='left',
                     title='Ab Initio Calculation:', 
                     loc='upper left', 
                     bbox_to_anchor=(0.025, 0.82),
                     title_fontsize=8, 
                     fontsize=8,
                     frameon=False)
            
        vfe_legend0 = Legend(axes[0], 
                     [vfe_raw0, vfe_lag0], 
                     ['Raw data', "Interpolation "+str(int(n_file))+" points"], 
                     alignment='left',
                     title='This work:', 
                     loc='upper left', 
                     bbox_to_anchor=(0.025, 0.46),
                     title_fontsize=8, 
                     fontsize=8,
                     frameon=False)

        vfe_legend1 = Legend(axes[1], 
                     [vfe_raw1, vfe_lag1], 
                     ['Raw data', "Interpolation "+str(int(n_file))+" points"], 
                     alignment='left',
                     title='This work:', 
                     loc='upper left', 
                     bbox_to_anchor=(0.025, 0.98),
                     title_fontsize=8, 
                     fontsize=8,
                     frameon=False)
        
        if experiment:
            axes[0].add_artist(exp_legend)
        if theory:
            axes[0].add_artist(calc_legend)
        axes[0].add_artist(vfe_legend0)
        axes[1].add_artist(vfe_legend1)

        if savefigname is not None:
            plt.savefig(savefigname, dpi=300)
            print(f"{GREEN}Figure saved as {savefigname}{RESET}")

    return rho_hugoniot, rho_std_hugoniot, P_hugoniot, P_std_hugoniot, H, H_std, rho, P_ave, P_std

def hugoniot_point_inf(csv_filename, T,
                       savefigname: str = None, 
                       dark_mode: bool = False, 
                       silent_mode: bool = False,
                       experiment: bool = True,
                       theory: bool = True,
                       energy_unit: str = 'Ry',
    ):
    """
    Evaluate the Hugoniot point from the given inference csv file.
    Input:
        csv_filename: str
    Output:
        rho_hugoniot: float, rho/rho0 on Hugoniot curve.
        rho_std_hugoniot: float, standard deviation of rho/rho0 on Hugoniot curve.
        P_hugoniot: float, pressure on Hugoniot curve, unit: GPa
        P_std_hugoniot: float, standard deviation of pressure on Hugoniot curve, unit: GPa
    """

    df = pd.read_csv(csv_filename)

    filtered_df = df[(df['T'] == T) & (df['rs'] >= 1.86) & (df['rs'] <= 2)]
    print(filtered_df)
    n_file = len(filtered_df)

    H = np.zeros(n_file)
    H_std = np.zeros(n_file)
    E_ave = np.zeros(n_file)
    E_std = np.zeros(n_file)
    P_ave = np.zeros(n_file)
    P_std = np.zeros(n_file)
    EP_cov = np.zeros(n_file)
    n = np.zeros(n_file)
    T = np.zeros(n_file)
    rs = np.zeros(n_file)
    rho = np.zeros(n_file)

    for i_file in range(n_file):
        H[i_file], H_std[i_file], E_ave[i_file], E_std[i_file], P_ave[i_file], P_std[i_file], EP_cov[i_file], \
            n[i_file], T[i_file], rs[i_file], rho[i_file]= eval_df_hugoniot(filtered_df.iloc[i_file])

    if not np.all(n == n[0]):
        raise ValueError("The number of atoms should be the same.")
    if not np.all(T == T[0]):
        raise ValueError("The temperature should be the same.")

    if not silent_mode:
        print(f"{YELLOW}\n========= Calculate Hugoniot ========{RESET}")
        print(f"{MAGENTA}Loaded n:{RESET} {n[0]}")
        print(f"{MAGENTA}Loaded T:{RESET} {T[0]}")
        for i_file in range(n_file):
            print(f"{YELLOW}---------------------{RESET}")
            print(f"{MAGENTA}Loaded file:{RESET} {i_file}")
            print(f"{MAGENTA}Loaded rs{i_file+1}:{RESET} {rs[i_file]}")
            print(f"{MAGENTA}Loaded rho{i_file+1}:{RESET} {rho[i_file]}")
            print(f"{MAGENTA}Loaded E{i_file+1} (Ha):{RESET} {E_ave[i_file]} ± {E_std[i_file]}")
            print(f"{MAGENTA}Loaded P{i_file+1} (GPa):{RESET} {P_ave[i_file]} ± {P_std[i_file]}")
            print(f"{MAGENTA}Loaded H{i_file+1}:{RESET} {H[i_file]} ± {H_std[i_file]}")

    rho_hugoniot, rho_std_hugoniot, P_hugoniot, P_std_hugoniot, H_rho_lagrange_interpolation_func, P_rho_lagrange_interpolation_func \
        = eval_rho_P_hugoniot_lagrange_interpolation(E_ave, E_std, P_ave, P_std, EP_cov, rs)

    if not silent_mode:
        print(f"{YELLOW}----- Hugoniot Point -----{RESET}")
        print(f"{MAGENTA}Hugoniot rho/rho0:{RESET} {rho_hugoniot} ± {rho_std_hugoniot}")
        print(f"{MAGENTA}Hugoniot P (GPa):{RESET} {P_hugoniot} ± {P_std_hugoniot}")
    
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
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4.5, 7.2), dpi=300, sharex=True, facecolor=facecolor)

        rho_min = np.min(rho)
        rho_max = np.max(rho)
        rho_mesh = np.linspace(rho_min, rho_max, 1000)
        H_rho = H_rho_lagrange_interpolation_func(rho_mesh)
        P_rho = P_rho_lagrange_interpolation_func(rho_mesh)

        if experiment:
            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Zmachine.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='D', markerfacecolor='none', markeredgecolor='sienna', color='sienna', capsize=2, label='Z-machine', linewidth=0.7, markersize=4, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Omega.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='X', markerfacecolor='none', markeredgecolor='darkslategray', color='darkslategray', capsize=2, label='Omega', linewidth=0.7, markersize=4, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Omega_reanalyzed.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='s', markerfacecolor='none', markeredgecolor='darkblue', color='darkblue', capsize=2, label='Omega, reanalyzed', linewidth=0.7, markersize=4, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Explosions.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='*', markerfacecolor='none', markeredgecolor='cadetblue', color='cadetblue', capsize=2, label='Explosions', linewidth=0.7, markersize=4, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Sano.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='o', markerfacecolor='none', markeredgecolor='darkgoldenrod', color='darkgoldenrod', capsize=2, label='Sano', linewidth=0.7, markersize=4, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Fernandez.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='P', markerfacecolor='none', markeredgecolor='purple', color='purple', capsize=2, label='Fernandez', linewidth=0.7, markersize=4, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Dick.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].plot(rho_ref_exp, P_ref_exp, marker='^', markerfacecolor='none', markeredgecolor='darkolivegreen', color='darkolivegreen', label='Dick', linewidth=0.7, markersize=4, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Nellis.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            # rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            # P_ref_exp = ref_df['p'].to_numpy() # GPa
            # P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            # axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='1', markerfacecolor='none', markeredgecolor='darkmagenta', color='darkmagenta', capsize=2, label='Nellis', linewidth=0.7, markersize=4, alpha=0.4)

            ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Zmachine2017.csv'
            ref_df = pd.read_csv(ref_filename)
            rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            P_ref_exp = ref_df['p'].to_numpy() # GPa
            P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            zmachine = axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='D', markerfacecolor='none', markeredgecolor='slategray', color='slategray', capsize=2, label='Z-machine (Knudson2017)', linewidth=0.7, markersize=4, alpha=0.5)

            ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Fernandez.csv'
            ref_df = pd.read_csv(ref_filename)
            rho_ref_exp = ref_df['rho'].to_numpy() # rho/rho0
            rho_err_ref_exp = ref_df['rho_err'].to_numpy() # rho/rho0
            P_ref_exp = ref_df['p'].to_numpy() # GPa
            P_err_ref_exp = ref_df['p_err'].to_numpy() # GPa
            fernandez = axes[0].errorbar(rho_ref_exp, P_ref_exp, xerr=rho_err_ref_exp, fmt='s', markerfacecolor='white', markeredgecolor='slategrey', color='slategrey', capsize=2, label='Laser (Fernandez2019)', linewidth=0.7, markersize=4, alpha=0.5)

        if theory:
            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs/prl2000_hugoniot_dft.csv'
            # ref_df = pd.read_csv(ref_filename, delimiter=r"\s+")
            # rho_ref_dft = ref_df['rho'].to_numpy()/rho0 # rho/rho0
            # P_ref_dft = ref_df['p'].to_numpy() * 100 # GPa
            # axes[0].plot(rho_ref_dft, P_ref_dft, '-.', color='darkgreen', label='DFT-MD', linewidth=0.7)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_SESAME.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_dft = ref_df['rho'].to_numpy() # rho/rho0
            # P_ref_dft = ref_df['p'].to_numpy() # GPa
            # axes[0].plot(rho_ref_dft, P_ref_dft, '-', color='BLACK', label='SESAME', linewidth=0.7, alpha=0.4)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_CEIMC.csv'
            # ref_df = pd.read_csv(ref_filename)
            # rho_ref_dft = ref_df['rho'].to_numpy() # rho/rho0
            # P_ref_dft = ref_df['p'].to_numpy() # GPa
            # axes[0].plot(rho_ref_dft, P_ref_dft, marker='o', markerfacecolor='none', markeredgecolor='darkred', linestyle='', label='CEIMC', linewidth=0.7)

            # ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs/prl2000_hugoniot.csv'
            # ref_df = pd.read_csv(ref_filename, delimiter=r"\s+")
            # ref_n = 4
            # rho_ref = ref_df['rho'].to_numpy()[-ref_n:]/rho0 # rho/rho0
            # rho_err_ref = ref_df['rho_err'].to_numpy()[-ref_n:]/rho0 # rho/rho0
            # P_ref = ref_df['p'].to_numpy()[-ref_n:] * 100 # GPa
            # P_err_ref = ref_df['p_err'].to_numpy()[-ref_n:] * 100 # GPa
            # axes[0].errorbar(rho_ref, P_ref, xerr=rho_err_ref, yerr=P_err_ref, marker='^', markerfacecolor='none', color='mediumblue', capsize=3, label='RPIMC', linewidth=0.7, markersize=4)

            ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_DFTMD.csv'
            ref_df = pd.read_csv(ref_filename)
            rho_ref_dftmd = ref_df['rho'].to_numpy() # rho/rho0
            P_ref_dftmd = ref_df['p'].to_numpy() # GPa
            dftmd, = axes[0].plot(rho_ref_dftmd, P_ref_dftmd, linestyle="-", markerfacecolor='none', color='darkblue', label='DFT-MD (Caillabet2011)', linewidth=0.8, markersize=4)

            ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_DFTFT_Knudson.csv'
            ref_df = pd.read_csv(ref_filename)
            rho_ref_dftft = ref_df['rho'].to_numpy() # rho/rho0
            P_ref_dftft = ref_df['p'].to_numpy() # GPa
            dftft, = axes[0].plot(rho_ref_dftft, P_ref_dftft, linestyle="-.", markerfacecolor='none', color='darkorchid', label='DFT-FT (Knudson2017)', linewidth=0.8, markersize=4)

            ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Filinov2005.csv'
            ref_df = pd.read_csv(ref_filename)
            rho_ref_pimc = ref_df['rho'].to_numpy() # rho/rho0
            P_ref_pimc = ref_df['p'].to_numpy() # GPa
            filinov2005, = axes[0].plot(rho_ref_pimc, P_ref_pimc, linestyle="", marker="P", markeredgecolor='black', color='orchid', markeredgewidth=0.5, label='Direct PIMC (Filinov2005)', linewidth=0.8, markersize=4.5)

            ref_filename ='/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_CEIMC2.csv'
            ref_df = pd.read_csv(ref_filename)
            rho_ref = ref_df['rho'].to_numpy() # rho/rho0
            rho_err_ref = ref_df['rho_err'].to_numpy() # rho/rho0
            P_ref = ref_df['p'].to_numpy() # GPa
            P_err_ref = ref_df['p_err'].to_numpy() # GPa
            ceimc = axes[0].errorbar(rho_ref, P_ref, xerr=rho_err_ref, yerr=P_err_ref, marker='d', color='mediumaquamarine', markeredgecolor='black', markeredgewidth=0.5, capsize=3.5, label='CEIMC (Tubman2015)', linewidth=0.8, markersize=4.5)

            ref_filename ='/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_Deltalearning_LRDMC.csv'
            ref_df = pd.read_csv(ref_filename)
            rho_ref = ref_df['rho'].to_numpy() # rho/rho0
            rho_err_ref = ref_df['rho_err'].to_numpy() # rho/rho0
            P_ref = ref_df['p'].to_numpy() # GPa
            P_err_ref = ref_df['p_err'].to_numpy() # GPa
            deltalearning = axes[0].errorbar(rho_ref, P_ref, xerr=rho_err_ref, yerr=P_err_ref, marker='s', color='teal', markeredgecolor='black', markeredgewidth=0.5, capsize=3.5, label=r'$\Delta$-learning(Tenti2024)', linestyle='', linewidth=1, markersize=4)

            ref_filename ='/home/lizh/hydrogen/hydrogen/main/analysis/refs_hugoniot/hugoniot_khairallah2011.csv'
            ref_df = pd.read_csv(ref_filename)
            rho_ref = ref_df['rho'].to_numpy() # rho/rho0
            rho_err_ref = ref_df['rho_err'].to_numpy() # rho/rho0
            P_ref = ref_df['p'].to_numpy() # GPa
            P_err_ref = ref_df['p_err'].to_numpy() # GPa
            rpimc = axes[0].errorbar(rho_ref, P_ref, xerr=rho_err_ref, yerr=P_err_ref, marker='v', linestyle='', markeredgecolor='black', color='darkslateblue', markeredgewidth=0.5, capsize=3.5, label='RPIMC (Khairallah2011)', linewidth=0.8, markersize=4.5)

            ref_filename = '/home/lizh/hydrogen/hydrogen/main/analysis/refs/prl2000_hugoniot.csv'
            ref_df = pd.read_csv(ref_filename, delimiter=r"\s+")
            ref_n = 5
            rho_ref = ref_df['rho'].to_numpy()[-ref_n:]/rho0 # rho/rho0
            rho_err_ref = ref_df['rho_err'].to_numpy()[-ref_n:]/rho0 # rho/rho0
            P_ref = ref_df['p'].to_numpy()[-ref_n:] * 100 # GPa
            P_err_ref = ref_df['p_err'].to_numpy()[-ref_n:] * 100 # GPa
            pimc = axes[0].errorbar(rho_ref, P_ref, xerr=rho_err_ref, yerr=P_err_ref, marker='^', color='cornflowerblue', markeredgecolor='black', markeredgewidth=0.5, capsize=3.5, label='RPIMC (Militzer2000)', linewidth=0.8, markersize=4.5)

        rho_min = np.min(rho)
        rho_max = np.max(rho)
        rho_mesh = np.linspace(rho_min, rho_max, 1000)
        H_rho = H_rho_lagrange_interpolation_func(rho_mesh)
        P_rho = P_rho_lagrange_interpolation_func(rho_mesh)

        vfe_raw0 = axes[0].errorbar(rho, P_ave, yerr=P_std, marker='s', color='salmon', markeredgecolor='black', markeredgewidth=0.5, capsize=3.5, label='Raw data', linewidth=1, markersize=4)
        axes[0].plot(rho_mesh, P_rho, linestyle='-', color='salmon')
        vfe_lag0 = axes[0].errorbar(rho_hugoniot, P_hugoniot, xerr=rho_std_hugoniot, yerr=P_std_hugoniot, marker='o', color='orangered', markeredgecolor='black', markeredgewidth=0.5, capsize=4.5, linewidth=1, markersize=5, label="Interpolation "+str(int(n_file))+" points")

        axes[0].set_xlabel(r'$\rho/\rho_0$', color=fontcolor)
        axes[0].set_ylabel(r'$P$ (GPa)', color=fontcolor)
        axes[0].axvline(rho_hugoniot, color=fontcolor, linestyle='--', linewidth=0.5)

        axes[0].set_facecolor(facecolor) 
        axes[0].spines['bottom'].set_color(fontcolor)
        axes[0].spines['top'].set_color(fontcolor)
        axes[0].spines['left'].set_color(fontcolor)
        axes[0].spines['right'].set_color(fontcolor)
        axes[0].tick_params(axis='both', colors=fontcolor)
        axes[0].tick_params(axis="y", direction="in", which='both', left=True, right=True)
        axes[0].tick_params(axis="x", direction="in", which='both', top=True, bottom=True)
        axes[0].set_ylim(0, 273)

        
        if energy_unit == 'Ry':
            vfe_raw1 = axes[1].errorbar(rho, H*2, yerr=H_std*2, marker='s', color='salmon', markeredgecolor='black', markeredgewidth=0.5, capsize=3.5, label='Raw data', linewidth=1, markersize=4)
            axes[1].plot(rho_mesh, H_rho * 2, linestyle='-', color='salmon')
        else:
            vfe_raw1 = axes[1].errorbar(rho, H, yerr=H_std, marker='s', color='salmon', markeredgecolor='black', markeredgewidth=0.5, capsize=3.5, label='Raw data', linewidth=1, markersize=4)
            axes[1].plot(rho_mesh, H_rho, linestyle='-', color='salmon')
        vfe_lag1 = axes[1].errorbar(rho_hugoniot, 0, xerr=rho_std_hugoniot, marker='o', color='red', markeredgecolor='black', markeredgewidth=0.5, capsize=4.5, linewidth=1, markersize=5, label="Interpolation "+str(int(n_file))+" points")
        axes[1].set_xlabel(r'$\rho/\rho_0$', color=fontcolor)
        axes[1].set_ylabel(r'$H$ ('+energy_unit+')', color=fontcolor)
        axes[1].axhline(0, color=fontcolor, linestyle='--', linewidth=0.5)
        axes[1].axvline(rho_hugoniot, color=fontcolor, linestyle='--', linewidth=0.5)
        
        axes[1].set_facecolor(facecolor) 
        axes[1].spines['bottom'].set_color(fontcolor)
        axes[1].spines['top'].set_color(fontcolor)
        axes[1].spines['left'].set_color(fontcolor)
        axes[1].spines['right'].set_color(fontcolor)
        axes[1].tick_params(axis='both', colors=fontcolor)
        axes[1].tick_params(axis="y", direction="in", which='both', left=True, right=True)
        axes[1].tick_params(axis="x", direction="in", which='both', top=True, bottom=True)
        axes[1].set_xlim(2.4, 5.3)

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(left=0.18, right=0.94, top=0.99, bottom=0.07)
        if experiment:
            exp_legend = Legend(axes[0], 
                    [
                     zmachine, 
                     fernandez
                     ], 
                    [
                     'Z-machine (Knudson2017)', 
                     'Laser (Fernandez2019)'
                     ], 
                    alignment='left',
                    title='Experiment:', 
                    loc='upper left', 
                    bbox_to_anchor=(0.025, 0.98), 
                    title_fontsize=8,
                    fontsize=8,
                    frameon=False)
        if theory:
            calc_legend = Legend(axes[0], 
                     [dftmd, 
                      dftft, 
                      filinov2005,
                      rpimc,
                      ceimc, 
                      deltalearning,
                      pimc], 
                     ['DFT-MD (Caillabet2011)', 
                      'DFT-FT (Knudson2017)', 
                      'Direct PIMC (Filinov2005)',
                      'RPIMC (Khairallah2011)',
                      'CEIMC (Tubman2015)', 
                      r'$\Delta$-learning (Tenti2024)',
                      'RPIMC (Militzer2000)'], 
                     alignment='left',
                     title='Ab Initio Calculation:', 
                     loc='upper left', 
                     bbox_to_anchor=(0.025, 0.82),
                     title_fontsize=8, 
                     fontsize=8,
                     frameon=False)
            
        vfe_legend0 = Legend(axes[0], 
                     [vfe_raw0, vfe_lag0], 
                     ['Raw data', "Interpolation "+str(int(n_file))+" points"], 
                     alignment='left',
                     title='This work:', 
                     loc='upper left', 
                     bbox_to_anchor=(0.025, 0.41),
                     title_fontsize=8, 
                     fontsize=8,
                     frameon=False)

        vfe_legend1 = Legend(axes[1], 
                     [vfe_raw1, vfe_lag1], 
                     ['Raw data', "Interpolation "+str(int(n_file))+" points"], 
                     alignment='left',
                     title='This work:', 
                     loc='upper left', 
                     bbox_to_anchor=(0.025, 0.98),
                     title_fontsize=8, 
                     fontsize=8,
                     frameon=False)
        
        if experiment:
            axes[0].add_artist(exp_legend)
        if theory:
            axes[0].add_artist(calc_legend)
        axes[0].add_artist(vfe_legend0)
        axes[1].add_artist(vfe_legend1)

        if savefigname is not None:
            plt.savefig(savefigname, dpi=300)
            print(f"{GREEN}Figure saved as {savefigname}{RESET}")

    return rho_hugoniot, rho_std_hugoniot, P_hugoniot, P_std_hugoniot, H, H_std, rho, P_ave, P_std
