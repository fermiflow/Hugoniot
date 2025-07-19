import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

def make_exchange_func(exchange):
    """
        Make exchange functional.
        This is epsilon_x(rho), not E_x[rho(r)].
    """
    def lda_x(rho: float) -> float:
        """
            the LDA exchange functional Vxc(rho).
        """
        return -3/4*(3/jnp.pi)**(1/3)*rho**(1/3)

    if exchange == 'lda':
        return lda_x
    else:
        raise ValueError("exchange functional not supported yet.")

def make_correlation_func(correlation):
    """
        Make correlation functional.
        This is epsilon_c(rho), not E_c[rho(r)].
    """
    # Wigner-Seitz parameter
    ws = (3/4/jnp.pi)**(1/3)

    # PZ81 parameters
    pz81_A = 0.0311
    pz81_B = -0.048
    pz81_C = 0.002
    pz81_D = -0.0116
    pz81_gamma = -0.1423
    pz81_beta1 = 1.0529
    pz81_beta2 = 0.3334

    # VWN parameters
    vwn_A = 0.0621814
    vwn_y0 = -0.10498
    vwn_b = 3.72744
    vwn_c = 12.93532
    vwn_Yy0 = vwn_y0**2 + vwn_b * vwn_y0 + vwn_c
    vwn_Q = (4*vwn_c - vwn_b**2)**(1/2)
    vwn_c0 = vwn_A / 2
    vwn_c1 = vwn_c0 * 2 * vwn_b / vwn_Q
    vwn_c2 = vwn_c0 * vwn_b * vwn_y0 / vwn_Yy0
    vwn_c3 = vwn_c2 * 2 * (vwn_b + 2 * vwn_y0) / vwn_Q
    vwn_c4 = vwn_c1 - vwn_c3

    # HL71 parameters
    hl71_A = 21
    hl71_C = 0.045

    # BH72 parameters
    bh72_A = 30
    bh72_C = 0.0504

    # GL77 parameters
    gl77_A = 11.4
    gl77_C = 0.0666
        
    def zero(rho: float) -> float:
        """
            Not to use correlation.
        """
        return 0

    def pz81(rho: float) -> float:
        """
            Correlation of Perdew-Zunger (1981).
        """
        rs = ws/rho**(1/3) # Wigner-Seitz rs
        pz81_cor = (pz81_A * jnp.log(rs) + pz81_B + pz81_C * rs * jnp.log(rs) + pz81_D * rs) * jnp.heaviside(1-rs, 1) \
             + (pz81_gamma/(1 + pz81_beta1 * jnp.sqrt(rs) + pz81_beta2 * rs)) * jnp.heaviside(rs-1, 0)
        return pz81_cor

    def vwn(rho: float) -> float:
        """
            Correlation of Vosko-Wilkes-Nusiar, Can.J.Phys.58, 1200 (1980). 
        """
        rs = ws/rho**(1/3)
        y = jnp.sqrt(rs)
        Yy = y**2 + vwn_b * y + vwn_c
        atnp = jnp.arctan(vwn_Q / (2 * y + vwn_b))
        vwn_cor = vwn_c0 * jnp.log(y**2/Yy) + vwn_c4 * atnp - vwn_c2 * jnp.log((y-vwn_y0)**2/Yy)
        return vwn_cor

    def hl71(rho: float) -> float:
        """
            Correlation of Hedin-Lundqvist, J. Phys. C: Solid State Phys. 4 2064 (1971).
        """
        rs = ws/rho**(1/3)
        x = rs/hl71_A
        hl71_cor = -0.5 * hl71_C * ((1 + x**3) * jnp.log(1 + 1/x) + x/2 - x**2 - 1/3)
        return hl71_cor

    def bh72(rho: float) -> float:
        """
            Correlation of U.von.Barth and L.Hedin, J.Phys.C5, 1629 (1972)
        """
        rs = ws/rho**(1/3)
        x = rs/bh72_A
        bh72_cor = -0.5 * bh72_C * ((1 + x**3) * jnp.log(1 + 1/x) + x/2 - x**2 - 1/3)
        return bh72_cor
    
    def gl77(rho: float) -> float:
        """
            Correlation of Gunnarsson-Lundqvist, Phys.Rev.B 15,6006 (1977)
        """
        rs = ws/rho**(1/3)
        x = rs/gl77_A
        gl77_cor = -0.5 * gl77_C * ((1 + x**3) * jnp.log(1 + 1/x) + x/2 - x**2 - 1/3)
        return gl77_cor

    if correlation == '':
        return zero
    elif correlation == 'pz81' or correlation == 'pz':
        return pw91
    elif correlation == 'vwn':
        return vwn
    elif correlation == 'hl71' or correlation == 'hl':
        return hl71
    elif correlation == 'bh72' or correlation == 'bh':
        return bh72
    elif correlation == 'gl77' or correlation == 'gl':
        return gl77
    else:
        raise ValueError("correlation functional not supported yet.")