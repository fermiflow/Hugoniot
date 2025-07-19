import jax
import jax.numpy as jnp

def make_occupation_func(n, n_mo, smearing=True, smearing_method='fermi', 
                         smearing_sigma=0., search_method='bisect', 
                         search_cycle=100, search_tol=1e-7):
    """
        Make occupation function. (closed-shell situation)
        Input:
            n: int, number of particles.
            n_mo: int, number of molecular orbitals.
            smearing: bool, whether to use smearing.
            smearing_method: str, smearing method, 'fermi' or 'gauss'.
            smearing_sigma: float, smearing sigma, Unit: Hartree.
            search_method: str, search method, 'bisect' or 'newton'.
        Output:
            occupation: occupation function.
    """
    assert n_mo >= n//2

    if smearing:
        assert smearing_method in ['fermi', 'gauss']

        def fermi_dirac_f_func(mu, w1):
            """
                Fermi-Dirac distribution f function. (closed-shell situation)
                    f_i = 2/(exp((w1_i-mu)/sigma)+1)
                Input:
                    mu: float, chemical potential, Unit: Hartree.
                    w1: array of shape (n_mo,), orbital energies, Unit: Hartree.
                Output:
                    f: array of shape (n_mo,), Fermi-Dirac distribution f function.
            """
            return 2/(jnp.exp((w1-mu)/smearing_sigma)+1)

        def gaussian_f_func(mu, w1):
            """
                Gaussing smearing f function. (closed-shell situation)
                    f_i = 1 + erf((mu-w1_i)/sigma)
                Input:
                    mu: float, chemical potential, Unit: Hartree.
                    w1: array of shape (n_ao,), orbital energies, Unit: Hartree.
                Output:
                    N_g: array of shape (n_ao,), Gaussian smearing f function.
            """
            return 0.5 * (1 + jax.lax.erf((mu-w1)/smearing_sigma))

        if smearing_method == 'fermi':
            f_func = fermi_dirac_f_func
        elif smearing_method == 'gauss':
            f_func = gaussian_f_func

        N_func = lambda mu, w1: jnp.sum(f_func(mu, w1))-n

        if search_method == 'bisect':
            # mu_shift is used to make sure:
            # 1. N_func(min(w1)+shift, w1) < 0
            # 2. N_func(max(w1)+shift, w1) > 0
            if smearing_method == 'fermi':
                mu_shift = -smearing_sigma * jnp.log(2*n_mo/n-1)
            elif smearing_method == 'gauss':
                mu_shift = -smearing_sigma * jax.lax.erf_inv(1-n/n_mo)

            def search_mu_bisect(w1, max_cycle=100, mu_tol=1e-7):
                """
                    Search chemical potential by bisection algorithm
                    Input:
                        w1: array of shape (n_mo,), orbital energies, Unit: Hartree.
                        N_func: N(mu)-n, find mu make N_func(mu) == 0.
                        max_cycle: int, maximum number of cycles.
                        mu_tol: float, tolerance.
                    Output:
                        mu: float, chemical potential, Unit: Hartree.
                """
                mu_lo = jnp.min(w1) + mu_shift
                mu_hi = jnp.max(w1) + mu_shift

                def body_fun(carry):
                    mu_lo, mu_hi, _, _, loop = carry
                    mu_mid = (mu_lo + mu_hi) / 2
                    y_mid = N_func(mu_mid, w1)
                    mu_lo = (y_mid < 0) * mu_mid + (y_mid >= 0) * mu_lo
                    mu_hi = (y_mid >= 0) * mu_mid + (y_mid < 0) * mu_hi
                    # ======================= debug =======================
                    # jax.debug.print("======= bisect =======")
                    # jax.debug.print("loop: {x}", x=loop)
                    # jax.debug.print("mu_lo:{x}, mu_hi:{y}", x=mu_lo, y=mu_hi)
                    # =====================================================
                    return mu_lo, mu_hi, N_func(mu_lo, w1), N_func(mu_hi, w1), loop+1
                
                def cond_fun(carry):
                    return (carry[2] < -mu_tol) * (carry[3] > mu_tol) * (carry[4] < max_cycle)
                    
                mu_lo, mu_hi, y_lo, y_hi, loop = jax.lax.while_loop(cond_fun, body_fun, (mu_lo, mu_hi, -1., 1., 0))

                return (mu_lo > -mu_tol) * mu_lo + (mu_hi < mu_tol) * mu_hi + (mu_lo + mu_hi)/2 * (loop == max_cycle)
            
            search_mu = search_mu_bisect

        elif search_method == 'newton':

            N2_func1 = jax.grad(lambda mu, w1: (N_func(mu, w1))**2)
            N2_func2 = jax.grad(N2_func1) # second derivative
            Newton_iter_func = lambda mu, w1: mu - N2_func1(mu, w1)/jnp.abs(N2_func2(mu, w1))

            def search_mu_newton(w1, max_cycle=10, mu_tol=1e-7):
                """
                    Search chemical potential by Newton minimization method.
                    Input:
                        w1: array of shape (n_ao,), orbital energies, Unit: Hartree.
                        Newton_iter_func: Newton iteration function.
                        max_cycle: int, maximum number of cycles.
                        mu_tol: float, tolerance.
                    Output:
                        mu: float, chemical potential, Unit: Hartree.
                """
                mu_init = w1[n//2] # initial guess

                def body_fun(carry):
                    _, mu, loop = carry
                    mu_new = Newton_iter_func(mu, w1)
                    # ======================= debug =======================
                    # jax.debug.print("======= newton =======")
                    # jax.debug.print("loop: {x}", x=loop)
                    # jax.debug.print("mu:{x}, mu_new:{y}", x=mu, y=mu_new)
                    # =====================================================
                    return mu, mu_new, loop+1
                
                def cond_fun(carry):
                    return jnp.abs(carry[1] - carry[0]) > mu_tol
                    
                _, mu, _ = jax.lax.while_loop(cond_fun, body_fun, (1., mu_init, 0))

                return mu
            
            search_mu = search_mu_newton

        def occupation_smearing(w1):
            """
                Occupation function with smearing. (closed-shell situation)
                    n_i = 2*f_i if i < n/2 else 0
                Input:
                    mu: float, chemical potential, Unit: Hartree.
                    w1: array of shape (n_mo,), orbital energies, Unit: Hartree.
                Output:
                    n: array of shape (n_mo,), occupation function.
            """
            mu = search_mu(w1, max_cycle=search_cycle, mu_tol=search_tol)
            return f_func(mu, w1)
        
        return occupation_smearing
    
    else:

        def occupation_nosmeasring(w1):
            """
                Occupation function without smearing. (closed-shell situation)
                    n_i = 2 if i < n/2 else 0
                Input:
                    mu: float, chemical potential, Unit: Hartree.
                    w1: array of shape (n_mo,), orbital energies, Unit: Hartree.
                Output:
                    n: array of shape (n_mo,), occupation function.
            """
            return 2 * jnp.concatenate((jnp.ones(n//2), jnp.zeros(n_mo-n//2)))
        
        return occupation_nosmeasring
