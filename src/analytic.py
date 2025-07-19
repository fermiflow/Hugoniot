from mpmath import mpf, mp
mp.dps = 1200

def z_e(dim, L, beta, Emax=None):
    """
        The partition function and expected energy of a single particle in a cubic 
    box of dimension `dim` and size `L`, at inverse temperature `beta`.

        The infinite sum is truncated according to the energy cutoff `Emax`. When
    `Emax` takes the special value `None`, the infinite sum is evaluated exactly
    (to a given precision).
    """
    if Emax:
        from orbitals import sp_orbitals
        _, Es = sp_orbitals(dim, Emax)
        Es = [(2*mp.pi/L)**2 * E for E in Es]
        z = mp.fsum(mp.exp(-beta*E) for E in Es)
        e = mp.fsum(E*mp.exp(-beta*E) for E in Es) / z
    else:
        z_single_dim = mp.jtheta(3, 0, mp.exp(-beta * (2*mp.pi/L)**2))
        e_single_dim = mp.jtheta(3, 0, mp.exp(-beta * (2*mp.pi/L)**2), derivative=2) \
                            / (-4) * (2*mp.pi/L)**2 / z_single_dim
        z = z_single_dim**dim
        e = dim * e_single_dim
    return z, e

def Z_E(n, dim, L, beta, Emax=None):
    """
        The partition function and relevant thermodynamic quantities of `n` free
    (spinless) fermions in dimension `dim` and inverse-temperature `beta`, computed using
    recursion relations.

        The argument `Emax` determine the energy cutoff used in evaluating the
    single-particle partition function. See function "z_e" for details.
    """

    #print("L:", L, "\nbeta:", beta)

    zs, es = tuple(zip( *[z_e(dim, L, k*beta, Emax) for k in range(1, n+1)] )) 
    #print("zs:", zs)
    #print("es:", es)

    Zs = [mpf(1)]
    Es = [mpf(0)]
    for N in range(1, n+1):
        Z = mp.fsum( (-1)**(k-1) * zs[k-1] * Zs[N-k]
                     for k in range(1, N+1)
                   ) / N
        E = mp.fsum( (-1)**(k-1) * zs[k-1] * Zs[N-k] * (k * es[k-1] + Es[N-k])
                     for k in range(1, N+1)
                   ) / N / Z
        Zs.append(Z)
        Es.append(E)
    #print("Zs:", Zs)

    F = -mp.log(Zs[-1])/beta
    E = Es[-1]
    S = beta*(E - F)
    return F, E, S

if __name__ == "__main__":
    import numpy as np 
    import argparse
    parser = argparse.ArgumentParser(description="Finite-temperature ideal electron gas")

    # physical parameters.
    parser.add_argument("--n", type=int, default=16, help="total number of electrons")
    parser.add_argument("--rs", type=float, default=1.86, help="rs")
    parser.add_argument("--dim", type=int, default=3, help="spatial dimension")
    parser.add_argument("--T", type=float, default=10000, help="temperature in K")
    parser.add_argument("--Emax", type=int, default=10, help="energy cutoff for the single-particle orbitals")

    args = parser.parse_args()

    n, dim = args.n, args.dim
    assert (n%2==0)

    T, Emax = args.T, args.Emax
    Ef = 1.841584/args.rs**2 * 2   # Ef in Ry
    Theta =  (args.T/157888.088922572) / Ef
    
    print (args)
    print ('Theta', Theta)

    from orbitals import sp_orbitals
    _, Es = sp_orbitals(dim, args.Emax)
    num_states = Es.size
    from scipy.special import comb
    print("Total number of many-body states (%d in %d)^2: %f" % (n//2, num_states, comb(num_states, n//2)**2))

    if dim == 3:
        L = (4/3*np.pi*n)**(1/3)
        beta = 1 / ((2.25*np.pi)**(2/3) * Theta)
    elif dim == 2:
        L = np.sqrt(np.pi*n)
        beta = 1/ (2 * Theta)

    from orbitals import sp_orbitals
    sp_indices, Es = sp_orbitals(dim, Emax)
    sp_indices = np.array(sp_indices[::-1])
    Es = (2*np.pi/L)**2 * np.array(Es[::-1])

    from mpmath import mpf, mp
    from analytic import Z_E
    F, E, S = Z_E(n//2, dim, mpf(str(L)), mpf(str(beta)), Emax)
    print("F: %s, E: %s, S: %s" % (mp.nstr(2*F/n), mp.nstr(2*E/n), mp.nstr(2*S/n)))
