import jax
import numpy as np
import jax.numpy as jnp
from hqc.basis.parse import parse_quant_num, parse_gto, normalize_gto_coeff

def make_pbc_gto(basis: str, L: float, rcut: float = 24, 
                 gamma: bool = True, lcao_xyz: bool = False):
    """
        Make PBC GTO orbitals function by using cartesian representation.
        Args:
            basis: basis name, eg:'gth-szv'.
            L: float, unit cell length. (Unit: Bohr)
                Note: this L(in hqc.pbc.gto) = L * rs(in hqc.pbc.lcao)
            rcut: float, cutoff radius. (Unit: Bohr)
            gamma: bool, if True, return eval_pbc_ao(xp, xe) for gamma point only, 
                         else, return eval_pbc_ao_kpt(xp, xe, kpt) for a single k-point.
            lcao_xyz: bool, return the function for cartesian GTO orbitals.
                Note: This is an internal parameter. True is designed only for hqc.pbc.lcao function.
        Returns:
            eval_pbc_gto: PBC gto orbitals function.
    """
    element = 'H'

    # load basis parameters
    def load_gto(element, basis):
        """
            Load basis parameters from basis file.
        """
        gto_coeffs = parse_gto(element, basis)
        quant_num = parse_quant_num(element, basis)
        norm_gto_coeffs = normalize_gto_coeff(quant_num, gto_coeffs)
        num_sets = len(quant_num)

        l_min = 10
        l_max = 0
        for set_i in range(num_sets):
            if quant_num[set_i][1] < l_min:
                l_min = quant_num[set_i][1]
            if quant_num[set_i][2] > l_max:
                l_max = quant_num[set_i][2]
        return quant_num, norm_gto_coeffs, num_sets, l_min, l_max
    
    quant_num, norm_gto_coeffs, num_sets, l_min, l_max = load_gto(element, basis)
    if l_min != 0:
        raise ValueError("l_min != 0")

    # make 1d lattice for pbc summation
    tmax = rcut//L
    lattice_1d = jnp.arange(-tmax, tmax+1) * L

    def _make_alpha():
        """
            List all alpha of GTO orbitals in a 1d array.
            Returns:
                alpha: array of shape (n_all_alpha,)
        """
        all_alpha = jnp.array([])
        for set_i in range(num_sets):
            all_alpha = jnp.concatenate((all_alpha, norm_gto_coeffs[set_i][:, 0]), axis=0)
        return all_alpha
    
    all_alpha = _make_alpha() # (n_all_alpha,)

    def _eval_num_cart(l):
        """
            Return the number of cartesian functions for angular momentum l.
        Returns:
            num_cart: int
        """
        return (l+2)*(l+1)//2

    def _make_alpha_coeff_cart():
        """
            Returns:
                alpha_coeff_cart: array of shape (n_all_alpha, n_gto_cart+1)
                alpha and coefficients all GTO orbitals in a 2d array.
        """
        n_gto = 0
        num_alpha = np.zeros(num_sets, dtype=int)

        for set_i in range(num_sets):
            num_alpha[set_i] = norm_gto_coeffs[set_i].shape[0]
            l_1 = quant_num[set_i][1] # min l in set_i
            l_2 = quant_num[set_i][2] # max l in set_i
            n_copy = 0
            for l in range(l_1, l_2+1):
                i_1 = np.sum(quant_num[set_i][4:l-l_1+4])+1
                i_2 = np.sum(quant_num[set_i][4:l-l_1+5])+1
                for i in range(i_1, i_2):
                    column_to_copy = norm_gto_coeffs[set_i][:, i+n_copy:i+n_copy+1]
                    copy_num = _eval_num_cart(l)-1 # number of cartesians
                    copied_columns = np.tile(column_to_copy, (1, copy_num))
                    norm_gto_coeffs[set_i] = np.insert(norm_gto_coeffs[set_i], [i+n_copy+1]*copy_num, copied_columns, axis=1)
                    n_copy += copy_num
                    n_gto += copy_num+1

        alpha_coeff_cart = all_alpha.reshape(np.sum(num_alpha), 1)
        for set_i in range(0, num_sets):
            alpha_coeff_cart = np.hstack((alpha_coeff_cart, np.insert(np.zeros((np.sum(num_alpha)-num_alpha[set_i], 
                norm_gto_coeffs[set_i].shape[1]-1)), np.sum(num_alpha[:set_i]),
                norm_gto_coeffs[set_i][:, 1:], axis=0)))

        if n_gto+1 != alpha_coeff_cart.shape[1]:
            raise ValueError('n_gto != alpha_coeff_cart.shape[0]')

        return alpha_coeff_cart

    def _make_cart2gto(l_max):
        """
            Convert cartesian to gto orbitals.
            Returns:
                cart2gto: array of shape (n_cart, n_gto_cart)
        """
        num_cart = jax.vmap(_eval_num_cart)(np.arange(0, l_max+1)) # n_l
        cart2gto = np.zeros((np.sum(num_cart), 1))
        for set_i in range(num_sets):   
            l_1 = quant_num[set_i][1] # min l in set_i
            l_2 = quant_num[set_i][2] # max l in set_i
            for l in range(l_1, l_2+1):
                i_1 = np.sum(quant_num[set_i][4:l-l_1+4])+1
                i_2 = np.sum(quant_num[set_i][4:l-l_1+5])+1
                for i in range(i_1, i_2):
                    cart2gto = np.hstack((cart2gto, 
                        np.insert(np.zeros((np.sum(num_cart)-num_cart[l], num_cart[l])), 
                        np.sum(num_cart[:l]), np.identity(num_cart[l]), axis=0)))
        cart2gto = cart2gto[:, 1:]    
        return cart2gto
    
    def _make_alpha_coeff_gto_cart():
        """
            make alpha_coeff_gto_cart, array of shape (n_all_alpha, n_cart, n_gto_cart)
        """
        alpha_coeff_cart = _make_alpha_coeff_cart() # (n_all_alpha, n_gto_cart+1)
        cart2gto = _make_cart2gto(l_max) # (n_cart, n_gto_cart)
        alpha_coeff_gto_cart = jnp.einsum('ag,cg->acg', alpha_coeff_cart[:, 1:], cart2gto) # (n_all_alpha, n_cart, n_gto_cart)
        return alpha_coeff_gto_cart

    def _power2cart_x(l):
        """
            Return the transformation matrix from cartesian in one dimension x
            to three dimension x component.
            Transformation demo:
                          1 -> 1
                     (1, x) -> (x, 1, 1)
                (1, x, x^2) -> (x^2, x, x, 1, 1, 1)
                            ...
        Returns:
            power2cart_x: array of shape (l+1, (l+1)*(l+2)/2)
        """
        if l == 0:
            return np.array([[1]])
        elif l == 1:
            return np.array([[0,1,1],[1,0,0]])
        elif l == 2:
            return np.array([[0,0,0,1,1,1],[0,1,1,0,0,0],[1,0,0,0,0,0]])
        elif l == 3:
            return np.array([[0,0,0,0,0,0,1,1,1,1],[0,0,0,1,1,1,0,0,0,0],
                             [0,1,1,0,0,0,0,0,0,0],[1,0,0,0,0,0,0,0,0,0]])
        else:
            raise ValueError("l is not supported.")
        
    def _power2cart_y(l):
        """
            Return the transformation matrix from cartesian in one dimension y
            to three dimension y component.
            Transformation demo:
                          1 -> 1
                     (1, y) -> (1, y, 1)
                (1, y, y^2) -> (1, y, 1, y^2, y, 1)
                            ...
        Returns:
            power2cart_y: array of shape (l+1, (l+1)*(l+2)/2)
        """
        if l == 0:
            return np.array([[1]])
        elif l == 1:
            return np.array([[1,0,1],[0,1,0]])
        elif l == 2:
            return np.array([[1,0,1,0,0,1],[0,1,0,0,1,0],[0,0,0,1,0,0]])
        elif l == 3:
            return np.array([[1,0,1,0,0,1,0,0,0,1],[0,1,0,0,1,0,0,0,1,0],
                             [0,0,0,1,0,0,0,1,0,0],[0,0,0,0,0,0,1,0,0,0]])
        else:
            raise ValueError("l is not supported.")

    def _power2cart_z(l):
        """
            Return the transformation matrix from cartesian in one dimension z
            to three dimension z component.
            Transformation demo:
                          1 -> 1
                     (1, z) -> (1, 1, z)
                (1, z, z^2) -> (1, 1, z, 1, z, z^2)
                            ...
        Returns:
            power2cart_z: array of shape (l+1, (l+1)*(l+2)/2)
        """
        if l == 0:
            return np.array([[1]])
        elif l == 1:
            return np.array([[1,1,0],[0,0,1]])
        elif l == 2:
            return np.array([[1,1,0,1,0,0],[0,0,1,0,1,0],[0,0,0,0,0,1]])
        elif l == 3:
            return np.array([[1,1,0,1,0,0,1,0,0,0],[0,0,1,0,1,0,0,1,0,0],
                             [0,0,0,0,0,1,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]])
        else:
            raise ValueError("l is not supported.")

    def _make_power2cart():
        """
            Convert cartesian in one dimension to three dimension component.
            Returns:
                power2cart: array of shape (3, n_l, n_cart)
        """
        power2cart_x = np.zeros((l_max+1, 1))
        power2cart_y = np.zeros((l_max+1, 1))
        power2cart_z = np.zeros((l_max+1, 1))
        for l in range(l_min, l_max+1):
            power2cart_x_l = _power2cart_x(l)
            power2cart_y_l = _power2cart_y(l)
            power2cart_z_l = _power2cart_z(l)
            power2cart_x = np.hstack((power2cart_x, np.vstack((power2cart_x_l, np.zeros((l_max-l, power2cart_x_l.shape[1]))))))
            power2cart_y = np.hstack((power2cart_y, np.vstack((power2cart_y_l, np.zeros((l_max-l, power2cart_y_l.shape[1]))))))
            power2cart_z = np.hstack((power2cart_z, np.vstack((power2cart_z_l, np.zeros((l_max-l, power2cart_z_l.shape[1]))))))
        power2cart = jnp.array([power2cart_x, power2cart_y, power2cart_z])[..., 1:]
        return power2cart

    power2cart = _make_power2cart() # (3, n_pbc_gto_cart_x, n_gto_cart)

    def _cart2sph(l):
        """
            Return the transformation matrix from cartesian gto to
            spherical gto for angular momentum l.
            https://en.wikipedia.org/wiki/Table_of_spherical_harmonics#Real_spherical_harmonics
        Returns:
            cart2sph_l: array of shape ((l+1)*(l+2)/2, 2l+1)
        """
        if l == 0:
            return np.eye(1) / np.sqrt(4*np.pi)
        elif l == 1:
            return np.eye(3) * np.sqrt(3/(4*np.pi))
        elif l == 2:
            a = np.sqrt(3)
            return np.array([[0,0,-1/2,0,a/2],
                             [a,0,0,0,0],
                             [0,0,0,a,0],
                             [0,0,-1/2,0,-a/2],
                             [0,a,0,0,0],
                             [0,0,1,0,0]]) * np.sqrt(5/(4*np.pi))
        elif l == 3:
            a = np.sqrt(2)
            b = np.sqrt(3)
            c = np.sqrt(5)
            return np.array([[0,0,0,0,-a*b/4,0,a*c/4],
                             [3*a*c/4,0,-a*b/4,0,0,0,0],
                             [0,0,0,-1.5,0,b*c/2,0],
                             [0,0,0,0,-a*b/4,0,-3*a*c/4],
                             [0,b*c,0,0,0,0,0],
                             [0,0,0,0,a*b,0,0],
                             [-a*c/4,0,-a*b/4,0,0,0,0],
                             [0,0,0,-1.5,0,-b*c/2,0],
                             [0,0,a*b,0,0,0,0],
                             [0,0,0,1,0,0,0]]) * np.sqrt(7/(4*np.pi))
        else:
            raise ValueError("l is not supported.")

    def _make_cart2sph():
        """
            Convert cartesian gto to spherical gto.
            Returns:
                cart2sph: array of shape (n_gto_cart, n_gto_sph)
        """
        cart2sph = np.array([[0]])
        for set_i in range(num_sets):
            l_1 = quant_num[set_i][1] # min l in set_i
            l_2 = quant_num[set_i][2] # max l in set_i
            for l in range(l_1, l_2+1):
                i_1 = np.sum(quant_num[set_i][4:l-l_1+4])+1
                i_2 = np.sum(quant_num[set_i][4:l-l_1+5])+1
                for i in range(i_1, i_2):
                    cart2sph_rd = _cart2sph(l)
                    a, b = cart2sph.shape
                    c, d = cart2sph_rd.shape
                    zeros_ru = np.zeros((a, d))
                    zeros_ld = np.zeros((c, b))
                    cart2sph = np.block([[cart2sph, zeros_ru], [zeros_ld, cart2sph_rd]])
        return cart2sph[1:, 1:]

    def _make_alpha_coeff_gto_cart2sph():
        """
            Returns:
                alpha_coeff_gto_cart2sph: array of shape (n_all_alpha, n_cart, n_gto_sph)
        """
        alpha_coeff_gto_cart = _make_alpha_coeff_gto_cart()
        cart2sph = _make_cart2sph() # (n_gto_cart, n_gto_sph)
        alpha_coeff_gto_cart2sph = jnp.einsum('acg,gs->acs', alpha_coeff_gto_cart, cart2sph) # (n_all_alpha, n_cart, n_gto_sph)
        return alpha_coeff_gto_cart2sph
    
    alpha_coeff_gto_cart2sph = _make_alpha_coeff_gto_cart2sph() # (n_all_alpha, n_cart, n_gto_sph)

    def _make_power_x():
        """
            make the cartesian function for l_min and l_max.
            Input of each cartesian function is a float x.
        """
        def power_x(x):
            return jnp.power(x, np.arange(0, l_max+1))
        
        return power_x
    
    _power_x = _make_power_x()

    def _eval_gaussian(alpha, x):
        """
            Evaluates one s GTO orbital centered at 0 for one dimensional electron coordinate x.
                phi(r) = exp(-alpha * x^2)
        Args:
            x: float, position of one dimensional electron. (Unit: Bohr)
            alpha: float or array, alpha of GTO orbital.
        Returns:
            gaussian: float or array, GTO orbital at x
        """
        return jnp.exp(-alpha * x**2)

    def eval_pbc_gaussian_power_x(x: float):
        """
            Evaluates PBC cartesian GTO orbitals centered at 0 for 
            one dimensional electron coordinate x.
        Args:
            x: float, position of one dimensional electron. (Unit: Bohr)
        Returns:
            pbc_gaussian_power_x: PBC gaussian power orbitals at x, shape:(n_all_alpha, n_l)
                real float.
        """        
        power_x_T =  jax.vmap(_power_x, 0, 1)(x-lattice_1d) # (n_l, n_lattice_1d)
        gaussian_x_T = jax.vmap(_eval_gaussian, (None, 0), 1)(all_alpha, x-lattice_1d) # (n_all_alpha, n_lattice_1d)
        pbc_gaussian_power_x = jnp.einsum('lt,at->al', power_x_T, gaussian_x_T) # (n_all_alpha, n_l)
        return pbc_gaussian_power_x

    def eval_pbc_gto_sph(r):
        """
            Evaluates PBC GTO orbitals centered at (0, 0, 0) for one electron coordinate.
        Args:
            r: array of shape (3,), position of one electron. (Unit: Bohr)
        Returns:
            pbc_gto: PBC gto orbitals at xe, shape:(n_gto_sph,)
                n_gto_sph = n_gto_sph_s + n_gto_sph_p + n_gto_sph_d + ...
                real float.
        """
        pbc_gaussian_power_xyz = jax.vmap(eval_pbc_gaussian_power_x)(r) # (3, n_all_alpha, n_l)
        pbc_gaussian_cart_xyz = jnp.einsum('dal,dlc->dac', pbc_gaussian_power_xyz, power2cart) # (3, n_all_alpha, n_cart)
        pbc_gaussian_cart = jnp.prod(pbc_gaussian_cart_xyz, axis=0) # (n_all_alpha, n_cart)
        pbc_gto_sph = jnp.einsum('ac,acs->s', pbc_gaussian_cart, alpha_coeff_gto_cart2sph)
        return pbc_gto_sph

    def eval_pbc_ao(xp: jnp.ndarray, xe: jnp.ndarray):
        """
            Evaluates PBC GTO orbitals for several protons at one electron coordinate.
        Args:
            xp: array of shape (n, 3), position of protons in unit cell. (Unit: Bohr)
            xe: array of shape (3,), position one electron in unit cell. (Unit: Bohr)
        Returns:
            pbc_gto: PBC gto orbitals at xe, shape:(n_ao,)
                n_ao = n_p * (n_gto)
                n_gto = n_gto_s + n_gto_p + n_gto_d + ...
                real float.
        """        
        return jax.vmap(eval_pbc_gto_sph)(xe[None, :]-xp).reshape(-1)

    def eval_pbc_gaussian_power_x_kpt(x: float, kpt_x: float):
        """
            Evaluates PBC cartesian GTO orbitals centered at 0 for 
            one dimensional electron coordinate x.
        Args:
            x: float, position of one dimensional electron. (Unit: Bohr)
            kpt_x: float, k point in one axis. (Unit: 1/Bohr)
                1BZ: (-pi/L, pi/L)
        Returns:
            pbc_gaussian_power_x_kpt: PBC gaussian power orbitals at x and kpt_x, shape:(n_all_alpha, n_l)
                complex float.
        """        
        power_x_T =  jax.vmap(_power_x, 0, 1)(x-lattice_1d) # (n_l, n_lattice_1d)
        gaussian_x_T = jax.vmap(_eval_gaussian, (None, 0), 1)(all_alpha, x-lattice_1d) # (n_all_alpha, n_lattice_1d)
        pbc_gaussian_power_x_kpt = jnp.einsum('lt,at,t->al', power_x_T, gaussian_x_T, jnp.exp(1j*kpt_x*lattice_1d)) # (n_all_alpha, n_l), complex
        return pbc_gaussian_power_x_kpt

    def eval_pbc_gto_sph_kpt(r, kpt):
        """
            Evaluates PBC GTO orbitals centered at (0, 0, 0) for one electron coordinate.
        Args:
            r: array of shape (3,), position of one electron. (Unit: Bohr)
            kpt: array of shape (3,), k point. (Unit: 1/Bohr)
                1BZ: (-pi/L, pi/L)
        Returns:
            pbc_gto: PBC gto orbitals at xe and kpt, shape:(n_gto_sph,)
                n_gto_sph = n_gto_sph_s + n_gto_sph_p + n_gto_sph_d + ...
                complex float.
        """
        pbc_gaussian_power_xyz_kpt = jax.vmap(eval_pbc_gaussian_power_x_kpt, (0, 0), 0)(r, kpt) # (3, n_all_alpha, n_l), complex
        pbc_gaussian_cart_xyz_kpt = jnp.einsum('dal,dlc->dac', pbc_gaussian_power_xyz_kpt, power2cart) # (3, n_all_alpha, n_cart), complex
        pbc_gaussian_cart_kpt = jnp.prod(pbc_gaussian_cart_xyz_kpt, axis=0) # (n_all_alpha, n_cart), complex
        pbc_gto_sph_kpt = jnp.einsum('ac,acs->s', pbc_gaussian_cart_kpt, alpha_coeff_gto_cart2sph) # (n_gto_sph,), complex
        return pbc_gto_sph_kpt

    def eval_pbc_ao_kpt(xp: jnp.ndarray, xe: jnp.ndarray, kpt: jnp.ndarray):
        """
            Evaluates PBC GTO orbitals for several protons at one electron coordinate.
        Args:
            xp: array of shape (n, 3), position of protons in unit cell. (Unit: Bohr)
            xe: array of shape (3,), position one electron in unit cell. (Unit: Bohr)
            kpt: array of shape (3,), k point. (Unit: 1/Bohr)
                1BZ: (-pi/L, pi/L)
        Returns:
            pbc_gto_kpt: PBC gto orbitals at xe, shape:(n_ao,)
                n_ao = n_p * (n_gto)
                n_gto = n_gto_s + n_gto_p + n_gto_d + ...
                complex float.
        """        
        return jax.vmap(eval_pbc_gto_sph_kpt, (0, None), 0)(xe[None, :]-xp, kpt).reshape(-1) # (n_ao,), complex

    if gamma:
        if lcao_xyz:
            return jax.vmap(eval_pbc_gaussian_power_x), power2cart, alpha_coeff_gto_cart2sph
        else:
            return eval_pbc_ao
    else: 
        if lcao_xyz:
            return jax.vmap(eval_pbc_gaussian_power_x_kpt, (0, None), 0), power2cart, alpha_coeff_gto_cart2sph
        else:
            return eval_pbc_ao_kpt
        