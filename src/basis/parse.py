import os
import numpy as np

def _find_basis_files(basis):
    """
        Find basis file in current directory.
        OUTPUT:
            basis_files: basis file name. (str)
    """
    file_name = basis + ".dat"
    current_directory = os.path.abspath(os.path.dirname(__file__))
    for root, dirs, files in os.walk(current_directory):
        for file in files:
            if file == file_name:
                return os.path.join(root, file)
    return None


def load_as_str(element, basis):
    """
        Load gto basis as a multi-line string from '.dat' files.
        INPUT:
            element: element name. (str)
                for us, "H" only.
            basis: basis name. (str)
        OUTPUT:
            basis_str: multi-line string of gto basis.
    """
    current_directory = os.path.abspath(os.path.dirname(__file__))
    file_name = _find_basis_files(basis)

    if file_name is None:
        raise FileNotFoundError(f"No '{file_name}' in '{current_directory}'.")

    with open(file_name, 'r') as f:
        basis_str = f.read()

    basis_str = basis_str.split("#BASIS SET\n")
    element_basis_str = None
    for elem in range(len(basis_str)):
        if basis_str[elem][:len(element)] == element:
            element_basis_str = "#BASIS SET\n"+basis_str[elem]
            break

    if element_basis_str is None:
        raise ValueError(f"No '{element}' in '{file_name}'.")

    return element_basis_str


def _basis_str_to_list(basis_str):
    """
        Convert multi-line string of gto basis to list.
        INPUT:
            basis_str: multi-line string of gto basis.
        OUTPUT:
            basis_list: list of str.
    """
    lines = basis_str.rsplit("\n")
    basis_list = [line.strip() for line in lines if line.strip()]
    return basis_list


def _parse_num_sets(basis_str):
    """
        Parse number of sets from gto basis.
        INPUT:
            basis_str: multi-line string of gto basis.
        OUTPUT:
            num_sets: number of sets. (int)
    """
    return int(_basis_str_to_list(basis_str)[2].strip())


def _string_to_array_int(input_string):
    """
        Convert string to numpy 1D int array.
        INPUT:
            input_string: string of numbers.
                Eg.  "  1  0  0  6  4"
        OUTPUT:
            integer_array: 1D array of integers.
                Eg. np.array([1, 0, 0, 6, 4])
    """
    words = input_string.split()
    integer_list = [int(word) for word in words]
    return np.array(integer_list)


def _string_to_array_float(string_list):
    """
        Convert list of strings to numpy 2D float array.
        INPUT:
            string_list: list of strings containing several floats.
        OUTPUT:
            float_array: 2D array of floats.
    """
    float_list = [list(map(float, line.split())) for line in string_list]
    float_array = np.array(float_list)
    return float_array


def parse_quant_num(element, basis):
    """
        Parse layer from gto basis.
    """
    basis_str = load_as_str(element, basis)
    num_sets = _parse_num_sets(basis_str)
    basis_str_list = _basis_str_to_list(basis_str)
    
    quant_num = []
    init_line = 3 # the first line of layer
    line = init_line
    for set_i in range(num_sets):

        if basis_str_list[line][:3] == "   ":
            raise ValueError(f"Line {basis_str_list[line]} has incorrect layer format.")

        layer = _string_to_array_int(basis_str_list[line])

        if len(layer) != 5+layer[2]-layer[1]:
            raise ValueError(f"Line {basis_str_list[line]} has incorrect number of l orbitals.")

        quant_num.append(layer)
        line += layer[3] + 1

    return quant_num


def parse_gto(element, basis):
    """
        Load gto basis coefficents from '.dat' files.
        INPUT:
            element: element name. (str)
                for us, "H" only.
            basis: basis name. (str)
        OUTPUT:
            gto_coeffs: list of numpy 2D float array.
                gto_coeffs[set_i] is the coefficents of set_i th set.
                gto_coeffs[set_i].shape = (quant_num[set_i][3], sum(quant_num[set_i][4:-1])+1)
    """
    basis_str = load_as_str(element, basis)
    num_sets = _parse_num_sets(basis_str)
    quant_num = parse_quant_num(element, basis)
    basis_str_list = _basis_str_to_list(basis_str)

    gto_coeffs = []
    init_line = 4 # the first line of coefficents
    line = init_line
    for set_i in range(num_sets):
        layer = quant_num[set_i]

        # set_i th coefficents, shape = (layer[3], sum(layer[4:-1])+1)
        coeff = _string_to_array_float(basis_str_list[line:line+layer[3]])

        if coeff.shape != (layer[3], np.sum(layer[4:])+1):
            raise ValueError(f"Line {basis_str_list[line]} has incorrect number of coefficents.")

        gto_coeffs.append(coeff)
        line += layer[3] + 1

    return gto_coeffs


def double_factorial(n):
    """
        Calculate double factorial.
        INPUT:
            n: integer.
        OUTPUT:
            double factorial of n.
    """
    if n <= 0:
        return 1
    else:
        return n * double_factorial(n-2)


def normalize_gto_coeff(quant_num, gto_coeffs):
    """
        Normalize the GTO coefficients.
        INPUT:
            quant_num: list of arrays, quant_num[set_i] is the quant_num of set_i th set.
                Eg. for gth-dzvp: quant_num = [array([1, 0, 0, 4, 2]), array([2, 1, 1, 1, 1])]
            coeff (unnormalized): list of arrays, alpha and coefficients of each set
        OUTPUT:
            coeff (normalized): list of arrays, alpha and coefficients of each set
    """
    # check the input
    if len(quant_num) != len(gto_coeffs):
        raise ValueError(f"wrong coefficients for {basis} basis")

    norm_gto_coeffs = []
    num_sets = len(quant_num)
    
    for set_i in range(num_sets):
        
        # check the input
        if gto_coeffs[set_i].shape != (quant_num[set_i][3], np.sum(quant_num[set_i][4:])+1):
            raise ValueError(f"wrong coefficients for {basis} basis")

        alpha = gto_coeffs[set_i][:, 0] # (n_contracted,)
        alpha_sum = alpha[:, None] + alpha[None, :] # (n_contracted, n_contracted)
        alpha_prod = alpha[:, None] * alpha[None, :] # (n_contracted, n_contracted)

        # initialize norm_gto_coeff
        norm_gto_coeff = gto_coeffs[set_i][:, 0:1] # (n_contracted, 1)

        l_min = quant_num[set_i][1]
        l_max = quant_num[set_i][2]
        for l in range(l_min, l_max+1):
            const = np.power(2, l+1.75)/np.power(np.pi, 0.25)/np.power(double_factorial(2*l+1), 0.5)
            i_min = np.sum(quant_num[set_i][4:l-l_min+4])+1
            i_max = np.sum(quant_num[set_i][4:l-l_min+5])+1
            for i in range(i_min, i_max):
                norm = np.power(np.power(2, l+1.5) * \
                        np.einsum('i,j,ij,ij', gto_coeffs[set_i][:, i],
                                                gto_coeffs[set_i][:, i],
                                                np.power(alpha_sum, -l-1.5),
                                                np.power(alpha_prod, l/2+0.75)), -0.5)
                coeff = gto_coeffs[set_i][:, i:i+1] * np.power(alpha, l/2+0.75)[:, None] * const * norm
                norm_gto_coeff = np.concatenate((norm_gto_coeff, coeff), axis=1)

        norm_gto_coeffs.append(norm_gto_coeff)

    return norm_gto_coeffs

if __name__ == "__main__":
    element = 'H'
    basis = "gth-1-dzvp"
    print(_find_basis_files(basis))
    basis_str = load_as_str(element, basis)
    print(basis_str)
    num_sets = _parse_num_sets(basis_str)
    print(num_sets)
    quant_num = parse_quant_num(element, basis)
    print(quant_num)
    gto_coeffs = parse_gto(element, basis)
    print(gto_coeffs)
    norm_gto_coeffs = normalize_gto_coeff(quant_num, gto_coeffs)
    print(norm_gto_coeffs)
