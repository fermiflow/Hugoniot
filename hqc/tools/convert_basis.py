#!/usr/bin/env python3
"""
Convert PySCF NWChem format basis sets to CP2K format for HQC.

Source: PySCF (Apache License 2.0)
https://github.com/pyscf/pyscf
"""

import re
import sys


def parse_nwchem_basis(filename):
    """Parse NWChem format basis set file."""
    with open(filename, 'r') as f:
        lines = f.readlines()

    basis_sets = {}
    current_element = None
    current_shell = None
    current_data = []

    for line in lines:
        line = line.strip()

        # Skip comments and empty lines
        if not line or line.startswith('#') or line.startswith('BASIS'):
            continue

        # Check if this is an element line (starts with element symbol)
        parts = line.split()
        if len(parts) >= 2 and parts[0].isalpha() and len(parts[0]) <= 2:
            # Save previous shell if exists
            if current_element and current_shell and current_data:
                if current_element not in basis_sets:
                    basis_sets[current_element] = []
                basis_sets[current_element].append({
                    'shell': current_shell,
                    'data': current_data
                })

            current_element = parts[0]
            current_shell = parts[1]
            current_data = []
        else:
            # This is data line
            try:
                values = [float(x) for x in parts]
                if values:
                    current_data.append(values)
            except ValueError:
                continue

    # Save last shell
    if current_element and current_shell and current_data:
        if current_element not in basis_sets:
            basis_sets[current_element] = []
        basis_sets[current_element].append({
            'shell': current_shell,
            'data': current_data
        })

    return basis_sets


def convert_to_cp2k(basis_sets, basis_name):
    """Convert to CP2K format."""
    output = []

    for element in sorted(basis_sets.keys()):
        shells = basis_sets[element]

        output.append("#BASIS SET")
        output.append(f"{element} {basis_name}")
        output.append(f"  {len(shells)}")

        for shell in shells:
            shell_type = shell['shell']
            data = shell['data']
            n_primitives = len(data)

            # Determine l_min, l_max based on shell type
            if shell_type == 'S':
                l_min, l_max = 0, 0
                n_coeffs = 1
            elif shell_type == 'P':
                l_min, l_max = 1, 1
                n_coeffs = 1
            elif shell_type == 'SP':
                l_min, l_max = 0, 1
                n_coeffs = 2
            else:
                print(f"Warning: Unknown shell type {shell_type} for {element}")
                continue

            # Write shell header
            output.append(f"  1  {l_min}  {l_max}  {n_primitives}  {n_coeffs}")

            # Write data
            for row in data:
                # Format: exponent, coefficient(s)
                line = f"       {row[0]:15.10f}"
                for coeff in row[1:]:
                    line += f"  {coeff:15.10f}"
                output.append(line)

        output.append("#")

    return '\n'.join(output) + '\n'


def main():
    if len(sys.argv) != 4:
        print("Usage: python convert_basis.py <input_nwchem> <output_cp2k> <basis_name>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    basis_name = sys.argv[3]

    print(f"Converting {input_file} to CP2K format...")
    basis_sets = parse_nwchem_basis(input_file)

    print(f"Found basis sets for elements: {', '.join(sorted(basis_sets.keys()))}")

    cp2k_format = convert_to_cp2k(basis_sets, basis_name)

    with open(output_file, 'w') as f:
        f.write(cp2k_format)

    print(f"Saved to {output_file}")


if __name__ == '__main__':
    main()
