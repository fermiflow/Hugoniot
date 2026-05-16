#!/bin/bash
# Download Pople basis sets from PySCF

POPLE_BASIS=(
    "3-21G"
    "6-31G"
    "6-311G"
    "6-31Gs"
    "6-311Gs"
)

for basis in "${POPLE_BASIS[@]}"; do
    echo "Downloading $basis..."
    curl -s "https://raw.githubusercontent.com/pyscf/pyscf/master/pyscf/gto/basis/pople-basis/${basis}.dat" -o "hqc/basis/pople/${basis}-raw.dat"
    if [ -s "hqc/basis/pople/${basis}-raw.dat" ]; then
        echo "  Converting $basis..."
        python tools/convert_basis.py "hqc/basis/pople/${basis}-raw.dat" "hqc/basis/pople/${basis}.dat" "${basis}"
        rm "hqc/basis/pople/${basis}-raw.dat"
    else
        echo "  Failed to download $basis"
        rm -f "hqc/basis/pople/${basis}-raw.dat"
    fi
done

echo "Done!"
