import numpy as np
import sys
from pyscf import gto, cc, lib

from src.BSE_Helper import lowest_eigvals as eigvals
from src.cc_BSE_spatialorb import CC_BSE_spatial
from src.cc_BSE_spinorb import CC_BSE_spin
from src.molecules import molecules

np.set_printoptions(precision=10, suppress=True, linewidth=100000)


# Select molecule from terminal command
label = sys.argv[1] if len(sys.argv) > 1 else "H2"

if label not in molecules:
    raise ValueError(
        f"Unknown molecule '{label}'. "
        f"Available: {list(molecules)}"
    )

# Construct molecule and specify basis set
mol = gto.M(
    **molecules[label],
    basis = "aug-cc-pVTZ",
    verbose=0
)

print()
print(label)
print()

#Perform BCCD and return the MO coefficients and doubles amplitudes
myhf = mol.HF().run() 
mycc = cc.BCCD(myhf,max_cycle = 200, conv_tol_normu=1e-8).run()
mo = mycc.mo_coeff
t2 = mycc.t2

#Number of occupied and virtual MOs
n_occ_spa = int(t2.shape[0])
n_vir_spa = int(t2.shape[2])

print(f'nocc:{n_occ_spa}, nvir:{n_vir_spa}')
print()

#Performs spin-adapted (spatial) calculation and returns lowest singlet/triplet excitation energies
H_sing_spa,H_trip_spa,ext_fock, Hbaresing, Ht2sing, Hbaretrip, Ht2trip,_,_,_,_ = CC_BSE_spatial(mol,mo,t2,n_occ_spa,n_vir_spa)
print('CC-BSE in spin-free basis COMPLETED.')
print("Singlet energies (eV):", eigvals(H_sing_spa,10))
print("Triplet energies (eV):", eigvals(H_trip_spa,10))
print()

#Performs spin-orbital calculation and returns lowest singlet/triplet excitation energies
Hbse,H_sing_spin,H_trip_spin,_,_,_,_,_,_,_ = CC_BSE_spin(mol,mo,t2,n_occ_spa,n_vir_spa)
print('CC-BSE in spin basis COMPLETED.')
print("Singlet energies (eV):", eigvals(H_sing_spin,10))
print("Triplet energies (eV):", eigvals(H_trip_spin,10))

#Calculates eigenvalues of separate contributions to singlet and triplet BSE Hamiltonians
print("Fock only:")
print(eigvals(ext_fock))

print("Fock + bare interaction, singlet:")
print(eigvals(ext_fock + Hbaresing))

print("Full CC-BSE singlets:")
print(eigvals(ext_fock + Hbaresing + Ht2sing))

print("Fock + bare interaction, triplet:")
print(eigvals(ext_fock + Hbaretrip))

print("Full CC-BSE triplets:")
print(eigvals(ext_fock + Hbaretrip + Ht2trip))
print()






































