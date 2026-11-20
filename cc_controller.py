import numpy as np
from pyscf import gto, scf, cc, lib

import src.BSE_Helper as helper
from src.cc_BSE_spatialorb import CC_BSE_spatial
from src.cc_BSE_spinorb import CC_BSE_spin
from pyscf.data.nist import HARTREE2EV

np.set_printoptions(precision=10, suppress=True, linewidth=100000)


# label = 'H2'
# mol = gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 2.00",
#             basis='aug-cc-pVTZ',
#             spin=0,
#             symmetry=False,
#             unit="Bohr",verbose=0)

label = 'Be'
mol = gto.M(atom="Be 0.00000000 0.00000000 0.00000000",
            basis='aug-cc-pVTZ',
            spin=0,
            symmetry=False,
            unit="Bohr",verbose=0)

# label = 'H2O'
# mol = gto.M(
#     atom="""
#     O  0.000000  0.000000  0.117300 
#     H  0.000000  0.757200 -0.469200 
#     H  0.000000 -0.757200 -0.469200
#     """,
#     unit="Angstrom",
#     basis="aug-cc-pVTZ",
#     symmetry=False,
#     spin=0,
#     verbose=0
# )

print()
print(label)
print()

myhf = mol.HF().run() 
mycc = cc.BCCD(myhf,max_cycle = 200, conv_tol_normu=1e-8).run()
mo = mycc.mo_coeff
t2 = mycc.t2

n_occ_spatial = int(t2.shape[0])
n_vir_spatial = int(t2.shape[2])

print(f'nocc:{n_occ_spatial}, nvir:{n_vir_spatial}')
print()

#spin-free
singE_spa, tripE_spa, ext_fock, Hbaresing, Ht2sing, Hbaretrip, Ht2trip,_,_,_,_,_,_ = CC_BSE_spatial(mol,mo,t2,n_occ_spatial,n_vir_spatial)
print('CC-BSE in spin-free basis COMPLETED.')
print("Singlet energies (eV):", np.sort(singE_spa * HARTREE2EV)[:10])
print("Triplet energies (eV):", np.sort(tripE_spa * HARTREE2EV)[:10])
print()

def lowest_eigenvalues(H, n=5):
    N = H.shape[0] * H.shape[1]
    matrix = H.reshape(N, N)

    eig = np.linalg.eigvals(matrix)
    eig = np.real_if_close(eig)

    return np.sort(eig)[:n] * HARTREE2EV

print("Fock only:")
print(lowest_eigenvalues(ext_fock))

print("Fock + bare interaction, singlet:")
print(lowest_eigenvalues(ext_fock + Hbaresing))

print("Full CC-BSE singlets:")
print(lowest_eigenvalues(ext_fock + Hbaresing + Ht2sing))
print()

#spin
singE_spin, tripE_spin, allE_spin,_,_,_,_,_,_,_,_,_,_ = CC_BSE_spin(mol,mo,t2,n_occ_spatial,n_vir_spatial)
print('CC-BSE in spin basis COMPLETED.')
# print(np.sort(allE_spin * HARTREE2EV)[:20])
print("Singlet energies (eV):", np.sort(singE_spin * HARTREE2EV)[:10])
print("Triplet energies (eV):", np.sort(tripE_spin * HARTREE2EV)[:10])




































