import numpy as np
from pyscf import gto, scf, cc, lib

import src.BSE_Helper as helper
from src.cc_BSE_spatialorb import CC_BSE_spinfree
from src.cc_BSE_spinorb import CC_BSE_spin

np.set_printoptions(precision=10, suppress=True, linewidth=100000)
eV2au = 0.0367493


label = 'H2'
mol = gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 2.00",
            basis='aug-cc-pVTZ',
            spin=0,
            symmetry=False,
            unit="Bohr",verbose=0)

print()
print(label)
print()

myhf = mol.HF().run() 
mycc = cc.BCCD(myhf,max_cycle = 200, conv_tol_normu=1e-8).run()
mo = mycc.mo_coeff
t2 = mycc.t2

n_occ_spatial = int(t2.shape[0])
n_vir_spatial = int(t2.shape[2])
n_occ_spin = int(t2.shape[0]*2)
n_vir_spin = int(t2.shape[2]*2)

print()
print(f'nocc:{n_occ_spatial}, nvir:{n_vir_spatial}')
print()

#spin-free
selfener_occ_spa, selfener_vir_spa, fock_occ_spa, fock_vir_spa, se_occ_spa, se_vir_spa, hbse_eig, singE_spa, tripE_spa = \
 CC_BSE_spinfree(mol,mo,myhf,mycc,t2,label,eV2au,n_occ_spatial,n_vir_spatial,n_occ_spin,n_vir_spin)
print()
print('CC-BSE in spin-free basis COMPLETED.')
print()
print("Singlet energies (eV):", np.sort(singE_spa / eV2au)[:5])
print("Triplet energies (eV):", np.sort(tripE_spa / eV2au)[:5])

#spin
hbse,hbse_sing, hbse_trip, selfener_occ_spin, selfener_vir_spin, fock_occ_spin, fock_vir_spin, se_occ_spin, se_vir_spin, val, singE_spin, tripE_spin= \
  CC_BSE_spin(mol,mo,myhf,mycc,label,eV2au,n_occ_spatial,n_vir_spatial,n_occ_spin,n_vir_spin)
print()
print('CC-BSE in spin basis COMPLETED.')
print()
print(np.sort(val / eV2au)[:20])
print("Singlet energies (eV):", np.sort(singE_spin / eV2au)[:5])
print("Triplet energies (eV):", np.sort(tripE_spin / eV2au)[:5])


# def extract_spin_energies(hbse, no, nv):

#     # Spin-conserving blocks
#     A = hbse[:no, :nv, :no, :nv]
#     B = hbse[:no, :nv, no:, nv:]
#     C = hbse[no:, nv:, :no, :nv]
#     D = hbse[no:, nv:, no:, nv:]

#     # Singlet and M_S = 0 triplet
#     H_sing = 0.5 * (A + B + C + D)
#     H_trip0 = 0.5 * (A - B - C + D)

#     # Spin-flip triplet sectors
#     H_trip_plus = hbse[:no, nv:, :no, nv:]
#     H_trip_minus = hbse[no:, :nv, no:, :nv]

#     shape = (no * nv, no * nv)

#     H_sing = H_sing.reshape(shape)
#     H_trip0 = H_trip0.reshape(shape)
#     H_trip_plus = H_trip_plus.reshape(shape)
#     H_trip_minus = H_trip_minus.reshape(shape)

#     # Eigenvalues
#     E_sing = np.linalg.eigvals(H_sing)
#     E_trip0 = np.linalg.eigvals(H_trip0)
#     E_trip_plus = np.linalg.eigvals(H_trip_plus)
#     E_trip_minus = np.linalg.eigvals(H_trip_minus)

#     return E_sing, E_trip0, E_trip_plus, E_trip_minus

# E_sing, E_t0, E_tp, E_tm = extract_spin_energies(
#     hbse, n_occ_spatial, n_vir_spatial
# )

# H_full = hbse.reshape(
#     n_occ_spin * n_vir_spin,
#     n_occ_spin * n_vir_spin
# )

# E_full = np.linalg.eigvals(H_full)

# E_reconstructed = np.concatenate([
#     E_sing, E_t0, E_tp, E_tm
# ])

# print("Full eigenvalues:", len(E_full))
# print("Reconstructed eigenvalues:", len(E_reconstructed))

# print(
#     "Maximum difference:",
#     np.max(np.abs(
#         np.sort(E_full.real)
#         - np.sort(E_reconstructed.real)
#     ))
# )

# print(
#     "Triplet M=0 vs M=+1:",
#     np.max(np.abs(
#         np.sort(E_t0.real) - np.sort(E_tp.real)
#     ))
# )

# print(
#     "Triplet M=0 vs M=-1:",
#     np.max(np.abs(
#         np.sort(E_t0.real) - np.sort(E_tm.real)
#     ))
# )

# print(
#     "Spin-flip +1 vs -1:",
#     np.max(np.abs(
#         np.sort(E_tp.real) - np.sort(E_tm.real)
#     ))
# )




































