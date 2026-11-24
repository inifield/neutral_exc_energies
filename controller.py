import numpy as np
import sys
from pyscf import gto, cc, lib

from src.BSE_Helper import lowest_eigvals as eigvals
from src.cc_BSE_spatialorb import CC_BSE_spatial
from src.cc_BSE_spinorb import CC_BSE_spin
from src.molecules import molecules

np.set_printoptions(precision=10, suppress=True, linewidth=100000)

#Control basis set and BCCD tolerance
tol = 1e-8
BASIS = "aug-cc-pVTZ"

timings = {}

#Functions that control time measurings
def start_timer():
    return (
        lib.logger.process_clock(),
        lib.logger.perf_counter()
    )

def save_timer(label, start):
    cpu = lib.logger.process_clock() - start[0]
    wall = lib.logger.perf_counter() - start[1]

    timings[label] = (cpu, wall)


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
    basis = BASIS,
    verbose=0
)
print("========== SYSTEM INFORMATION ==========")
print(label, BASIS)
print()

#Perform BCCD and return the MO coefficients and doubles amplitudes
t0 = start_timer()
myhf = mol.HF().run() 
print("Hartree-Fock energy:", myhf.e_tot)
save_timer("HF", t0)

t0 = start_timer()
mycc = cc.BCCD(myhf, max_cycle = 200, conv_tol_normu=tol).run()
print("\nBCCD energy:", mycc.e_tot)
print("Correlation energy:", mycc.e_corr)
mo = mycc.mo_coeff
t2 = mycc.t2
save_timer("BCCD", t0)

#Number of occupied and virtual MOs
n_occ_spa = int(t2.shape[0])
n_vir_spa = int(t2.shape[2])

print()
print(f'nocc:{n_occ_spa}, nvir:{n_vir_spa}')
print()

#Performs spin-adapted (spatial) calculation and returns lowest singlet/triplet excitation energies as well as modified Koopman's IP/EAs
t0 = start_timer()
H_sing_spa,H_trip_spa,ext_fock, Hbaresing, Ht2sing, Hbaretrip, Ht2trip, F_ij, F_ab,_,_,selfener_occ,selfener_vir = CC_BSE_spatial(mol,mo,t2,n_occ_spa,n_vir_spa)
save_timer("CC-BSE",t0)

t0 = start_timer()
print("========== CC-BSE results ==========")
print("Ionisation potentials (eV):", eigvals(F_ij,10))
print("Electron affinities (eV):", eigvals(F_ab,10))
print("\nSinglet excitation energies (eV):", eigvals(H_sing_spa,10))
print("Triplet excitation energies (eV):", eigvals(H_trip_spa,10))

#Calculates eigenvalues of separate contributions to singlet and triplet BSE Hamiltonians
print("\nExtended Fock contribution (eV):", eigvals(ext_fock,10))

# print("Fock + bare interaction, singlet:")
# print(eigvals(ext_fock + Hbaresing))

# print("Full CC-BSE singlets:")
# print(eigvals(ext_fock + Hbaresing + Ht2sing))

# print("Fock + bare interaction, triplet:")
# print(eigvals(ext_fock + Hbaretrip))

# print("Full CC-BSE triplets:")
# print(eigvals(ext_fock + Hbaretrip + Ht2trip))
# print()

save_timer("Matrix diagonalisation",t0)

print("\n========== CALCULATION TIMINGS ==========")
print(f"{'Component':<32} {'CPU (s)':>12} {'Wall (s)':>12}")
print("-" * 64)
for label, (cpu, wall) in timings.items():
    print(f"{label:<32} {cpu:>12.6f} {wall:>12.6f}")

print("-" * 64)
print(f"{'Total time':<32} "
    f"{sum(x[0] for x in timings.values()):>12.6f} "
    f"{sum(x[1] for x in timings.values()):>12.6f}")

# #Performs spin-orbital calculation and returns lowest singlet/triplet excitation energies
# Hbse,H_sing_spin,H_trip_spin,_,_,_,_,_,_,_ = CC_BSE_spin(mol,mo,t2,n_occ_spa,n_vir_spa)
# print('CC-BSE in spin basis COMPLETED.')
# print("Singlet energies (eV):", eigvals(H_sing_spin,10))
# print("Triplet energies (eV):", eigvals(H_trip_spin,10))








































