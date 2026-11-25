import numpy as np
import sys
from pyscf import gto, cc, lib
from pathlib import Path

from src.BSE_Helper import lowest_eigvals as eigvals
from src.cc_BSE_spatialorb import CC_BSE_spatial
from src.cc_BSE_spinorb import CC_BSE_spin

np.set_printoptions(precision=10, suppress=True, linewidth=100000)

#Ensures results are printed to terminal and output file
class Tee:
    def __init__(self, file):
        self.file = file
        self.terminal = sys.stdout

    def write(self, text):
        self.terminal.write(text)
        self.file.write(text)

    def flush(self):
        self.terminal.flush()
        self.file.flush()

#MAIN CONTROL PARAMETERS
max_cycle = 200
tol = 1e-8
BASIS = "aug-cc-pVTZ"

#Functions that control time measurings
def start_timer():
    return (
        lib.logger.process_clock(),
        lib.logger.perf_counter()
    )

def save_timer(label, start,timings):
    cpu = lib.logger.process_clock() - start[0]
    wall = lib.logger.perf_counter() - start[1]

    timings[label] = (cpu, wall)

labels = sorted((p.stem for p in Path("geometries").glob("*.xyz")),key=str.casefold)

# Select molecule from terminal command and return error if molecule not available
label = sys.argv[1] if len(sys.argv) > 1 else "beryllium"

if label == 'all':
    selected = labels
else:
    if label not in labels:
        sys.exit(f"Unknown molecule '{label}'. " f"Available: {labels}")
    selected = [label]


def perform_calculation(label,basis_set,tol,max_cycle):

    timings = {}
    xyz_file = Path("geometries") / f"{label}.xyz"

# Construct molecule and specify basis set
    mol = gto.M(
        atom=gto.fromfile(str(xyz_file)),
        basis = basis_set,
        unit = "Angstrom",
        spin=0,
        symmetry=False,
        verbose=0
    )

    print("========== SYSTEM INFORMATION ==========")
    print(label, basis_set)
    print()

    #Perform BCCD and return the MO coefficients and doubles amplitudes
    t0 = start_timer()
    myhf = mol.HF().run() 
    print("Hartree-Fock energy, Ha:", myhf.e_tot)
    save_timer("HF", t0, timings)

    t0 = start_timer()
    mycc = cc.BCCD(myhf, max_cycle = max_cycle, conv_tol_normu=tol).run()
    print("\nBCCD energy, Ha:", mycc.e_tot)
    print("Correlation energy, Ha:", mycc.e_corr)
    mo = mycc.mo_coeff
    t2 = mycc.t2
    save_timer("BCCD", t0, timings)

    #Number of occupied and virtual MOs
    n_occ_spa = int(t2.shape[0])
    n_vir_spa = int(t2.shape[2])

    print()
    print(f'nocc:{n_occ_spa}, nvir:{n_vir_spa}')
    print()

    #Performs spin-adapted (spatial) calculation and returns lowest singlet/triplet excitation energies as well as modified Koopman's IP/EAs
    t0 = start_timer()
    H_sing_spa,H_trip_spa,ext_fock, Hbaresing, Ht2sing, Hbaretrip, Ht2trip, F_ij, F_ab,_,_,selfener_occ,selfener_vir = CC_BSE_spatial(mol,mo,t2,n_occ_spa,n_vir_spa)
    save_timer("CC-BSE", t0, timings)

    t0 = start_timer()
    print("========== CC-BSE RESULTS ==========")
    print("Ionisation potentials (eV):", eigvals(F_ij,10))
    print("Electron affinities (eV):", eigvals(F_ab,10))

    print("\nExtended Fock contribution (eV):", eigvals(ext_fock,10))
    print("\nFock and bare contribution, singlet (eV):", eigvals(ext_fock+Hbaresing,10))
    print("Singlet excitation energies (eV):", eigvals(H_sing_spa,10))
    print("\nFock and bare contribution, triplet (eV):", eigvals(ext_fock+Hbaretrip,10))
    print("Triplet excitation energies (eV):", eigvals(H_trip_spa,10))

    save_timer("Matrix diagonalisation",t0, timings)

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

for label in selected:
    with open(f"results/{label}.txt","w",buffering=1) as output_file:
        original_stdout = sys.stdout
        sys.stdout = Tee(output_file)
        try:
            perform_calculation(label,BASIS,tol,max_cycle)
        finally:
            sys.stdout = original_stdout






































