
import numpy as np
import sys
from scipy.linalg import eig, solve
from pyscf import gto, scf
from pyscf.tdscf import ghf
from pyscf.data.nist import HARTREE2EV
from src.molecules import molecules

import src.BSE_Helper as helper
from src.cc_BSE_spinorb import build_hbse_spin

tol = 1e-8

def max_error(X, Y):
    return np.max(np.abs(X - Y))


def check(name, X, Y, tolerance=tol):
    error = max_error(X, Y)
    print(f"{name}: {error:.12e} Ha")
    if error > tolerance:
        raise AssertionError(f"{name} failed")
    return error

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

mf = mol.HF().run()

# Convert RHF orbitals to generalized spin orbitals
mf_ghf = mf.to_ghf()

print("\nHF energy:", mf.e_tot)
print("GHF energy:", mf_ghf.e_tot)

#Extract MO energies and orbital coefficients
mo_energy = mf_ghf.mo_energy
mo_coeff = mf_ghf.mo_coeff

#Extract dimensions of occupied and virtual matrices
no = mol.nelectron
nv = len(mo_energy) - no
N = no * nv

#Separate orbital coefficients and energies into occupied and virtual blocks
core_spinorbs = mo_coeff[:,:no]
vir_spinorbs = mo_coeff[:, no:]

eps_occ = mo_energy[:no]
eps_vir = mo_energy[no:]

print("\nOccupied spin orbitals:", no)
print("Virtual spin orbitals:", nv)

# Define the diagonal fock matrices to replace the extended Brueckner Fock matrices in CC-BSE functions
F_ij = np.diag(eps_occ)
F_ab = np.diag(eps_vir)

#Calculate relevant RPA quantities to construct RPA equations and verify their symmetry
A4, B4 = ghf.get_ab(mf_ghf)

A = A4.reshape(N, N)
B = B4.reshape(N, N)

print("\nRPA A symmetry:", max_error(A, A.conj().T))
print("RPA B symmetry:", max_error(B, B.T))

#Produce symmetric AOs with PySCF in chemist's notation
_, eri_ao = helper.spinor_one_and_two_e_int(mol)

# Convert to physicist's notation
eri_phys = eri_ao.transpose(0, 2, 1, 3)

# Construct antisymmetrised integrals
anti_eri_ao = eri_phys - eri_phys.swapaxes(2, 3)

#Build molecular antisymmetrised integrals for performing CC-BSE calculation
oovv, _, _, ovvo, _ = helper.build_double_ints(core_spinorbs, vir_spinorbs, anti_eri_ao)

#Extended Fock component of RPA Hamiltonian
H_gap4 = (np.einsum("ba,ij->iajb", F_ab, np.eye(no))
    - np.einsum("ij,ab->iajb", F_ij, np.eye(nv)))

# Corrected CC-BSE bare interaction: <ib||aj>
H_bare4 = np.einsum("ibaj->iajb", ovvo)

# RPA coupling tensor: <ik||ac>
B_integrals4 = oovv.transpose(0, 2, 1, 3)

H_gap = H_gap4.reshape(N, N)
H_bare = H_bare4.reshape(N, N)
B_integrals = B_integrals4.reshape(N, N)

print("\n--- Independent integral checks ---")

#Check the RPA A and B constructed from PySCF against their exact definitions in terms of molecular integrals
check("RPA A vs gap + bare", A, H_gap + H_bare)
check("RPA B vs oovv", B, B_integrals)

#Construct full RPA matrix and solve it for the eigenspectrum
R = np.block([
    [A, B],
    [-B.conj(), -A.conj()]
])

omega, vectors = eig(R)

#Construct tuple containing information on the number of expected positive eigenvalues and check against expected value
positive = (
    (omega.real > tol)
    & (np.abs(omega.imag) < tol)
)

print("\nExpected positive RPA modes:", N)
print("Actual positive RPA modes:", positive.sum())

if positive.sum() != N:
    raise ValueError("Incomplete stable RPA spectrum")

#Select out positive RPA eigenvalues and eigenvectors, then use them to construct ordered list of eigenvalues and eigenvectors
omega_rpa = omega[positive].real
V = vectors[:, positive]

order = np.argsort(omega_rpa)
omega_rpa = omega_rpa[order]
V = V[:, order]

#Construct forward and backward eigenvector sets X and Y
X = V[:N, :]
Y = V[N:, :]

#Use X and Y to find the ring amplitudes from T = Y X^-1
T = solve(X.T, Y.T).T

#Check these ring amplitudes satisfy the expected Riccati equation
riccati = (B.conj() + A.conj() @ T + T @ A + T @ B @ T)

print("\nRiccati residual:", np.max(np.abs(riccati)))

# Convert particle-hole matrix to four-index ring amplitudes - T[k,c,j,b] -> t2_ring[k,j,c,b]
t2_ring = T.reshape(no, nv, no, nv).transpose(0, 2, 1, 3)
print("Ring amplitudes:", t2_ring.shape)

#Construct downfolded RPA Hamiltonian
H_down = A + B @ T

#Find its eigenvalues and ensure they are real
omega_down = np.linalg.eigvals(H_down)

if np.max(np.abs(omega_down.imag)) > tol:
    raise ValueError("Complex downfolded RPA eigenvalues")

#Check the downfolded eigenvalues against the original RPA eigenvalues to ensure correct construction
check("RPA vs downfolded spectrum", omega_rpa, np.sort(omega_down.real))

#Construct the downfolded Hamiltonian from the CC-BSE functions with the replaced F_ij, F_ab and ring t2 amplitudes from RPA
_,_,_,hbse = build_hbse_spin(F_ij,F_ab,ovvo,oovv,t2_ring,no,nv)

H_test = np.asarray(hbse).reshape(N, N)

print("\n--- CC-BSE RPA-limit validation ---")

#Compare CC-BSE RPA limit with independently constructed downfolded RPA Hamiltonian
check("Full CC-BSE vs downfolded RPA", H_test, H_down)

#Calculate eigenvalues of CC-BSE RPA limit Hamiltonian, check if real and compare to those from independent RPA calculation
omega_test = np.linalg.eigvals(H_test)

if np.max(np.abs(omega_test.imag)) > tol:
    raise ValueError("Complex CC-BSE RPA-limit roots")

check("Complete excitation spectrum", np.sort(omega_test.real), omega_rpa)

#Display lowest eigenvalues of both methods
print("\nLowest 10 standard RPA energies (eV):")
print(omega_rpa[:10] * HARTREE2EV)

print("\nLowest 10 CC-BSE RPA-limit energies (eV):")
print(np.sort(omega_test.real)[:10] * HARTREE2EV)

