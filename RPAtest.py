
import numpy as np
from scipy.linalg import eig, solve
from pyscf import gto, scf
from pyscf.tdscf import ghf
from pyscf.data.nist import HARTREE2EV

# Adjust imports to your repository structure
from src import BSE_Helper as helper
from src.cc_BSE_spinorb import build_hbse_spin

# ============================================================
# 1. Configuration
# ============================================================

BASIS = "aug-cc-pVTZ"  # Replace with your actual H2O basis
TOL = 1e-8


def max_error(X, Y):
    return np.max(np.abs(X - Y))


def check(name, X, Y, tol=TOL):
    error = max_error(X, Y)
    print(f"{name}: {error:.12e} Ha")
    if error > tol:
        raise AssertionError(f"{name} failed")
    return error


# ============================================================
# 2. Construct H2O and canonical HF reference
# ============================================================
label = 'H2'
mol = gto.M(atom="H 0.00 0.00 0.00; H 0.00 0.00 2.00",
            basis=BASIS,
            spin=0,
            symmetry=False,
            unit="Bohr",verbose=0)

# label = 'H2O'
# mol = gto.M(
#     atom="""
#     O  0.000000   0.000000   0.000000
#     H  0.000000   0.757160   0.586260
#     H  0.000000  -0.757160   0.586260
#     """,
#     basis=BASIS,
#     unit="Angstrom",
#     spin=0,
#     charge=0,
#     symmetry=False,
#     verbose=0
# )

mf = mol.HF().run()

assert mf.converged

# Convert RHF orbitals to generalized spin orbitals
mf_ghf = mf.to_ghf()

assert mf_ghf.converged

print("\nHF energy:", mf.e_tot)
print("GHF energy:", mf_ghf.e_tot)


# ============================================================
# 3. Extract spin orbitals in PySCF GHF ordering
# ============================================================

mo_occ = mf_ghf.mo_occ
mo_energy = mf_ghf.mo_energy
mo_coeff = mf_ghf.mo_coeff

occ_mask = mo_occ > 0
vir_mask = ~occ_mask

core_spinorbs = mo_coeff[:, occ_mask]
vir_spinorbs = mo_coeff[:, vir_mask]

eps_occ = mo_energy[occ_mask]
eps_vir = mo_energy[vir_mask]

no = len(eps_occ)
no_spatial=no / 2
nv = len(eps_vir)
nv_spatial = nv / 2
N = no * nv

print("\nOccupied spin orbitals:", no)
print("Virtual spin orbitals:", nv)
print("Excitation dimension:", N)

# These replace the extended Brueckner Fock matrices
F_ij = np.diag(eps_occ)
F_ab = np.diag(eps_vir)

print("F_ij:", F_ij.shape)
print("F_ab:", F_ab.shape)


# ============================================================
# 4. Obtain independent PySCF spin-orbital RPA matrices
# ============================================================

A4, B4 = ghf.get_ab(mf_ghf)

assert A4.shape == (no, nv, no, nv)
assert B4.shape == (no, nv, no, nv)

A = A4.reshape(N, N)
B = B4.reshape(N, N)

print("\nRPA A symmetry:", max_error(A, A.conj().T))
print("RPA B symmetry:", max_error(B, B.T))


# ============================================================
# 5. Construct HF spin-orbital integrals using BSE_helper
# ============================================================

# Your existing helper returns spin-AO integrals
_, eri_ao = helper.spinor_one_and_two_e_int(mol)

# Convert PySCF's AO integral storage to physicists' order
eri_phys = eri_ao.transpose(0, 2, 1, 3)

# Antisymmetrise the last two indices
anti_eri = eri_phys - eri_phys.swapaxes(2, 3)

# Crucial: use GHF orbitals directly.
# Do not call get_spinorbs() here.
oovv, _, _, ovvo, _ = helper.build_double_ints(
    core_spinorbs,
    vir_spinorbs,
    anti_eri
)

assert oovv.shape == (no, no, nv, nv)
assert ovvo.shape == (no, nv, nv, no)

print("\nHF oovv:", oovv.shape)
print("HF ovvo:", ovvo.shape)


# ============================================================
# 6. Validate PySCF RPA A and B against your integral helper
# ============================================================

H_gap4 = (
    np.einsum("ba,ij->iajb", F_ab, np.eye(no))
    - np.einsum("ij,ab->iajb", F_ij, np.eye(nv))
)

# Corrected CC-BSE bare interaction: <ib||aj>
H_bare4 = np.einsum("ibaj->iajb", ovvo)

# RPA coupling tensor: <ik||ac>
B_integrals4 = oovv.transpose(0, 2, 1, 3)

H_gap = H_gap4.reshape(N, N)
H_bare = H_bare4.reshape(N, N)
B_integrals = B_integrals4.reshape(N, N)

print("\n--- Independent integral checks ---")

check(
    "RPA A vs gap + bare",
    A,
    H_gap + H_bare
)

check(
    "RPA B vs oovv",
    B,
    B_integrals
)


# ============================================================
# 7. Solve full standard RPA
# ============================================================

R = np.block([
    [A, B],
    [-B.conj(), -A.conj()]
])

omega, vectors = eig(R)

positive = (
    (omega.real > TOL)
    & (np.abs(omega.imag) < TOL)
)

print("\nExpected positive RPA modes:", N)
print("Actual positive RPA modes:", positive.sum())

if positive.sum() != N:
    raise ValueError("Incomplete stable RPA spectrum")

omega_rpa = omega[positive].real
V = vectors[:, positive]

order = np.argsort(omega_rpa)
omega_rpa = omega_rpa[order]
V = V[:, order]


# ============================================================
# 8. Construct ring-CCD amplitudes
# ============================================================

X = V[:N, :]
Y = V[N:, :]

T = solve(X.T, Y.T).T

# General complex Riccati residual
riccati = (
    B.conj()
    + A.conj() @ T
    + T @ A
    + T @ B @ T
)

print("\nRiccati residual:",
      np.max(np.abs(riccati)))

# Convert particle-hole matrix to four-index ring amplitudes
# T[k,c,j,b] -> t2_ring[k,j,c,b]
t2_ring = T.reshape(
    no, nv, no, nv
).transpose(0, 2, 1, 3)

print("Ring amplitudes:", t2_ring.shape)


# ============================================================
# 9. Verify downfolded RPA
# ============================================================

H_down = A + B @ T

omega_down = np.linalg.eigvals(H_down)

if np.max(np.abs(omega_down.imag)) > TOL:
    raise ValueError("Complex downfolded RPA eigenvalues")

check(
    "RPA vs downfolded spectrum",
    np.sort(omega_rpa),
    np.sort(omega_down.real)
)


# ============================================================
# 10. Validate your existing CC-BSE Hamiltonian
# ============================================================

# The existing builder must use:
# H = F_ba delta_ij - F_ij delta_ab
#     + <ib||aj>
#     + sum_kc <ik||ac> t_kj^cb

_,_,_,hbse = build_hbse_spin(F_ij,F_ab,ovvo,oovv,t2_ring,no,nv)

H_test = np.asarray(hbse).reshape(N, N)

print("\n--- CC-BSE RPA-limit validation ---")

# Check the doubles contraction separately
H_ring4 = np.einsum(
    "ikac,kjcb->iajb",
    oovv,
    t2_ring,
    optimize=True
)

H_ring = H_ring4.reshape(N, N)

check(
    "Ring contraction vs B @ T",
    H_ring,
    B @ T
)

check(
    "Full CC-BSE vs downfolded RPA",
    H_test,
    H_down
)


# ============================================================
# 11. Compare complete eigenvalue spectra
# ============================================================

omega_test = np.linalg.eigvals(H_test)

if np.max(np.abs(omega_test.imag)) > TOL:
    raise ValueError("Complex CC-BSE RPA-limit roots")

check(
    "Complete excitation spectrum",
    np.sort(omega_test.real),
    omega_rpa
)

print("\nLowest 10 standard RPA energies (eV):")
print(omega_rpa[:10] * HARTREE2EV)

print("\nLowest 10 CC-BSE RPA-limit energies (eV):")
print(np.sort(omega_test.real)[:10] * HARTREE2EV)

print("\nALL RPA-LIMIT TESTS PASSED")
