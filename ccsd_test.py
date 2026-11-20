import numpy as np
from pyscf import gto, cc
from pyscf.cc import eom_rccsd
from pyscf.data.nist import HARTREE2EV

mol = gto.M(
    atom="""
    O  0.000000   0.000000   0.000000
    H  0.000000   0.757160   0.586260
    H  0.000000  -0.757160   0.586260
    """,
    basis="cc-pVTZ",
    unit="Angstrom",
    spin=0,
    charge=0,
    verbose=0
)

mf = mol.HF().run()

mycc = cc.CCSD(mf)
mycc.conv_tol = 1e-9
mycc.conv_tol_normt = 1e-8
mycc.max_cycle = 200
mycc.kernel()

if not mycc.converged:
    raise RuntimeError("CCSD did not converge")

nroots = 10

E_sing, _ = mycc.eomee_ccsd_singlet(nroots=nroots)
E_trip, _ = mycc.eomee_ccsd_triplet(nroots=nroots)

print("Full EOM-CCSD singlets (eV):")
print(np.sort(np.atleast_1d(E_sing)) * HARTREE2EV)

print("Full EOM-CCSD triplets (eV):")
print(np.sort(np.atleast_1d(E_trip)) * HARTREE2EV)

def singles_block(mycc, spin="singlet"):
    no, _, nv, _ = mycc.t2.shape
    ns = no * nv

    if spin == "singlet":
        eom = eom_rccsd.EOMEESinglet(mycc)
    elif spin == "triplet":
        eom = eom_rccsd.EOMEETriplet(mycc)
    else:
        raise ValueError("Unknown spin")

    imds = eom.make_imds()
    size = eom.vector_size()

    J = np.zeros((ns, ns))

    for col in range(ns):
        vec = np.zeros(size)
        vec[col] = 1.0

        result = eom.matvec(vec, imds=imds)
        J[:, col] = result[:ns]

    return J


J_sing = singles_block(mycc, "singlet")
J_trip = singles_block(mycc, "triplet")

eig_sing = np.linalg.eigvals(J_sing)
eig_trip = np.linalg.eigvals(J_trip)

print("EOM-CCSD singles-block singlets (eV):")
print(np.sort(eig_sing.real)[:10] * HARTREE2EV)

print("EOM-CCSD singles-block triplets (eV):")
print(np.sort(eig_trip.real)[:10] * HARTREE2EV)