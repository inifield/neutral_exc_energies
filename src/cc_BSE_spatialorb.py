import numpy as np
import src.BSE_Helper as helper
from scipy.linalg.lapack import dgeev


def get_selfenergy_spatial(t2_spa, goovv):
    selfener_occ = (2*np.einsum("ikab,jkab->ij", goovv, t2_spa)
        - np.einsum("ikab,jkba->ij", goovv, t2_spa))

    selfener_vir = -(2*np.einsum("ijbc,ijac->ab", goovv, t2_spa)
        - np.einsum("ijbc,ijca->ab", goovv, t2_spa))
    
    return selfener_occ, selfener_vir


def build_hbse_singtrip(F_ij, F_ab, t2_spa, govvo, govov, goovv,n_occ_spa,n_vir_spa):
    # Corrected extended Fock contribution
    ext_fock = (np.einsum("ba,ij->iajb", F_ab, np.identity(n_occ_spa))
        - np.einsum("ij,ab->iajb", F_ij, np.identity(n_vir_spa)))

    # Corrected bare interaction
    J = np.einsum("ibaj->iajb", govvo)
    K = np.einsum("ibja->iajb", govov)

    # Doubles contribution after spin summation
    P = (2 * np.einsum("ikac,kjcb->iajb", goovv, t2_spa, optimize='optimal')
        - np.einsum("ikac,kjbc->iajb", goovv, t2_spa, optimize='optimal')
        - np.einsum("ikca,kjcb->iajb", goovv, t2_spa, optimize='optimal'))

    Q = np.einsum("ikca,kjbc->iajb", goovv, t2_spa, optimize='optimal')

    # Corrected spin-adapted Hamiltonians
    hbse_sing = ext_fock + 2*J - K + 2*P + Q
    hbse_trip = ext_fock - K + Q

    Hbaresing = 2*J-K
    Ht2sing = 2*P+Q
    Hbaretrip = -K
    Ht2trip = Q

    shape = (n_occ_spa*n_vir_spa, n_occ_spa*n_vir_spa)

    H_sing = hbse_sing.reshape(shape)
    H_trip = hbse_trip.reshape(shape)

    return H_sing, H_trip, ext_fock, Hbaresing, Ht2sing, Hbaretrip, Ht2trip


def CC_BSE_spatial(mol,mo,t2_spa,n_occ_spa,n_vir_spa):
    
    eri_ao = mol.intor('int2e') # two e integral
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation

    core_spatialorbs = mo[:, :n_occ_spa]
    vir_spatialorbs = mo[:, n_occ_spa:]

    # Build required two electron integrals
    goovv,_,_,govvo,govov = helper.build_double_ints(core_spatialorbs,vir_spatialorbs,eri_ao)

    # build self energy
    selfener_occ, selfener_vir = get_selfenergy_spatial(t2_spa,goovv) # n_occ x n_occ, n_vir x n_vir

    # build fock matrix
    fock_occ, fock_vir = helper.bccd_fock_mat(mol, mo, n_occ_spa, n_vir_spa,spin=False) # n_occ x n_occ, n_vir x n_vir

    # build gfock
    F_ij = selfener_occ + fock_occ
    F_ab = selfener_vir + fock_vir 

    hbse_sing, hbse_trip,ext_fock,Hbaresing,Ht2sing,Hbaretrip,Ht2trip = build_hbse_singtrip(F_ij, F_ab, t2_spa, govvo, govov, goovv, n_occ_spa, n_vir_spa)

    singE,_,_,_,_ = dgeev(hbse_sing)
    tripE,_,_,_,_ = dgeev(hbse_trip)
        
    return singE, tripE, ext_fock, Hbaresing, Ht2sing, Hbaretrip, Ht2trip, hbse_sing, hbse_trip, fock_occ, fock_vir, selfener_occ, selfener_vir
