import numpy as np
from pyscf import gto, scf, cc
import src.BSE_Helper as helper

def get_self_energy(t2_spin:np.ndarray, oovv:np.ndarray) -> tuple[np.ndarray,np.ndarray]:

    occ_selfeng = 0.5 * np.einsum("ikab,jkab -> ij", oovv,t2_spin,optimize="optimal")
    vir_selfeng = -0.5 * np.einsum("ijbc,ijac -> ab", oovv,t2_spin,optimize="optimal")

    return occ_selfeng, vir_selfeng


def build_gfock(mol,mo,t2_spin, oovv, n_occ_spa,n_vir_spa):
    occ_selfener, vir_selfener = get_self_energy(t2_spin,oovv) # n_occ x n_occ, n_vir x n_vir

    fock_occ, fock_vir = helper.bccd_fock_mat(mol,mo,n_occ_spa,n_vir_spa,spin=True) 

    F_ij = occ_selfener + fock_occ    #(n_occ x n_occ)
    F_ab = vir_selfener + fock_vir    #(n_vir x n_vir)

    return F_ij,F_ab, fock_occ, fock_vir, occ_selfener, vir_selfener

def build_hbse_spin(F_ij:np.ndarray,F_ab:np.ndarray,ovvo,oovv,t2_spin, n_occ_spin, n_vir_spin):
    F_baij = np.einsum('ba,ij->iajb', F_ab, np.identity(n_occ_spin),optimize='optimal')
    F_ijab = np.einsum('ij,ab->iajb', F_ij, np.identity(n_vir_spin),optimize='optimal')

    ext_fock = F_baij - F_ijab

    Hbare =  np.einsum('ibaj->iajb', ovvo, optimize='optimal')

    Ht2 = np.einsum("ikac,kjcb->iajb",oovv, t2_spin, optimize='optimal')

    H_bse = ext_fock + Hbare + Ht2
    
    return ext_fock, Hbare, Ht2, H_bse



def CC_BSE_spin(mol,mo,t2_spa,n_occ_spa,n_vir_spa):
    n_occ_spin=2*n_occ_spa
    n_vir_spin = 2*n_vir_spa
    # get molecular orbitals and t2 amplitudes
    t2_spin = helper.build_t2_spin(t2_spa,n_occ_spa,n_vir_spa)
 
    core_spinorbs, vir_spinorbs = helper.get_spinorbs(mo,n_occ_spa)

    # Build electron repulsion integrals
    _, eri_ao = helper.spinor_one_and_two_e_int(mol)                # Find eri in spinor form
    eri_ao = np.einsum("pqrs->prqs", eri_ao, optimize="optimal")    # Convert to Physicist's notation
    anti_eri_ao = eri_ao - np.einsum("prqs->prsq", eri_ao, optimize='optimal')

    # Build the required anti-symmetrised orbitals
    oovv,_,_,ovvo,_ = helper.build_double_ints(core_spinorbs,vir_spinorbs,anti_eri_ao)

    # Extended fock operator (fock + self energy)
    F_ij, F_ba, fock_occ, fock_vir, occ_selfener, vir_selfener = build_gfock(mol,mo,t2_spin,oovv,n_occ_spa,n_vir_spa)

    # (n_occ,n_vir,n_occ,n_vir,nspincase)
    ext_fock, Hbare, Ht2, hbse = build_hbse_spin(F_ij,F_ba,ovvo,oovv,t2_spin,n_occ_spin,n_vir_spin)

    hbse_sing, hbse_trip = helper.sing_trip_excitations(hbse, n_occ_spa, n_vir_spa)

    return hbse, hbse_sing, hbse_trip, Hbare, Ht2, ext_fock, fock_occ, fock_vir, occ_selfener, vir_selfener