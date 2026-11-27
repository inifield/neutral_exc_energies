import numpy as np

from pyscf import tdscf
from pyscf.data.nist import HARTREE2EV

def RPA_calc(myhf,NROOTS=10):
    #Constructs RPA object
    td = tdscf.TDHF(myhf)
    td.nstates = NROOTS

    #Does singlet diagonalisation
    td.singlet = True
    td.kernel()
    singlets = np.sort(td.e.real) * HARTREE2EV

    #Does triplet diagonalisation
    td.singlet = False
    td.kernel()
    triplets = np.sort(td.e.real) * HARTREE2EV

    return singlets, triplets