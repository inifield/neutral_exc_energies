# Excitation energies from Coupled Cluster Greens Function Theory

A number of different levels of theory have been implemented to calculate excitation energies for closed shell systems. Following [Coveney and Tew's paper](https://arxiv.org/abs/2309.10451):
- `cc_BSE_spinorb.py`: CC-BSE in Spin-orbital form 
- `cc_BSE_spatialorb.py`: CC-BSE in spin-adapted form
- `RPAtest.py`: Performs RPA independently and compares the downfolded Hamiltonian to that constructed from plugging the relevant RPA quantities into the CC-BSE Hamiltonian in the full spin-orbital space to test the implementation of the CC-BSE

Use the `cc_controller.py` script to select your molecule (from `molecules.py` dictionary), define your basis set and control which of the calculations desired to run.


