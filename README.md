# Excitation energies from Coupled Cluster Greens Function Theory

A number of different levels of theory have been implemented to calculate excitation energies for closed shell systems. Following [Coveney and Tew's paper](https://arxiv.org/abs/2309.10451):
- `cc_BSE_spinorb.py`: CC-BSE in Spin-orbital form 
- `cc_BSE_spatialorb.py`: CC-BSE in spin-adapted form
- `RPAtest.py`: Performs RPA independently and compares the downfolded Hamiltonian to that constructed from plugging the relevant RPA quantities into the CC-BSE Hamiltonian in the full spin-orbital space to test the implementation of the CC-BSE

Use the `cc_controller.py` to control the parameters for the calculations, i.e. basis set, BCCD tolerance, BCCD max cycle and whether full spin orbital calculation is performed. Define molecule geometries in Angstrom in /geometries with a .xyz file. Run calculations from terminal by running 'python -m controller' followed by list of names of molecules (case insensitive)
e.g. 'python -m controller beryllium water ammonia'
Calling no molecules defaults to beryllium. Calling 'python -m controller all' runs all the geometries defined. The --save argument appends the results to the .txt files in /results.
e.g. 'python -m controller h2 --save' adds the latest results to results/H2.txt



