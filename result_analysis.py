import numpy as np
from pathlib import Path
import pandas as pd

molecules = sorted((file.stem for file in Path("results").glob("*.txt")),key=str.casefold)
singlet_energies = {}
triplet_energies = {}
for label in molecules:
    with open(f"results/{label}.txt","r",) as file:
        text = file.read()
        for line in text.splitlines():
            if line.startswith("Singlet excitation energies (eV):"):
                singlets = np.fromstring(line.split("[")[1].split("]")[0],sep=" ")
                singlet_energies[label]=singlets

            if line.startswith("Triplet excitation energies (eV):"):
                triplets = np.fromstring(line.split("[")[1].split("]")[0],sep=" ")
                triplet_energies[label]=triplets    
print(singlet_energies)
print(triplet_energies)

sing_df = pd.DataFrame.from_dict(singlet_energies,orient="index")
sing_df.index.name = "Molecules"
sing_df.columns = [f"Excitation {i+1} (eV)" for i in range(sing_df.shape[1])]
print(sing_df.head())

trip_df = pd.DataFrame.from_dict(triplet_energies,orient="index")
trip_df.index.name = "Molecules"
trip_df.columns = [f"Excitation {i+1} (eV)" for i in range(trip_df.shape[1])]
print(trip_df.head())