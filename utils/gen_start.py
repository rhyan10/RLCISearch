import ase.io
import numpy as np    

def gen_start(db, random_numbers):
    mols = []
    for num in random_numbers:
        mol = db[num]
        mols.append(mol)    
    return mols
