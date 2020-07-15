#from qutip import *
from qutip import *
import numpy as np
from qutip.piqs import *
import scipy as sc

## test
#A = coherent_dm(5, 9)
print(fock_dm(10), coherent_dm(10, 0))
#col = np.linspace(1, 1e15, 1e15)
#data = np.linspace(1, 1e15, 1e15)
#c = sc.sparse.coo_matrix((data, (row, col)), shape=(1e15, 1e15))
#print(c)