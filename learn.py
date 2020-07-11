#from qutip import *
from qutip import *
import numpy as np
from qutip.piqs import *
import scipy as sc

## test
#A = coherent_dm(5, 9)
rho = ket2dm(basis(2, 1))
print(rho)
sm = destroy(2)
print((sm.dag() * sm * rho).tr())
#col = np.linspace(1, 1e15, 1e15)
#data = np.linspace(1, 1e15, 1e15)
#c = sc.sparse.coo_matrix((data, (row, col)), shape=(1e15, 1e15))
#print(c)