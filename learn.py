#from qutip import *
from qutip import *
import numpy as np
from qutip.piqs import *
import scipy as sc

## test
#A = coherent_dm(5, 9)
psi_tls = (np.sqrt(0.9)) * basis(2, 0) + (np.sqrt(0.1)) * basis(2, 1)
rho_tls = ket2dm(psi_tls)
print((sigmaz()))
#col = np.linspace(1, 1e15, 1e15)
#data = np.linspace(1, 1e15, 1e15)
#c = sc.sparse.coo_matrix((data, (row, col)), shape=(1e15, 1e15))
#print(c)