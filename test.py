from qutip import *
from qutip.piqs import *
import numpy as np

psi_phot = fock(2, 0) * np.sqrt((7e14 - 4300) / 7e14) + fock(2, 1) * np.sqrt(4300 / 7e14) # density matrix of the light field
rho_phot = ket2dm(psi_phot)
a = destroy(2)
n = ((rho_phot * a.dag() * a).tr()).real * 7e14
print(np.linspace(1, 10, 1000))