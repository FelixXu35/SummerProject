from qutip import *
from qutip.piqs import *

n_tls = int(70)
rho = dicke(n_tls, n_tls/2, 0.8 * n_tls/2)
print(rho)