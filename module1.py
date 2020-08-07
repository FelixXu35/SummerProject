## Packages
import numpy as np
import scipy.constants as sc
import scipy.sparse as sp
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from qutip import *
from qutip.piqs import *
import time

time_start = time.time()

## Definition
Kc = 2 * sc.pi * 0.18 # the cavity mode decay rate (MHz)
Ks = 2 * sc.pi * 0.11 # the spin dephasing rate (MHz)
gamma = 2 * sc.pi * 0.011 # the spin-lattice relaxation rate (MHz)
ge = 2 * sc.pi * 1.1 # the single spin-photon coupling strength (MHz)
gs = 2 * sc.pi * 0.042e-6 # the single spin–photon coupling strength (MHz)
wc = 2 * sc.pi * 1.45e3 # the cavity frequency (MHz)
ws = 2 * sc.pi * 1.45e3 # the spin traqnsition frequency (MHz)

# TLS parameters
n_tls = 1
N = n_tls
system = Dicke(N = n_tls)
[jx, jy, jz] = jspin(N)
jp = jspin(N,"+") / np.sqrt(n_tls)
jm = jp.dag()
system.hamiltonian = ws * jz
system.dephasing = Ks
D_tls = system.liouvillian() 

# Light-matter coupling parameters
nphot = 15
a = destroy(nphot)
h_int = ge * (tensor(a, jp) + tensor(a.dag(), jm))

# Photonic Liouvillian
c_ops_phot = [np.sqrt(Kc) * a]
D_phot = liouvillian(wc * a.dag()*a , c_ops_phot)

# Identity super-operators
nds = num_dicke_states(n_tls)
id_tls = to_super(qeye(nds))
id_phot = to_super(qeye(nphot))

# Define the total Liouvillian
D_int = -1j* spre(h_int) + 1j* spost(h_int)
D_tot = super_tensor(D_phot, id_tls) + super_tensor(id_phot, D_tls) + D_int

# Define operator in the total space
nphot_tot = tensor(a.dag()*a, qeye(nds))

excited_state = excited(N)
ground_phot = ket2dm(basis(nphot,0))
rho0 = tensor(ground_phot, excited_state)
t = np.linspace(0, 10, 1000)
result2 = mesolve(D_tot, rho0, t, [], e_ops = [nphot_tot], 
                  options = Options(store_states=True, num_cpus=4, nsteps=1e5))
rhot_tot = result2.states
nphot_t = result2.expect[0]
spin_phot = -2 * expect(tensor(a.dag(), jm), rhot_tot).imag
spin_spin = expect(tensor(qeye(nphot), jp*jm), rhot_tot)
inversion = 2 * expect(tensor(qeye(nphot), jz), rhot_tot)

J = 0.5 * np.ones(1000)
spin_spin = (J + inversion/2) * (J - inversion/2 + 1/7e14)


fig3 = plt.figure(1)
plt.subplot(4, 1, 1)
plt.plot(t, nphot_t, 'k-')
plt.ylabel('<${a^\dag}a$>')
plt.title('num of tls: %s' %(n_tls))
plt.subplot(4, 1, 2)
plt.plot(t, spin_phot, 'k-')
plt.ylabel('<${a^\dag}{S^-}$>')
plt.subplot(4, 1, 3)
plt.plot(t, spin_spin, 'k-')
plt.ylabel ('<${S^+}{S^-}$>/N')
plt.subplot(4, 1, 4)
plt.plot(t, inversion, 'k-')
plt.ylabel('<${S^z}$>')



plt.show()
plt.close()

time_end = time.time()

print(time_end - time_start)