
## Main.py
# This file is a part of summer project.
# This file is used to test function "mesolve".
# There is no input.
# Written by Xiaotian Xu(Felix), 13th July 2020.

## Import packages
import numpy as np
import scipy.constants as sc
import scipy.sparse as sp
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from qutip import *
from qutip.piqs import *

## Defination
dim_tls = 7e14# # the number of twp-level particles
dim_lit = int(10) # the dimension of the light field
num_steps = 3000 # the number of steps will be used in evolution
Kc = 2 * sc.pi * 0.18 # the cavity mode decay rate (MHz)
Ks = 2 * sc.pi * 0.11 # the spin dephasing rate (MHz)
ge = 2 * sc.pi * 1.1 # the single spin-photon coupling strength (MHz)
wc = 2 * sc.pi * 1.45e3 # the cavity frequency (MHz)
ws = 2 * sc.pi * 1.45e3 # the spin traqnsition frequency (MHz)

## Initialization
step_index = 0
n_phot = [] # the number of pjotons in the light field
spin_phot = []
spin_spin = []
inversion = []

## the initialization quantum state
psi_tls = np.sqrt(0.9) * basis(2, 0) + np.sqrt(0.1) * basis(2, 1)
rho_tls = ket2dm(psi_tls) # density matrix of a two-level system
rho_phot = fock_dm(dim_lit) # density matrix of the light field
rho = tensor(rho_tls, rho_phot)

## Define operator in the total space
a = destroy(dim_lit)
a_tot = tensor(qeye(2), a)
Sz = tensor(sigmaz(), qeye(dim_lit))
Sp = tensor(sigmap(), qeye(dim_lit))
Sm = tensor(sigmam(), qeye(dim_lit))

# Identity super-operators
nds = num_dicke_states(1)
id_tls = to_super(qeye(nds))
id_phot = to_super(qeye(dim_lit))

## Time evolution
t = np.linspace(0, 10, num_steps)
step_length = t[1] - t[0]

## The two-level systems
system = Dicke(1)
system.hamiltonian = 0.5 * ws * sigmaz()
system.dephasing = Ks
D_tls = system.liouvillian()

## The photons
c_ops_phot = [np.sqrt(Kc) * a]
D_phot = liouvillian(wc * a.dag() * a , c_ops_phot)

## The interaction
h_int = ge * tensor(sigmax(), a + a.dag()) # without RWA
D_int = -1j* spre(h_int) + 1j* spost(h_int)
D_tot = D_int + super_tensor(id_tls, D_phot) + super_tensor(D_tls, id_phot)

## Time evolution
result = mesolve(D_tot, rho, t, [], e_ops = [tensor(qeye(nds), a.dag() * a), tensor(sigmam(), a.dag()), 
                                             tensor(sigmap() * sigmam(), qeye(dim_lit)), tensor(sigmaz(), qeye(dim_lit))], options = Options(store_states=True, num_cpus=4))
n_phot = result.expect[0] * dim_tls # the number of pjotons in the light field
spin_phot = result.expect[1] * dim_tls
spin_spin = result.expect[2]
inversion = result.expect[3]

freq_dist = abs(fft(n_phot))

J = 0.5 * np.ones(num_steps)
spin_spin = (J + inversion/2) * (J - inversion/2 + 1 / dim_tls)

## Visualization
plt.figure(1)
plt.plot(t, n_phot)
plt.title('the number of photons')
plt.figure(2)
plt.plot(t, spin_phot)
plt.title('spin-photon correlation')
plt.figure(3)
plt.plot(t, spin_spin)
plt.title('spin-spin correlation')
plt.figure(4)
plt.plot(t, inversion)
plt.title('inversion')
plt.figure(5)
plt.plot(freq_dist)
plt.title('fft')
plt.show()
