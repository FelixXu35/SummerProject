## Main.py
# This file is a part of summer project.
# This file is used to call all other functions and output result.
# There is no input.
# Written by Xiaotian Xu(Felix), 1st July 2020.

## Testing
# 1. all frequency units
# 2. hbar is deleted

## Import packages
import numpy as np
import scipy.constants as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from qutip import *
from qutip.piqs import *

## Defination
n_tls = 1e3 # the number of twp-level particles
n_phot = 3e3 # the number of photons
Sz = 0.8 * n_tls # initial inversion
Kc = 2 * sc.pi * 0.18 # the cavity mode decay rate (MHz)
Ks = 2 * sc.pi * 0.11 # the spin dephasing rate (MHz)
ge = 2 * sc.pi * 1.1 # the ensemble spin-photon coupling strength (MHz)
wc = 1.45 * 1e3 # the cavity frequency (MHz)
ws = 1.45 * 1e3 # the spin traqnsition frequency (MHz)

## Define the collective spin operators
N = n_tls
[jx, jy, jz] = jspin(1) 
jp = jspin(1, "+")
jm = jp.dag()

## TLS parameters
system = Dicke(2)
system.hamiltonian = ws * jz # the Hamiltonian of two-level system (jz incloud a 1/2)
system.dephasing = Ks
D_tls = system.liouvillian() # the liouvillian of spin dephasing

## Interation parameters
a = destroy(n_phot)
h_int = ge * (tensor(jp_tilde, a) + tensor(jm_tilde, a.dag())) # the interaction Hamiltonian

## photon parameters
c_ops_phot = [np.sqrt(Kc) * a]
D_phot = liouvillian(wc * a.dag() * a, c_ops_phot)

## Indentity super-operators
nds = num_dicke_states(n_tls) # the number of dicke states
id_tls = to_super(qeye(nds))
id_phot = to_super(qeye(n_phot))

## Define the total liouvillian
D_int = -1j* spre(h_int) + 1j* spost(h_int)
D_tot = D_int + super_tensor(id_tls, D_phot) + super_tensor(D_tls, id_phot)

## Define operator in the total space
nphot_tot = tensor(qeye(nds), a.dag()*a)

## Time evolution
rho_tls = excited(N)
rho_phot = ket2dm(basis(n_phot, 0))
rho0 = tensor(rho_tls, rho_phot)
t = np.linspace(0, 10, 100)
result = mesolve(D_tot, rho0, t, [], e_ops = [nphot_tot], options = Options(store_states=True, num_cpus=4))
rhot_tot = result.states
nphot_t = result.expect[0]

##
label_size = 20
rho_ss = steadystate(D_tot)
nphot_ss = expect(rho_ss, nphot_tot)

## Visualization
fig3 = plt.figure(3)
plt.plot(t, nphot_t, 'k-', label='Time evolution')
plt.plot(t, t*0 + nphot_ss, 'g--', label = 'Steady-state value')
plt.title(r'Cavity photon population', fontsize = label_size)
plt.xlabel(r'$t$', fontsize = label_size)
plt.ylabel(r'$\langle a^\dagger a\rangle(t)$', fontsize = label_size)

plt.legend(fontsize = label_size)
plt.show()
plt.close()