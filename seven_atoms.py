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
dim_tls = 2# # the number of twp-level particles
dim_lit = int(2) # the dimension of the light field
prmt = 1e14
num_steps = 1000 # the number of steps will be used in evolution
Kc = 2 * sc.pi * 0.18 # the cavity mode decay rate (MHz)
Ks = 2 * sc.pi * 0.11 # the spin dephasing rate (MHz)
gamma = 2 * sc.pi * 0.011 # the spin-lattice relaxation rate (MHz)
ge = 2 * sc.pi * 1.1 # the single spin-photon coupling strength (MHz)
gs = 2 * sc.pi * 0.042e-6 # the single spin–photon coupling strength (MHz)
wc = 2 * sc.pi * 1.45e3 # the cavity frequency (MHz)
ws = 2 * sc.pi * 1.45e3 # the spin traqnsition frequency (MHz)

## Initialization
step_index = 0
n_phot = [] # the number of pjotons in the light field
spin_phot = []
spin_spin = []
inversion = []

## the initialization quantum state
rho_tls = dicke(dim_tls, dim_tls/2, dim_tls/2) * 0.8
psi_phot = fock(dim_lit)#fock(dim_lit, 1) * 4300 / dim_tls # density matrix of the light field
rho_phot = ket2dm(psi_phot)
rho = tensor(rho_tls, rho_phot)

## Define operator in the total space
nds = num_dicke_states(dim_tls)
a = destroy(dim_lit)
a_tot = tensor(qeye(nds), a)
[jx, jy, jz] = jspin(dim_tls)
jp = jspin(dim_tls, '+')
jm = jspin(dim_tls, '-')
Sz = tensor(jz, qeye(dim_lit))
Sp = tensor(jp, qeye(dim_lit))
Sm = tensor(jm, qeye(dim_lit))

# Identity super-operators
id_tls = to_super(qeye(nds))
id_phot = to_super(qeye(dim_lit))

## Time evolution
t = np.linspace(0, 10, num_steps)
step_length = t[1] - t[0]

## The two-level systems
system = Dicke(dim_tls)
system.hamiltonian = ws * jp * jm
system.dephasing = Ks
#system.collective_emission = gamma
#system.collective_pumping = gamma
D_tls = system.liouvillian()

## The photons
c_ops_phot = [np.sqrt(Kc) * a]
D_phot = liouvillian(wc * a.dag() * a, c_ops_phot)

## The interaction
h_int = ge * tensor(jx, a + a.dag()) # without RWA
D_int = -1j* spre(h_int) + 1j* spost(h_int)
D_tot = D_int + super_tensor(id_tls, D_phot) + super_tensor(D_tls, id_phot)

## Time evolution
result = mesolve(D_tot, rho, t, [], e_ops = [tensor(qeye(nds), a.dag() * a), tensor(jm, a.dag()), 
                                             tensor(jp * jm, qeye(dim_lit)), tensor(jz, qeye(dim_lit))], options = Options(store_states=True, num_cpus=4, nsteps=1e8))
n_phot = result.expect[0] * prmt # the number of pjotons in the light field
spin_phot = result.expect[1] * prmt
spin_spin = result.expect[2]
inversion = result.expect[3]

## Steady states
steady_tls = steadystate(D_tot)
n_phot_ss = np.ones(num_steps) * expect(tensor(qeye(nds), a.dag() * a), steady_tls)
spin_phot_ss = np.ones(num_steps) * expect(tensor(jm, a.dag()), steady_tls)
spin_spin_ss = np.ones(num_steps) * expect(tensor(jp * jm, qeye(dim_lit)), steady_tls)
inversion_ss = np.ones(num_steps) * expect(tensor(jz, qeye(dim_lit)), steady_tls)

#freq_dist = abs(fft(n_phot))

#J = 0.5 * np.ones(num_steps)
#S = dim_tls * inversion + 0.5 * dim_tls * np.ones(num_steps)
#spin_spin = S - S * (S - 1) / dim_tls

## Visualization
plt.figure(1)
plt.plot(t, n_phot)
plt.plot(t, n_phot_ss)
plt.title('the number of photons')
plt.figure(2)
plt.plot(t, spin_phot)
plt.plot(t, spin_phot_ss)
plt.title('spin-photon correlation')
plt.figure(3)
plt.plot(t, spin_spin)
plt.plot(t, spin_spin_ss)
plt.title('spin-spin correlation')
plt.figure(4)
plt.plot(t, inversion)
plt.plot(t, inversion_ss)
plt.title('inversion')
plt.figure(5)
#plt.plot(freq_dist)
#plt.title('fft')
plt.show()

