## Main.py
# This file is a part of summer project.
# This file is used to call all other functions and output result.
# There is no input.
# Written by Xiaotian Xu(Felix), 1st July 2020.

## Import packages
import numpy as np
import scipy.constants as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from qutip import *
from qutip.piqs import *

## Defination
dim_tls = 7e14# # the number of twp-level particles
dim_lit = int(10) # the dimension of the light field
num_steps = 1000 # the number of steps will be used in evolution
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

## Time evolution
t = np.linspace(0, 10, num_steps)
step_length = t[1] - t[0]

## The RK2 method, which is realized with a for loop
for step_index in range(num_steps):
    for step_index2 in range(2):

        ## Procedure
        print(step_index, '/', num_steps)

        ## Initialization
        rho_der = 0

        ## The Hamiltonian
        H = wc * a_tot.dag() * a_tot + 0.5 * ws * Sz + ge * (Sp * a_tot + Sm * a_tot.dag()) # 0.5 is normalization factor
        rho_der -= 1j * commutator(H, rho)

        ## Cavity decay
        rho_der += Kc * (a_tot * rho * a_tot.dag() - 0.5 * a_tot.dag() * a_tot * rho - 0.5 * rho * a_tot.dag() * a_tot)

        ## Spin dephasing
        rho_der += Ks * (Sz * rho * Sz.dag() - 0.5 * Sz.dag() * Sz * rho - 0.5 * rho * Sz.dag() * Sz)
        
        if step_index2 == 0:
            rho += rho_der * step_length # Predictor
            rho_dash = rho - 0.5 * rho_der * step_length

        if step_index2 == 1:
            rho = rho_dash + 0.5 * rho_der * step_length # Corrector

            ## Calculation
            rho0 = rho.ptrace(0)
            rho1 = rho.ptrace(1)
            n_phot.append(((rho1 * a.dag() * a).tr() * dim_tls).real)
            spin_phot.append(((rho * tensor(sigmam(), a.dag())).tr() * dim_tls).real)
            spin_spin.append(((rho0 * sigmap() * sigmam()).tr()).real)
            inversion.append(((rho0 * sigmaz()).tr()).real)

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
plt.show()