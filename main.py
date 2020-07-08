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
dim_tls = 100 # the number of twp-level particles
dim_lit = int(300) # the dimension of the light field
num_steps = 50 # the number of steps will be used in evolution
Kc = 2 * sc.pi * 0.18 # the cavity mode decay rate (MHz)
Ks = 2 * sc.pi * 0.11 # the spin dephasing rate (MHz)
gs = 2 * sc.pi * 4.2e-8 # the single spin-photon coupling strength (MHz)
wc = 1.45 * 1e3 # the cavity frequency (MHz)
ws = 1.45 * 1e3 # the spin traqnsition frequency (MHz)

## Initialization
step_index = 0
n_exc = [] # the number of atoms on the excited state
n_phot = [] # the number of pjotons in the light field

## Define the collective spin operators
[jx, jy, jz] = jspin(1) 
jp = jspin(1, "+")
jm = jp.dag()
a = destroy(dim_lit)

## the initialization quantum state
rho_tls = dicke(1, 0.5, 0.5)
rho_phot = ket2dm(basis(dim_lit, 0))
rho = tensor(rho_tls, rho_phot)

## Define operator in the total space
nphot_tot = tensor(qeye(2), a.dag() * a) # number operator
a_tot = tensor(qeye(2), a)
jz_tot = tensor(jz, qeye(dim_lit))
jp_tot = tensor(jp, qeye(dim_lit))
jm_tot = tensor(jm, qeye(dim_lit))

## Time evolution
t = np.linspace(0, 10, num_steps)
step_length = t[1] - t[0]

## The RK2 method, which is realized with a for loop
for step_index in range(num_steps):
    for step_index2 in range(2):
        ## Initialization
        rho_der = 0

        ## The number of excited atoms
        N = (np.trace(rho * jz_tot) + 0.5) * dim_tls 
        print(N)
        N = N.real

        ## The Hamiltonian
        H = wc * a_tot.dag() * a_tot + ws * jz_tot + np.sqrt(N) * gs * (jp_tot * a_tot + jm_tot * a_tot.dag())
        rho_der -= 1j * commutator(H, rho)

        ## Cavity decay
        rho_der += Kc * (a_tot * rho * a_tot.dag() - 0.5 * a_tot.dag() * a_tot * rho - 0.5 * rho * a_tot.dag() * a_tot)

        ## Spin dephasing
        rho_der += Ks * (jz_tot * rho * jz_tot.dag() - 0.5 * jz_tot.dag() * jz_tot * rho - 0.5 * rho * jz_tot.dag() * jz_tot)
        
        if step_index2 == 0:
            rho += rho_der * step_length # Predictor
            rho_dash = rho - 0.5 * rho_der * step_length

        if step_index2 == 1:
            rho = rho_dash + 0.5 * rho_der * step_length # Corrector
            P = np.trace(rho * nphot_tot)
            P = P.real
            n_phot.append(P)
            n_exc.append(N)

## Visualization
fig = plt.figure(3)
plt.plot(t, n_exc)
plt.plot(t, n_phot)
plt.show()