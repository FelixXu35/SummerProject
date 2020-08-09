## This module is used to simulate breeze's paper.
# This module uses the stochastic Liouville equation to simulate the couple between a maser's material with the cavity modes.
# This module uses the 4th-order Runge-Kutta method to solve the master equation.
# This module is written by Xiaotian Xu(Felix), 8th August, 2020.

## Import packages
import numpy as np
import matplotlib.pyplot as plt
from qutip import *
from qutip.piqs import *

## Defination
dim_tls = 7e14# # the total number of two-level particles
num_steps = 1000 # the number of steps will be used in evolution
Kc = 2 * np.pi * 0.18 # the cavity mode decay rate (0.18 MHz)
Ks = 2 * np.pi * 0.11 # the spin dephasing rate (MHz)
gamma = 2 * np.pi * 0.011 # the spin-lattice relaxation rate (MHz)
ge = 2 * np.pi * 1.1 # the single spin-photon coupling strength (MHz)
gs = 2 * np.pi * 0.042e-6 # the single spin–photon coupling strength (MHz)
wc = 2 * np.pi * 1.45e3 # the cavity frequency (MHz)
ws = 2 * np.pi * 1.45e3 # the spin traqnsition frequency (MHz)

## Initialization
step_index = 0
RK_index = 0
n_phot = [] # the number of photons in the light field <a.dag()*a>
spin_phot = [] # the spin-photon correlation <a.dag()*S->
spin_spin = [] # the spin-spin correlation <S+S->
inversion = [] # the inversion <S_z>

## the initialization quantum state
psi_tls = np.sqrt(0.9) * basis(2, 0) + np.sqrt(0.1) * basis(2, 1)
rho_tls = ket2dm(psi_tls) # density matrix of a two-level system
psi_phot = fock(2, 0) * np.sqrt((dim_tls - 4300) / dim_tls) + fock(2, 1) * np.sqrt(4300 / dim_tls) # the number of thermal photon is 4300
rho_phot = ket2dm(psi_phot) # density matrix of the light field
rho = tensor(rho_tls, rho_phot)

## Define operator in the total space
a = destroy(2)
a_tot = tensor(qeye(2), a)
Sz = tensor(sigmaz(), qeye(2))
Sp = tensor(sigmap(), qeye(2))
Sm = tensor(sigmam(), qeye(2))

## Time evoluation
t = np.linspace(0, 10, num_steps)
step_length = t[1] - t[0]

## The RK4 method
for step_index in range(num_steps):
    for RK_index in range(4):

        ## Procedure
        print(step_index, '/', num_steps)

        ## The number of photon in the cavity
        rho1 = rho.ptrace(1)
        n = ((rho1 * a.dag() * a).tr()).real * 7e14

        ## The Hamiltonian
        H = wc * a_tot.dag() * a_tot + 0.5 * ws * Sz + ge * (Sp * a_tot + Sm * a_tot.dag()) # 0.5 is normalization factor
        rho_der = -1j * commutator(H, rho)

        ## The Liouvillians
        rho_der += Kc * (a_tot * rho * a_tot.dag() - 0.5 * a_tot.dag() * a_tot * rho - 0.5 * rho * a_tot.dag() * a_tot)
        rho_der += Ks * (Sz * rho * Sz.dag() - 0.5 * Sz.dag() * Sz * rho - 0.5 * rho * Sz.dag() * Sz)
        #rho_der += gamma * (Sm * rho * Sp - 0.5 * Sp * Sm * rho - 0.5 * rho * Sp * Sm)
        #rho_der += gamma * (Sp * rho * Sm - 0.5 * Sm * Sp * rho - 0.5 * rho * Sm * Sp)

        if RK_index == 0:
            k1 = rho_der * step_length
            rho_store = rho
            rho = rho_store + 0.5 * k1

        if RK_index == 1:
            k2 = rho_der * step_length
            rho = rho_store + 0.5 * k2

        if RK_index == 2:
            k3 = rho_der * step_length
            rho = rho_store + k3

        if RK_index == 3:
            k4 = rho_der + step_index
            rho = rho_store + 1/6 * (k1 + 2*k2 + 2*k3 + k4)

            ## the expectation value
            rho0 = rho.ptrace(0)
            rho1 = rho.ptrace(1)
            n_phot.append(((rho1 * a.dag() * a).tr()).real)
            spin_phot.append(((rho * tensor(sigmam(), a.dag())).tr()).imag)
            spin_spin.append(((rho0 * sigmap() * sigmam()).tr()).real)
            inversion.append(((rho0 * sigmaz()).tr()).real)

J = 0.5 * np.ones(num_steps)
inversion = np.array(inversion)
spin_spin = (J + inversion/2) * (J - inversion/2 + 1/dim_tls)

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
plt.subplot(4, 1, 1)
plt.plot(t, n_phot, 'k-')
plt.ylabel('<${a^\dag}a$>')
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