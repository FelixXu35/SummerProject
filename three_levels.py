
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
num_steps = 1001 # the number of steps will be used in evolution
Kc = 2 * sc.pi * 0.18 # the cavity mode decay rate (MHz)
Ks = 2 * sc.pi * 0.11 # the spin dephasing rate (MHz)
ge = 2 * sc.pi * 1.1 # the single spin-photon coupling strength (MHz)
wc = 2 * sc.pi * 1.45e3 # the cavity frequency (MHz)
ws = 2 * sc.pi * 1.45e3 # the spin traqnsition frequency (MHz)
w12 = 2 * sc.pi * 106.5 # the transition frequency between X and Y
w23 = 2 * sc.pi * 1.344e3 # the transition frequency between Y and Z
wxz = 2 * sc.pi * 0.011 # spin-lattice relaxation rate between X and Z
wyz = 2 * sc.pi * 0.022 # spin-lattice relaxation rate between Y and Z
wxy = 2 * sc.pi * 0.004 # spin-lattice relaxation rate between X and Y

## Initialization
step_index = 0
n_phot = [] # the number of pjotons in the light field
spin_phot = []
spin_spin = []
inversion = []

## Define the operator
Sx13 = Qobj([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
Sz13 = Qobj([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
Sp13 = Qobj([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
Sm13 = Qobj([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
Sx12 = Qobj([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
Sz12 = Qobj([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
Sp12 = Qobj([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
Sm12 = Qobj([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
Sx23 = Qobj([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
Sz23 = Qobj([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
Sp23 = Qobj([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
Sm23 = Qobj([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
a = destroy(dim_lit)
a_tot = tensor(qeye(3), a)

## the initialization quantum state
psi_tls = np.sqrt(0.76) * basis(3, 0) + np.sqrt(0.16) * basis(3, 1) + np.sqrt(0.08) * basis(3, 2)
rho_tls = ket2dm(psi_tls) # density matrix of a two-level system
rho_phot = fock_dm(dim_lit) # density matrix of the light field
rho = tensor(rho_tls, rho_phot)

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
        H = wc * a_tot.dag() * a_tot + 0.5 * ws * tensor(Sz13, qeye(dim_lit)) + ge * (tensor(Sp13, qeye(dim_lit)) * a_tot + tensor(Sm13, qeye(dim_lit)) * a_tot.dag()) # 0.5 is normalization factor
        rho_der -= 1j * commutator(H, rho)

        ## Cavity decay
        rho_der += Kc * (a_tot * rho * a_tot.dag() - 0.5 * a_tot.dag() * a_tot * rho - 0.5 * rho * a_tot.dag() * a_tot)

        ## Spin dephasing
        rho_der += Ks * (tensor(Sz13, qeye(dim_lit)) * rho * tensor(Sz13, qeye(dim_lit)).dag() - \
            0.5 * tensor(Sz13, qeye(dim_lit)).dag() * tensor(Sz13, qeye(dim_lit)) * rho - 0.5 * rho * tensor(Sz13, qeye(dim_lit)).dag() * tensor(Sz13, qeye(dim_lit)))

        ## Relaxation
        rho_der += wxz * (tensor(Sm13, qeye(dim_lit)) * rho * tensor(Sm13, qeye(dim_lit)).dag() - \
            0.5 * tensor(Sm13, qeye(dim_lit)).dag() * tensor(Sm13, qeye(dim_lit)) * rho - 0.5 * rho * tensor(Sm13, qeye(dim_lit)).dag() * tensor(Sm13, qeye(dim_lit)))
        rho_der += wxy * (tensor(Sm12, qeye(dim_lit)) * rho * tensor(Sm12, qeye(dim_lit)).dag() - \
            0.5 * tensor(Sm12, qeye(dim_lit)).dag() * tensor(Sm12, qeye(dim_lit)) * rho - 0.5 * rho * tensor(Sm12, qeye(dim_lit)).dag() * tensor(Sm12, qeye(dim_lit)))
        rho_der += wyz * (tensor(Sm23, qeye(dim_lit)) * rho * tensor(Sm23, qeye(dim_lit)).dag() - \
            0.5 * tensor(Sm23, qeye(dim_lit)).dag() * tensor(Sm23, qeye(dim_lit)) * rho - 0.5 * rho * tensor(Sm23, qeye(dim_lit)).dag() * tensor(Sm23, qeye(dim_lit)))
        rho_der += wxz * (tensor(Sp13, qeye(dim_lit)) * rho * tensor(Sp13, qeye(dim_lit)).dag() - \
            0.5 * tensor(Sp13, qeye(dim_lit)).dag() * tensor(Sp13, qeye(dim_lit)) * rho - 0.5 * rho * tensor(Sp13, qeye(dim_lit)).dag() * tensor(Sp13, qeye(dim_lit)))
        rho_der += wxy * (tensor(Sp12, qeye(dim_lit)) * rho * tensor(Sp12, qeye(dim_lit)).dag() - \
            0.5 * tensor(Sp12, qeye(dim_lit)).dag() * tensor(Sp12, qeye(dim_lit)) * rho - 0.5 * rho * tensor(Sp12, qeye(dim_lit)).dag() * tensor(Sp12, qeye(dim_lit)))
        rho_der += wyz * (tensor(Sp23, qeye(dim_lit)) * rho * tensor(Sp23, qeye(dim_lit)).dag() - \
            0.5 * tensor(Sp23, qeye(dim_lit)).dag() * tensor(Sp23, qeye(dim_lit)) * rho - 0.5 * rho * tensor(Sp23, qeye(dim_lit)).dag() * tensor(Sp23, qeye(dim_lit)))

        if step_index2 == 0:
            rho += rho_der * 0.75 *step_length # Predictor
            rho_dash = rho - 5/12 * rho_der * step_length

        if step_index2 == 1:
            rho = rho_dash + 2/3 * rho_der * step_length # Corrector

            ## Calculation
            n_phot.append(((rho * tensor(qeye(3), a.dag() * a)).tr() * dim_tls).real)
            spin_phot.append(((rho * tensor(Sz13, a.dag())).tr() * dim_tls).real)
            spin_spin.append(((rho * tensor(Sp13 * Sm13, qeye(dim_lit))).tr()).real)
            inversion.append(((rho * tensor(Sz13, qeye(dim_lit))).tr()).real)

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

