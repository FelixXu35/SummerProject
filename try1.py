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
from urop_week3_differentialeqns import maser_coupledeqns
plt.figure()
list_ind = []
max1 = []

for ind in range(10):

    ## Defination
    dim_tls = 7e14# # the number of twp-level particles
    dim_lit = int(2) # the dimension of the light field
    num_steps = 1000#32768 # the number of steps will be used in evolution
    Kc = 2 * sc.pi * 0.18 # the cavity mode decay rate (0.18 MHz)
    Ks = 2 * sc.pi * 0.11 # the spin dephasing rate (0.11 MHz)
    gamma = 2 * sc.pi * 0.011 # the spin-lattice relaxation rate (0.011 MHz)
    ge = 2 * sc.pi * 1.1 # the single spin-photon coupling strength (1.1 MHz)
    gs = 2 * sc.pi * 0.084e-6 # the single spin–photon coupling strength (0.042e-6 MHz)
    wc = 2 * sc.pi * 1.45e3 # the cavity frequency (1.45e3 MHz)
    ws = 2 * sc.pi * 1.45e3 # the spin traqnsition frequency (1.45e3 MHz)
    thermal_phot = (ind+1) * 1000 # the number of thermal photon in the initial state

    ## Initialization
    step_index = 0
    n_phot = [] # the number of pjotons in the light field
    spin_phot = []
    spin_spin = []
    inversion = []

    ## the initialization quantum state
    psi_tls = np.sqrt(0.9) * basis(2, 0) + np.sqrt(0.1) * basis(2, 1)
    rho_tls = ket2dm(psi_tls) # density matrix of a two-level system
    psi_phot = fock(2, 0) * np.sqrt((dim_tls - thermal_phot)) + fock(2, 1) * np.sqrt(thermal_phot) # density matrix of the light field
    rho_phot = ket2dm(psi_phot)
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
    t = [0, 10/(num_steps-1)]

    ## The two-level systems
    system = Dicke(1)
    system.hamiltonian = ws * sigmap()*sigmam()
    system.dephasing = Ks
    #system.collective_emission = gamma 
    #system.collective_pumping = gamma 
    D_tls = system.liouvillian()

    ## The photons
    c_ops_phot = [np.sqrt(Kc) * a]
    D_phot = liouvillian(wc * a.dag() * a, c_ops_phot)

    ## Time evolution
    for index in range(num_steps):
        ## The interaction
        n = expect(tensor(qeye(nds), a.dag() * a), rho)
        #print(n)
        h_int = gs * np.sqrt(n) * tensor(sigmax(), a + a.dag()) # without RWA
        D_int = -1j* spre(h_int) + 1j* spost(h_int)
        D_tot = D_int + super_tensor(id_tls, D_phot) + super_tensor(D_tls, id_phot)
        print(ind, index, '/', num_steps)
        result = mesolve(D_tot, rho, t, [], e_ops = [tensor(qeye(nds), a.dag() * a), tensor(sigmam(), a.dag()), 
                                                     tensor(sigmap() * sigmam(), qeye(dim_lit)), tensor(sigmaz(), qeye(dim_lit))], options = Options(store_states=True, num_cpus=4, nsteps=1e8))
        rho = result.states[1]
        n_phot.append(result.expect[0][0]) # the number of pjotons in the light field
        spin_phot.append(-2 * result.expect[1][0].imag / dim_tls)
        #spin_spin = result.expect[2]
        inversion.append(result.expect[3][0] / dim_tls)

    ## Steady states
    #steady_tls = steadystate(D_tot)
    #n_phot_ss = np.ones(num_steps) * expect(tensor(qeye(nds), a.dag() * a), steady_tls)
    #spin_phot_ss = np.ones(num_steps) * expect(tensor(sigmam(), a.dag()), steady_tls)
    #spin_spin_ss = np.ones(num_steps) * expect(tensor(sigmap() * sigmam(), qeye(dim_lit)), steady_tls)
    #inversion_ss = np.ones(num_steps) * expect(tensor(sigmaz(), qeye(dim_lit)), steady_tls)

    J = 0.5 * np.ones(num_steps)
    inversion = np.array(inversion)
    spin_spin = (J + inversion/2) * (J - inversion/2 + 1/dim_tls)
    t = np.linspace(1, 10, num_steps)

    ## comparation
    gs=0.042*(2.*np.pi)
    kc=0.18*(10**6)*(2.*np.pi)
    ks=0.11*(10**6)*(2.*np.pi)
    N=7.*(10.**14.)
    gamma=0
    time_span=[0,10*10**(-6)]
    num=num_steps
    delta=0

    y = maser_coupledeqns(gs,kc,ks,N,gamma,time_span,num,delta)

    m = max(n_phot)
    position = n_phot.index(m)
    
    max1.append(m)
    list_ind.append(position)
    print(list_ind)

t=np.linspace(1, 10, 10)
plt.figure(1)
plt.plot(t, max1)
plt.title('the maximum values')
plt.xlabel('the number of thermal photon / 1000')
plt.ylabel('the number of photon')
for a, b in zip(t, max1):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=20)
plt.figure(2)
plt.plot(t, list_ind)
plt.title('when the maximum values show up')
plt.xlabel('the number of thermal photon / 1000')
plt.ylabel('the time that the max value show up / 10ns')
for a, b in zip(t, list_ind):
    plt.text(a, b, b, ha='center', va='bottom', fontsize=20)
plt.show()
