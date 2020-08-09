
"""
This file uses 'mesovle' to solve the master eqn stated in the Breeze paper, 
the aim is to reproduce the graphs in that paper.
"""
import numpy as np
from qutip import *
from qutip.piqs import *
import matplotlib.pyplot as plt

## constants listed in the paper

dim_tls = 5
#7e14# # the number of two-level particles

dim_lit = int(15) # the dimension of the light field

Kc = 2 * np.pi * 0.18 # the cavity mode decay rate (MHz)

Ks = 2 * np.pi * 0.11 # the spin dephasing rate (MHz)

ge = 2 * np.pi * 1.1 # the collective spin-photon coupling strength (MHz)

gs = ge/np.sqrt(dim_tls) # the single spin-photon coupling strength (MHZ)

wc = 2 * np.pi * 1.45e3 # the cavity frequency (MHz)

ws = 2 * np.pi * 1.45e3 # the spin transition frequency (MHz)

# gamma = 2 * np.pi * 0.0177 # the spin-lattice relaxation rate
gamma=0

## time 
time=np.linspace(0, 10, 1000)

time_num=time.shape[0] # the number of time moments  evaluation 


## the inital state of the system
#psi_tls = np.sqrt(0.9) * basis(2, 0) + np.sqrt(0.1) * basis(2, 1) # the initial atom distribution 

rho_tls = np.sqrt(0.9)*excited(dim_tls)+np.sqrt(0.1)*ground(dim_tls) # density matrix of the two-level system

rho_phot = ket2dm(basis(dim_lit,0)) # density matrix of the light field

rho_init = tensor(rho_phot,rho_tls)

"""
Dynamics of the system, by defining the liouvilian 
"""

## TLS 
N = dim_tls
system = Dicke(N)
[jx, jy, jz] = jspin(N)

jp = jspin(N,"+")
Sp=(np.sqrt(N)**(-1))*jp

jm = jp.dag()
Sm=(np.sqrt(N)**(-1))*jm

system.hamiltonian = ws * jz 
system.dephasing = Ks
system.pumping=gamma
system.emission=gamma

D_tls = system.liouvillian()

## Light-matter coupling 
a = destroy(dim_lit)

h_int = ge * (tensor(a, Sp)+tensor(a.dag(),Sm))
#h_int = ge * tensor(a + a.dag(), jx)

# Photonic Liouvillian
c_ops_phot = [np.sqrt(Kc)*a]
D_phot = liouvillian(wc * a.dag()*a , c_ops_phot)

# Identity super-operators
nds = num_dicke_states(N)
id_tls = to_super(qeye(nds))
id_phot = to_super(qeye(dim_lit))

# Define the total Liouvillian
D_int = -1j* spre(h_int) + 1j* spost(h_int)

D_tot = D_int + super_tensor(D_phot, id_tls) + super_tensor(id_phot, D_tls)

# Define operator in the total space
nphot_op = tensor(a.dag()*a, qeye(nds))
spin_phot_op= tensor(a.dag(),Sm)
spin_spin_op=tensor(qeye(dim_lit),Sp*Sm)
inversion_op=tensor(qeye(dim_lit),jz)


result = mesolve(D_tot, rho_init, time, [], e_ops = [nphot_op,spin_phot_op,spin_spin_op,inversion_op],options = Options(store_states=True, nsteps=4500, num_cpus=4))

n_phot = result.expect[0].real  # the number of pjotons in the light field

spin_phot = result.expect[1].imag 

spin_spin = result.expect[2].real / N

inversion = result.expect[3].real


# plot the result
plt.figure(2)
plt.plot(time, n_phot)

fig=plt.figure()
plt.subplot(4,1,1)
plt.plot(time,n_phot)
plt.xticks([])
#plt.ylim(0,10)
plt.ylabel('<${a^\dag}a$>')


plt.subplot(4,1,2)
plt.plot(time,spin_phot)
plt.xticks([])
#plt.ylim(-1*(10**7),1*(10**7))
plt.ylabel('<${a^\dag}{S^-}$>')

plt.subplot(4,1,3)
plt.plot(time,spin_spin)
plt.xticks([])
#plt.ylim(0,0.2)
plt.ylabel ('<${S^+}{S^-}$>/N')

plt.subplot(4,1,4)
plt.plot(time,inversion)
#plt.ylim(-1,1)
plt.ylabel('<${S^z}$>')
    
#xlabel position 
fig.text(0.5,0.04,'time/microsecond',ha='center')

plt.show()



