
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import numpy as np

from qutip import *
from qutip.piqs import *

import matplotlib.animation as animation
from IPython.display import HTML
from IPython.core.display import Image, display

N = 20
system = Dicke(N = N)
[jx, jy, jz] = jspin(N)
jp = jspin(N,"+")
jm = jp.dag()
w0 = 0.5
wx = 1.0
system.hamiltonian = w0 * jz + wx * jx
system.emission = 0.05
D_tls = system.liouvillian() 

system.dephasing = 0.1
D_tls = system.liouvillian()

steady_tls = steadystate(D_tls)
jz_ss = expect(jz, steady_tls)
jpjm_ss = expect(jp*jm, steady_tls)

rho0_tls = dicke(N, N/2, -N/2)