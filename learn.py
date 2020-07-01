#from qutip import *
import qutip as qt
import numpy as np

## test
#A = coherent_dm(5, 9)
A = qt.coherent_dm(5, 3)
A = A.ptrace()
print(A)