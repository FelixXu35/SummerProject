from qutip import *
from qutip.piqs import *
import numpy as np

jz = jspin(1, 'z')
jp = jspin(3, "+")
jm = jp.dag()
print(jz, sigmaz())