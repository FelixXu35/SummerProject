## Main.py
# This file is a part of summer project.
# This file is used to call all other functions and output result.
# There is no input.
# Written by Xiaotian Xu(Felix), 1st July 2020.

## Import
import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

## Defination
N = 1e15 # the number of twp-level particles
Kc = 2 * np.pi * 0.18 # the cavity mode decay rate(MHz)
Ks = 2 * np.pi * 0.11 # the spin dephasing rate (MHz)
