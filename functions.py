## functions.py
# This file is a part of summer profect
# THis file is used to store all functions defined by writter himself.
# Written by Xiaotian Xu(Felix), 1st July 2020.

# get the lindbladian
# input the density matrix and the left operator
# the oputput is not a super operator, the result has the same dimension with the density matrix
def Lindblad(dm, oper):
    Lindbladian = 2 * oper * dm * oper.dag() - oper.dag() * oper * dm - dm * oper.dag() * oper
    return Lindbladian

