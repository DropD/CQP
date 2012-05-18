import math as m, numpy as np

from scipy.sparse.linalg as la
# Documentation at http://docs.scipy.org/doc/scipy/reference/sparse.linalg.html

d = # size of local basis
L = # length of the chain


def heisenberg(v):
"""
Given an input vector v, this function returns H*v.
"""
    pass

def sparse_diag():
	D = d**L
	H = la.LinearOperator( (D,D), matvec=heisenberg, dtype='D')
	(w,v) = la.eigsh(H, 10, which='SA')
	return w

