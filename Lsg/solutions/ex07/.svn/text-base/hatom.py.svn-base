import numpy as np
import scipy.linalg as la

alpha = [13.00773, 1.962079, 0.444529, 0.1219492]
dim = len(alpha)

def spq(p,q):
    return (np.pi/(alpha[p]+alpha[q]))**1.5

def tpq(p,q):
    return 3*alpha[p]*alpha[q]*np.pi**1.5/(alpha[p]+alpha[q])**2.5

def apq(p,q):
    return -2*np.pi/(alpha[p]+alpha[q])

h = np.array([[tpq(p,q)+apq(p,q) for q in range(dim)] for p in range(dim)]) # Hamiltonian
s = np.array([[spq(p,q)          for q in range(dim)] for p in range(dim)]) # overlap matrix
energies = la.eigh(h,s,eigvals_only=True)

print 'Ground state energy [Hartree]:', energies[0]