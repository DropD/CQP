import numpy as np
import scipy.linalg as la
from numpy import dot

alpha = [0.297104, 1.236745, 5.749982, 38.216677]
dim = len(alpha)

def sij(p,q):
    """overlap matrix elements"""
    return (np.pi/(alpha[p]+alpha[q]))**1.5

def tij(p,q):
    """non-interacting matrix elements"""
    return 3*alpha[p]*alpha[q]*np.pi**1.5/(alpha[p]+alpha[q])**2.5 - 4*np.pi/(alpha[p]+alpha[q])

def vijkl(i,j,k,l):
    """Hartree matrix elements"""
    return 2*np.pi**2.5/(alpha[i]+alpha[j])/(alpha[k]+alpha[l])/np.sqrt(alpha[i]+alpha[j]+alpha[k]+alpha[l])

s = np.array([[sij(p,q) for q in range(dim)] for p in range(dim)]) # overlap matrix
t = np.array([[tij(p,q) for q in range(dim)] for p in range(dim)]) # non-interacting matrix
v = np.array([[[[vijkl(i,j,k,l) for i in range(dim)] for j in range(dim)] 
                                for k in range(dim)] for l in range(dim)]) # Hartree matrix

d = np.ones(dim) # initial coefficient vector
d /= dot(d,dot(s,d)) # normalize

tol = 1e-10
eps = 0; oldeps = eps+2*tol
while abs(eps-oldeps) > tol:
    f = t + dot(dot(v,d),d) # Fock operator matrix
    ens,vecs = la.eigh(f,s) # solve GEV problem
    oldeps = eps
    minidx = np.argmin(ens)
    eps = ens[minidx]
    d  = vecs[:,minidx]
    d /= np.sqrt(dot(d,dot(s,d))) # normalize
    print 'eps =', eps

e0 = 2*dot(dot(t,d),d) + dot(dot(dot(dot(v,d),d),d),d)
print 'Ground state energy [Hartree]:', e0