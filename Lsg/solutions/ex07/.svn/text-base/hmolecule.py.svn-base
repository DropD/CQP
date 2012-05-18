import numpy as np
import scipy.linalg as la
from numpy import dot, pi, exp, sqrt
from scipy.special import erf

alpha = [13.00773, 1.962079, 0.444529, 0.1219492]*2
centr = [0]*4 + [1]*4   # center of basis functions
dim = len(alpha)

def kexp(i,j):
    """recurring factor in Gaussian overlap integrals"""
    return exp(-alpha[i]*alpha[j]/(alpha[i]+alpha[j])*(centr[i]-centr[j])**2)

def rp(i,j):
    """weighted center position R_P"""
    return (alpha[i]*centr[i] + alpha[j]*centr[j])/(alpha[i]+alpha[j])

def f0(q):
    """F_0(q)"""
    if q == 0: return 1
    return 0.5*sqrt(pi/q)*erf(sqrt(q))

def sij(i,j):
    """overlap matrix elements"""
    return (pi/(alpha[i]+alpha[j]))**1.5 * kexp(i,j)

def kinij(i,j):
    """kinetic energy matrix element"""
    a = alpha[i]; b = alpha[j]
    return a*b/(a+b) * (3 - 2*a*b/(a+b)*(centr[i]-centr[j])**2) * (pi/(a+b))**1.5 * kexp(i,j)

def nucij(i,j,rc):
    """nuclear attraction matrix element for nucleus at position rc"""
    a = alpha[i]; b = alpha[j]
    return -2*pi/(a+b) * kexp(i,j) * f0((a+b)*(rp(i,j)-rc)**2)

def tij(i,j):
    """non-interacting matrix elements"""
    return kinij(i,j) + nucij(i,j,0) + nucij(i,j,1)

def vijkl(i,j,k,l):
    """Hartree matrix elements"""
    aij = alpha[i]+alpha[j]
    akl = alpha[k]+alpha[l]
    q = aij*akl/(aij+akl) * (rp(i,j) - rp(k,l))**2
    return 2*sqrt(aij*akl/pi/(aij+akl))*sij(i,j)*sij(k,l)*f0(q)

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

e0 = 1 + 2*dot(dot(t,d),d) + dot(dot(dot(dot(v,d),d),d),d)
print 'Ground state energy [Hartree]:', e0