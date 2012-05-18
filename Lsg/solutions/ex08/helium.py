import numpy as np
import matplotlib.pyplot as plt
from numpy import exp,sqrt,conj,absolute
from math import pi

def numerov(k,x0,dx,n,psi0,psi1):
    dx2 = dx*dx
    if dx>0:
        idx = lambda i: i
    else:
        idx = lambda i: -i-1
    psi = np.zeros(n)
    psi[idx(0)] = psi0
    psi[idx(1)] = psi1
    kprev = k(idx(0))
    kcurr = k(idx(1))
    nodes = 0
    for i in range(2,n):
        # x = x0 + i*dx
        knext = k(idx(i))
        cprev = 1 + dx2/12.*kprev
        ccurr = 2 * (1 - 5.*dx2/12.*kcurr)
        cnext = 1 + dx2/12.*knext
        psi[idx(i)] = (ccurr*psi[idx(i-1)] - cprev*psi[idx(i-2)]) / cnext
        if psi[idx(i)]*psi[idx(i)] < 0:
            nodes += 1
        kprev = kcurr
        kcurr = knext
    return (psi,nodes)

def verlet(f0,f1,ddf,dx,n):
    dx2 = dx*dx
    if dx>0:
        idx = lambda i: i
    else:
        idx = lambda i: -i-1
    res = np.zeros(n)
    res[idx(0)] = f0
    res[idx(1)] = f1
    for i in range(2,n):
        res[idx(i)] = 2*res[idx(i-1)] - res[idx(i-2)] + dx**2*ddf(idx(i-1))
    return res


Etol = 0.001
stoptol = 0.0001
maxiter = 100
xmax = 10.
steps = int(1000*xmax)
xgrid, dx = np.linspace(0, xmax, steps+1, retstep=True)
xgrid = xgrid[1:]

def findstate(Vh,Vx):
    upper = 0.
    lower = -10.
    energy = (upper+lower)/2.
    psimax = stoptol+1
    it = 0
    while it < maxiter:
        # print it, ':', energy, 'in [', lower, ',', upper, ']',
        def V(i):
            return -2./xgrid[i] + Vh[i] + Vx[i]
        def k(i):
            return 2*(energy-V(i))
        it += 1
        psi0 = xmax*exp(-xmax)
        psi1 = (xmax-dx)*exp(-(xmax-dx))
        (psi,nodes) = numerov(k,xmax,-dx,steps,psi0,psi1)
        psimax = psi[0]
        if nodes == 0 and absolute(psimax) <= stoptol:
            print 'Converged after', it, 'iterations.'
            return energy, psi
        elif nodes == 0 and psimax > 0.:
            lower = energy
            energy = (upper+energy)/2.
        else:
            upper = energy
            energy = (lower+energy)/2.
    print 'Not converged in', maxiter, 'iterations. psimax =', psimax
    return energy, np.zeros(steps)


eps = 0.
itt = 0
u = np.zeros(steps)
Vh = np.zeros(steps)
Vx = np.zeros(steps)
while itt < maxiter:
    itt += 1
    epsold = eps
    eps, u = findstate(Vh, Vx)
    u /= sqrt(np.dot(u,u)*(dx))
    print '%s: eps = %s' % (itt, eps)
    
    U0 = 0.; U1 = dx;
    ddU = lambda i: -u[i]**2 / xgrid[i]
    U = verlet(U0,U1,ddU,dx,steps)
    alpha = (U[-1]-1.)/xgrid[-1]
    U -= alpha*xgrid
    
    Vx = -(1.5/pi**2 * u**2 / xgrid**2)**(1./3)
    Vh =  2.*U / xgrid
    
    if abs(eps-epsold) < Etol:
        print 'Self-consistency loop converged after %s iterations.' % itt
        print 'eps =', eps
        break

E = 2*eps - dx*np.dot(Vh, u**2) - .5*dx*np.dot(Vx, u**2)
print 'Energy =', E
