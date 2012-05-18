import numpy as np
import matplotlib.pyplot as plt
from numpy import exp,sqrt,conj,absolute

def numerov(k,x0,dx,n,psi0,psi1):
    dx2 = dx*dx
    if dx>0:
        idx = lambda i: i
    else:
        idx = lambda i: -i-1
    psi = np.zeros(n)
    psi[idx(0)] = psi0
    psi[idx(1)] = psi1
    kprev = k(x0)
    kcurr = k(x0+dx)
    nodes = 0
    for i in range(2,n):
        x = x0 + i*dx
        knext = k(x)
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


stoptol = 0.0001
maxiter = 100
xmax = 20.
steps = int(200*xmax)
xgrid, dx = np.linspace(0, xmax, steps+1, retstep=True)
xgrid = xgrid[1:]

def findstate(energy=None):
    upper = 0.
    lower = -10.
    if energy == None:
        energy = (upper+lower)/2.
    psimax = stoptol+1
    it = 0
    while it < maxiter:
        print it, ':', energy, 'in [', lower, ',', upper, ']',
        def V(x):
            return -1./x
        def k(x):
            return 2*(energy-V(x))
        it += 1
        psi0 = xmax*exp(-xmax)
        psi1 = (xmax-dx)*exp(-(xmax-dx))
        (psi, nodes) = numerov(k,xmax,-dx,steps,psi0,psi1)
        psimax = psi[0]
        print ', u(0) =', psimax
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
    return energy, zeros(steps)


E, psi = findstate()
print 'Found ground state energy =', E
psi /= sqrt(np.dot(psi,psi)*(dx))
exact = xgrid*exp(-xgrid)
exact /= sqrt(np.dot(exact,exact)*(dx))
plt.figure()
plt.title('Hydrogen $l=0$ wavefunction')
plt.plot(xgrid, psi, label='numerov')
plt.plot(xgrid, exact, label='exact')
plt.xlabel('r')
plt.ylabel('u(r)')
plt.legend(loc='best')

# reversing grid order: [dx, xmax]
# dx = -dx
# xgrid = xgrid[::-1]
# psi = psi[::-1]
# exact = exact[::-1]

plt.figure()
Uexact = -(xgrid+1)*exp(-2*xgrid)+1
# U0 = Uexact[0]
# U1 = Uexact[1]
U0 = 0.
U1 = dx
ddU = lambda i: -psi[i]**2 / xgrid[i]
U = verlet(U0,U1,ddU,dx,steps)
alpha = (U[-1]-1.)/xgrid[-1]
U -= alpha*xgrid
plt.title('Hartree potential')
plt.plot(xgrid, U, label='verlet')
plt.plot(xgrid, Uexact, label='exact')
plt.xlabel('r')
plt.ylabel('U(r)')
plt.legend(loc='best')

plt.show()
