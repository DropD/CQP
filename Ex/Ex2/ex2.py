#!/usr/bin/python

from numerov import numerov
import numpy as np

def VBound(c):
    return lambda x : 0 <= x <= 1 and c * (x**2 - x) or 0

def k_gen(V):
    return lambda x : 2 * (E - V(x))

if __name__ == '__main__':
    dx = 0.001
    c  = 1e3
    integrator = numerov(dx)
    V = VBound(c)
    k = k_gen(V)

    # choose a point b
    b = 0.5
    E = V(b)*1e2

    psiL0 = np.exp(-1j * np.sqrt(np.complex(k(0))) * -dx)
    psiR0 = np.exp(-1j * np.sqrt(np.complex(k(1))) * (1+dx))
    psiL1 = psiR1 = 1

    pli = [psiL0, psiL1]
    pri = [psiR0, psiR1]

    psiL = integrator.integrate(0, b, k, pli)
    integrator.dx = -dx
    psiR = integrator.integrate(1, b, k, pri)

    numnodes = 0

    for i in range(0, len(psiL)-1):
        if psiL[i+1] * psiL[i] < 0:
            numnodes += 1

    for i in range(0, len(psiR)-1):
        if psiR[i+1] * psiR[i] < 0:
            numnodes += 1

    dlpsiL = (psiL[-1] - psiL[-2]) / dx / psiL[-1]
    dlpsiR = (psiR[-1] - psiR[-2]) / dx / psiR[-1]

    print dlpsiL - dlpsiR
    print numnodes

    
