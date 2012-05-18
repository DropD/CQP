# Numerov integration of 1D scattering problems.
# Jan Gukelberger and Michele Dolfi (2011)
import numpy as np
import matplotlib.pyplot as plt
from numpy import exp,sqrt,conj,absolute,cos,sin

# we set m=hbar=1

def numerov_step(k,x0,dx,psi0,psi1):
    """
    One Numerov integration step.
    Arguments:  k       --  function k(x) = 2(E-V(x))
                x0      --  first point
                dx      --  discretization step
                psi0    --  \psi(x0)
                psi1    --  \psi(x0+dx)
    """
    dx2 = dx*dx
    c0 = 1 + dx2/12.*k(x0)
    c1 = 2 * (1 - 5.*dx2/12.*k(x0+dx))
    c2 = 1 + dx2/12.*k(x0+2*dx)
    return (c1*psi1 - c0*psi0) / c2

def wavefunction(k,x0,xn,dx,psi0,psi1):
    """
    Calculate a wave function by iterating numerov_step.
    Arguments:  k       --  function k(x) = 2(E-V(x))
                [x0,xn] --  integration range
                dx      --  discretization step
                psi0    --  \psi(x0)
                psi1    --  \psi(x0+dx)
    """
    n = int(round((xn-x0)/dx))+1
    psi = 1j*np.zeros(n)
    psi[0] = psi0
    psi[1] = psi1
    for i in xrange(n-2):
        x = x0 + i*dx
        psi[i+2] = numerov_step(k,x,dx,psi[i],psi[i+1])
    return psi

def transmission(V,xmax,en,dx):
    """
    Arguments:  V       --  potential function to be integrated
                xmax    --  range of the potential
                en      --  energy
                dx      --  integration step
    Returns a tuple of transmission and reflection rate for a system
    with the given potential between [0, xmax] at energy en.
    """
    k = lambda x: 2. * (en - V(x))
    xmin = -dx
    xmax += dx
    psi0 = exp(1j * dx * sqrt(k(xmax) + 0j))
    psi1 = 1.
    psi = wavefunction(k, xmax, xmin, -dx, psi0, psi1)
    
    qstart = sqrt(2. * en)
    exp_pos = exp(1j * qstart * dx)
    exp_neg = exp(-1j * qstart * dx)
    A = (psi[-1] - psi[-2] * exp_pos) / (exp_neg - exp_pos)
    B = psi[-2] - A
    T = 1. / absolute(A)**2
    R = absolute(B)**2 / absolute(A)**2
    return (T, R)

def rect_potential(x,w):
    """Rectangular potential of width w"""
    if x >= 0 and x < w:    return 1.
    else:                   return 0.

def transmission_exact(E,a):
    """
    Exact result for the transmission probability of a rectangular
    potential with V=1 for 0<x<a, and 0 otherwise
    """
    k = sqrt(2*(1. - E))
    if E == 0:
        return 0
    if E == 1:
        return 1. / (1. + a**2 / 2.)
    return 1. / (1. + np.sinh(k*a)**2 / (4.*E*(1.-E)))
    

class rect_exact_wavefunction:
    """
    Exact result of the wavefunction for a rectangular barrier
    betwwen 0 and a with strengh V.
    """
    def __init__(self, a, V, E):
        self.a = a
        self.V = V
        self.E = E
    def psi1(self, x):
        """Psi for x < 0"""
        k1 = sqrt( 2 * self.E )
        k2 = sqrt( 2 * (self.E - self.V) + 0j )
        C = exp(1j * self.a) / (4. * k1 * k2)
        A = C * ( 4. * k1 * k2 * cos(k2 * self.a) - 2j * (k1**2 + k2**2) * sin(k2*self.a) )
        B = C * 2j * (k2**2 - k1**2) * sin(k2 * self.a)
        return A*exp(1j*k1*x) + B*exp(-1j*k1*x)
    def psi2(self, x):
        """Psi for 0 <= x < a"""
        k1 = sqrt( 2 * self.E )
        k2 = sqrt( 2 * (self.E - self.V) + 0j )
        A = .5 * (k2 + k1) / k2 * exp( -1j * (k2 - k1) * self.a )
        B = .5 * (k2 - k1) / k2 * exp( 1j * (k2 + k1) * self.a )
        return A * exp(1j * k2 * x) + B * exp(-1j * k2 * x)
    def psi3(self, x):
        """Psi for a <= x"""
        k1 = sqrt( 2 * self.E)
        return 1. * exp(1j * k1 * x)
    def __call__(self, x):
        if x < 0: return self.psi1(x)
        elif x >= 0 and x < self.a: return self.psi2(x)
        else: return self.psi3(x)


# parameters for \psi(x) plots
E = 0.5
a = 1.
dx = .001
xmin = -5*a
xmax = 5*a

# transmission rates vs. E
Egrid = np.linspace(0, 1, 100)
rect_trans = np.zeros(len(Egrid))
rect_reflc = np.zeros(len(Egrid))
for i, Ei in enumerate(Egrid):
    (rect_trans[i], rect_reflc[i]) = transmission( lambda x: rect_potential(x,a), a, Ei, dx )
# exact transmission for square potential
trans_exact = np.zeros(len(Egrid))
trans_exact = [transmission_exact(Ei, a) for Ei in Egrid]
# plot
plt.figure()
plt.title('Transmission at barriers of width $a=%s$' % a)
plt.plot(Egrid, rect_trans, label='$T$ numerov')
plt.plot(Egrid, trans_exact, label='$T$ exact')
plt.legend(loc='center right')
plt.xlabel('Energy, $E/V_0$')
plt.ylabel('Transmission rate, $T = 1 / |A|^2$')
plt.grid()

# transmission rates vs. dx
dxgrid = np.array([.1, .01, .001, .0001])
rect_trans = np.zeros(len(dxgrid))
rect_reflc = np.zeros(len(dxgrid))
for i, dxi in enumerate(dxgrid):
    (rect_trans[i], rect_reflc[i]) = transmission( lambda x: rect_potential(x,a), a, E, dxi )
# exact transmission for square potential
trans_exact = transmission_exact(E, a)
# plot
plt.figure()
plt.title('Transmission at barriers of width $a=%s$ for $E/V_0=%s$' % (a, E))
plt.plot(dxgrid, rect_trans, 'o', label='$T$ numerov')
plt.plot(dxgrid, dxgrid*0+trans_exact, '--',label='$T$ exact')
plt.legend(loc='best')
plt.xscale('log')
plt.xlabel('$dx$')
plt.ylabel('Transmission rate, $T = 1 / |A|^2$')
plt.grid()

# rectangular - wave function
xgrid = np.linspace(xmax,xmin,(xmax-xmin)/dx+1)
psi_rect = wavefunction( lambda x: 2*(E-rect_potential(x,a)), xmax, xmin, -dx, 1., exp(-1j*sqrt(2*E)*dx) )
psi_rect_exact = rect_exact_wavefunction(a=a, V=1., E=E)
psi_exact = [psi_rect_exact(x) for x in xgrid]
plt.figure()
plt.title('Rectangular potential')
plt.xlabel('$x$')
plt.plot(xgrid,[rect_potential(x,a) for x in xgrid],'k',label='$V$')
plt.plot(xgrid,absolute(psi_rect)/np.max(absolute(psi_rect)),label='$|\psi|$') # normalized such that max |\psi|=1
plt.plot(xgrid,absolute(psi_exact)/np.max(absolute(psi_exact)),label='$|\psi|$ - analytic') # normalized such that max |\psi|=1
plt.legend()

# transmission rates vs. a
agrid = np.linspace(.01, 5, 100)
rect_trans = np.zeros(len(agrid))
rect_reflc = np.zeros(len(agrid))
for i, width in enumerate(agrid):
    (rect_trans[i], rect_reflc[i]) = transmission( lambda x: rect_potential(x,width), width, E, dx )
# exact transmission for square potential
trans_exact = np.zeros(len(agrid))
alpha = np.sqrt(2*E)
kk = np.sqrt(2*(1.-E))
trans_exact = [1/(1+np.sinh(a)**2) for i, a in enumerate(agrid)]
# plot
plt.figure()
plt.title('Transmission at barriers of width $a$ for $E/V_0=%s$' % E)
plt.plot(agrid, rect_trans, label='$T$')
# plt.plot(agrid, rect_reflc, label='$R$')
plt.plot(agrid, trans_exact, label='$T$ exact')
plt.legend(loc='center right')
plt.yscale('log')
plt.xlabel('$a$')
plt.ylabel('Transmission rate $T$')
plt.grid()

plt.show()
