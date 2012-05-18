import os, sys
import numpy as np
import scipy.special as sp
import scipy.constants as const
import math
import scipy.special as sps
import matplotlib.pyplot as plt

eps = 5.9 # meV
sigma = 1 # unit of length
pref = 1/(6.12*sigma*sigma) # == hbar**2/2m, in units of meV and sigma

def V(r):
	t = pow(sigma/r,6)
	return eps*(t*t-2*t)

def Veff(r,l):
	return V(r) + pref*l*(l+1)/(r*r)

def F(l,r,E):
	return (V(r) + pref*l*(l+1)/(r*r) - E)/pref

def exact_inner_limit(r):
	C = math.sqrt(eps*6.12/25)
	return math.exp(-C*pow(r,-5))

def numerov(h,l,r0,rmax,E,initial):
	f = lambda t: F(l,t,E)
	
	w = lambda t,ut: (1-h*h/12*f(t))*ut
	u = lambda t,wt: (1-h*h/12*f(t))**(-1)*wt
	
	step = lambda t,wp,wt: 2*wt - wp + h*h*f(t)*u(t,wt)
	
	w0 = w(r0,initial[0])
	w1 = w(r0+h,initial[1])
	t = r0+h
	
	full = [(r0-h,w0),(r0,w1)]
	
	while t < rmax+h:
		w2 = step(t,w0,w1)
		
		w0 = w1
		w1 = w2
		t += h
		
		full.append((t,w2))
	
	full = np.array(map(lambda x: (x[0], u(x[0],x[1])), full[1:])).transpose()
	
	# line = plt.plot(full[0], full[1])[0]
	# line.set_label('l = %s' % l)
	
	return ( (t-h,u(t-h,w0)), (t,u(t,w1)), full )

def jl(l,x):
	return sps.sph_jn(l,x)[0][-1]

def nl(l,x):
	return sps.sph_yn(l,x)[0][-1]

def delta(E,l):
	# These values do not make a significant difference
	r0 = 0.5*sigma
	rmax = 5*sigma
	
	h = 0.01
	
	res = numerov(h, l, r0, rmax, E, (exact_inner_limit(r0), exact_inner_limit(r0+h)) )
	
	r0 = res[0][0]
	u0 = res[0][1]
	
	r1 = res[1][0]
	u1 = res[1][1]
	
	K = (r0*u1)/(r1*u0)
	k = math.sqrt(E/pref)
	tand = (K*jl(l,k*r0)-jl(l,k*r1))/(K*nl(l,k*r0)-nl(l,k*r1))
	delta_ = math.atan(tand)
	
	return delta_

def spartial(E,l):
    d = delta(E,l)
    k2 = E/pref
    s = (4*math.pi/k2) * (2*l+1)*math.sin(d)**2
    return s

def stot(E,lmax=10):
	#lmax = 10 # 10 is just enough
	
	deltas = [delta(E,l) for l in range(lmax)]
	s1 = [(2*l+1)*math.sin(deltas[l])**2 for l in range(lmax)]
		
	k2 = E/pref
	return (4*math.pi/k2)*sum(s1)


# MAIN

# potential
plt.figure()
plt.title('Effective potential')
xs = np.linspace(0.5, 5.5,1000)

for l in range(0,11):
    pot = [Veff(r,l) for r in xs]
    plt.plot(xs, pot, label='$l = %s$' % l)
plt.xlabel('$r / \\sigma$')
plt.ylabel('$V_{eff}(r)$ [meV]')
plt.ylim(-6.5,15)
plt.grid()
plt.legend(loc='best')


# sigmapartial vs. E
plt.figure()
plt.title('Partial waves scattering')
Erange = np.linspace(0.1, 3.5,500)

for l in [4,5,6]:
    sigmap = [spartial(E,l) for E in Erange]
    plt.plot(Erange, sigmap, label='$l = %s$' % l)
plt.xlabel('$E$ [meV]')
plt.ylabel('$\\sigma_{partial} / \\sigma^2$')    
plt.legend(loc='best')


# sigmatot vs. E
plt.figure()
plt.title('Total cross-section')
Erange = np.linspace(0.1, 3.5,500)

for l in [3,4,5,6,7,8,10,11]:
    sigmatot = [stot(E,l) for E in Erange]
    plt.plot(Erange, sigmatot, label='$l_{max} = %s$' % (l-1))
plt.xlabel('$E$ [meV]')
plt.ylabel('$\\sigma_{tot} / \\sigma^2$')    
plt.legend(loc='best')

plt.show()

