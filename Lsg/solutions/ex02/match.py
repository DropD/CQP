import numpy as np
import matplotlib.pyplot as plt
from numpy import exp,sqrt,conj,absolute

# we set m=hbar=1

def numerov_step(k,x0,dx,psi0,psi1):
    dx2 = dx*dx
    c0 = 1 + dx2/12.*k(x0)
    c1 = 2 * (1 - 5.*dx2/12.*k(x0+dx))
    c2 = 1 + dx2/12.*k(x0+2*dx)
    return (c1*psi1 - c0*psi0) / c2

def wavefunction(k,x0,xn,dx,psi0,psi1):
    n = int(round((xn-x0)/dx))+1
    psi = np.zeros(n)
    psi[0] = psi0
    psi[1] = psi1
    for i in range(n-2):
        x = x0 + i*dx
        psi[i+2] = numerov_step(k,x,dx,psi[i],psi[i+1])
    return psi

def dlogpsi(k,x0,xn,dx,psi0,psi1):
    n = int(round((xn-x0)/dx))+1
    numnodes = 0
    for i in range(n-2):
        x = x0 + i*dx
        psi2 = numerov_step(k,x,dx,psi0,psi1)
        psi0 = psi1
        psi1 = psi2
        if psi0*psi1 < 0:   numnodes += 1
    dpsi = (psi1-psi0)/dx
    return (dpsi/psi1,numnodes)

def matchpsi(energy,cv,stepsize):
    v = lambda x: potential(cv,x)
    k = lambda x: 2*(energy-v(x))
    b = .5+sqrt(.25+energy/cv)
    matchx = round(b/stepsize)*stepsize
    psi0 = exp(-stepsize*sqrt(-2.*energy))
    (leftdlog,leftnodes) = dlogpsi(k,-stepsize,matchx,stepsize,psi0,1.)
    (rightdlog,rightnodes) = dlogpsi(k,1.+stepsize,matchx,-stepsize,psi0,1.)
    return (leftdlog-rightdlog,leftnodes+rightnodes,matchx,psi0,k)

def numboundstates(cv,stepsize=1.e-3):
    """
    Counting the number of bound stated by the number of nodes for a very high energy:
    energy = -cv * 10^-6
    """
    energy = -cv/1.e6
    (diff,nodes,matchx,psi0,k) = matchpsi(energy,cv,stepsize)
    if diff > 0:    nodes -= 1
    return nodes+1

def findstate(cv,numnodes=0,stepsize=1.e-3,stoptol=1.e-5,maxiter=100,verbose=False,plottrials=False):
    emin = -cv/4.
    emax = 0.
    it = 0
    while it < maxiter:
        it += 1
        energy = (emax+emin)/2.
        (diff,nodes,matchx,psi0,k) = matchpsi(energy,cv,stepsize)
        if verbose:
            print it, ': E='+str(energy), 'in [', emin, ',', emax, '],\tdiff='+str(diff),'\tnodes='+str(nodes)
        # Plot trial wave functions
        if plottrials:
            psil = wavefunction(k,-stepsize,matchx,stepsize,psi0,1.)
            psir = wavefunction(k,1.+stepsize,matchx,-stepsize,psi0,1.)
            alpha = psil[-1]/psir[-1]
            psi = np.concatenate((psil,alpha*psir[-1::-1]))
            psi /= stepsize*np.sum(psi)
            xgrid = np.linspace(-stepsize,1+stepsize,len(psi))
            plt.plot(xgrid,psi,label='$\psi(E='+str(energy)+')$')
        # converged
        if nodes == numnodes and absolute(diff) < stoptol:
            print 'Converged after', it, 'iterations.'
            psil = wavefunction(k,-stepsize,matchx,stepsize,psi0,1.)
            psir = wavefunction(k,1.+stepsize,matchx,-stepsize,psi0,1.)
            alpha = psil[-1]/psir[-1]
            psi = np.concatenate((psil,alpha*psir[-1::-1]))
            psi /= sqrt(stepsize*np.sum(psi*psi))
            return (energy,psi)
        # E too low
        elif nodes < numnodes or (nodes == numnodes and diff > 0):
            emin = energy
        # E too high
        else:
            emax = energy
    
def potential(cv,x):
    return cv*min(0.,x*x-x)


if __name__ == '__main__':
    cV = 300
    stepsize = 1.e-3
    # Computing number of bound states
    numstates = numboundstates(cV,stepsize)
    print '# of bound states:', numstates
    
    # Plotting (dlog PsiL - dlog PsiR) [for teaching purposes]
    Egrid = []
    dlogdiff = []
    currnodes = 0
    plt.figure()
    for E in np.linspace(-cV/4.,0,100):
        (diff,nodes,matchx,psi0,k) = matchpsi(E,cV,stepsize)
        if nodes == currnodes:
            Egrid.append(E)
            dlogdiff.append(diff)
        else:
            plt.plot(Egrid,dlogdiff,label='$n=%s$'%currnodes)
            Egrid = []
            dlogdiff = []
            currnodes = nodes
    plt.plot(Egrid,dlogdiff,label='$n=%s$'%currnodes)
    plt.xlabel('E')
    plt.ylabel('$d\\log \\psi_L - d\\log \\psi_R$')
    plt.ylim(-20,10)
    plt.grid()
    plt.legend(loc='best')
    
    # Computing eigenenergies and plotting wave function
    plt.figure()
    plt.title('$c=%s,\; dx=%s$'%(cV,stepsize))
    for n in range(numstates):
        print '+++ '+str(n)+'\'th bound state:'
        (E,psi) = findstate(cV,n,stepsize=stepsize)
        print 'E_'+str(n)+' =', E
        xgrid = np.linspace(-stepsize,1+stepsize,len(psi))
        psi *= cV/16./numstates / np.max(psi) # rescale psi to look nicer in the plot
        plt.plot(xgrid,xgrid*0+E,'b')
        plt.plot(xgrid,psi+E,'b',label='$\psi(E_%d=%4g)$' % (n,E),linewidth=2)
    pot = np.array([potential(cV,x) for x in xgrid])
    plt.plot(xgrid,pot,'r',label='$V(x)$')
    plt.legend(loc='best')
    plt.show()
