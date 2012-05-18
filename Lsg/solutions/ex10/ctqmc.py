import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt
from sys import stdout
import pyalps
import pyalps.alea as alpsalea

# seed random number generator once at program start
rnd.seed(42)

class Worldline:
    def __init__(self,beta,gamma):
        self.beta  = beta  # inverse temperature \beta
        self.gamma = gamma # transverse field \Gamma
        self.s0    = 1  # spin at \tau=0
        self.kinks = [] # position of kinks (domain walls), sorted from 0 to beta
        self.magnobs = alpsalea.RealObservable('magnetization')
    
    def insertUpdate(self):
        # propose to add kinks at times ta, tb, flipping all spins in range (ta,tb)
        ta = rnd.uniform(0,self.beta)
        tb = rnd.uniform(0,self.beta)
        n = len(self.kinks)
        # acceptance ratio
        pacc = (self.beta*self.gamma)**2 / ((n+1.) * (n+2.))
        if pacc >= 1 or rnd.uniform(0,1) < pacc:
            self.kinks += (ta,tb)
            self.kinks.sort()
            if ta > tb: 
                self.s0 *= -1
    
    def removeUpdate(self):
        if len(self.kinks) == 0:
            return
        # propose to remove i'th and j'th kink
        i = j = rnd.randint(0,len(self.kinks))
        while j == i:
            j = rnd.randint(0,len(self.kinks))
        n = len(self.kinks)
        pacc = n*(n-1.) / (self.beta*self.gamma)**2
        if pacc >= 1 or rnd.uniform(0,1) < pacc:
            if i > j:
                del self.kinks[i]
                del self.kinks[j]
                self.s0 *= -1
            else:
                del self.kinks[j]
                del self.kinks[i]
    
    def update(self):
        if rnd.uniform(0,1) < .5:
            self.insertUpdate()
        else:
            self.removeUpdate()
    
    def measure(self):
        self.magnobs << (len(self.kinks) / (self.beta*self.gamma))


if __name__ == "__main__":
    steps = int(1e5)
    thermsteps = steps/5
    beta = 1.
    results = []
    for gamma in np.linspace(0.05,3,50):
        wl = Worldline(beta,gamma)
        for i in xrange(thermsteps):    wl.update()
        for i in xrange(steps):
            wl.update()
            wl.measure()
        print gamma, wl.magnobs.mean, '+-', wl.magnobs.error, '\ttau =', wl.magnobs.tau
        results.append([gamma, wl.magnobs.mean, wl.magnobs.error, wl.magnobs.tau])
    
    results = np.transpose(results)
    plt.figure()
    plt.errorbar(results[0],results[1],results[2],fmt='.',label='Monte Carlo')
    plt.plot(results[0],np.tanh(beta*results[0]),label='exact')
    plt.xlabel('$\\Gamma$')
    plt.ylabel('$\\langle \\sigma_x \\rangle$')
    plt.title('Ising spin in transverse field ($\\beta='+str(beta)+'$)')
    plt.legend(loc='best')
    plt.show()

