import numpy as np
import numpy.linalg as la
import os, sys
import matplotlib.pyplot as plt
from math import sqrt

def anharm_mtx(N,K,verbose=False):
	# set m = hbar = 1
	w = 1

	f = K/(4*w*w)

	mtx = np.zeros((N,N))
    
	for row in range(N):
		for col in range(N):
			n = min(row,col)
			if col == row:
				mtx[row,col] += w*row+0.5
				mtx[row,col] += f*3*(row**2+(row+1)**2)
			if col == row + 2 or col + 2 == row:
				mtx[row,col] += f*(4*n+6)*sqrt((n+1)*(n+2))
			if col == row + 4 or col + 4 == row:
			 	mtx[row,col] += f*sqrt((n+1)*(n+2)*(n+3)*(n+4))
	return mtx

def anharm(N,K,verbose=False):
	# set m = hbar = 1
	w = 1

	f = K/(4*w*w)

	mtx = np.zeros((N,N))
    
	for row in range(N):
		for col in range(N):
			n = min(row,col)
			if col == row:
				mtx[row,col] += w*row+0.5
				mtx[row,col] += f*3*(row**2+(row+1)**2)
			if col == row + 2 or col + 2 == row:
				mtx[row,col] += f*(4*n+6)*sqrt((n+1)*(n+2))
			if col == row + 4 or col + 4 == row:
			 	mtx[row,col] += f*sqrt((n+1)*(n+2)*(n+3)*(n+4))
	
	if verbose:
	    print '=== Hamiltonian for N=%s, K=%s ===' % (N, K)
	    print np.around(mtx, decimals=1)
	
	vals,vecs = la.eig(mtx)
	return np.sort(vals)
	
N = 100
keeps = 5
Ks = np.arange(0,5,0.1)

markers = ['+', 'x', '-', 'o']
colors = ['b', 'r', 'g', 'k', 'y']
plt.figure()
imark = 0
for N in [20,60,100]:
    vals = np.zeros((len(Ks), keeps))
    for k,K in enumerate(Ks):
        eigs = anharm(N=N, K=K)
        for i in range(keeps):
            vals[k,i] = eigs[i]
    icolor = 0
    for i in range(keeps):
        plt.plot(Ks, vals[:,i], markers[imark % len(markers)] + colors[icolor % len(colors)],
                 label='$E_%d, N=%s$' % (i, N))
        icolor += 1
    imark += 1
plt.legend(loc='best')
plt.xlabel('K')
plt.ylabel('E')
plt.grid()
plt.show()
