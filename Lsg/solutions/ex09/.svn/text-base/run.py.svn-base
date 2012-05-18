import numpy as np
from math import *

gamma = 1
delta = 0.05
J = np.linspace(0.5,2)
nsweeps = {8: 1000000, 16: 1000000, 24: 1000000,
          32: 1000000, 40: 1000000, 48: 1000000}

i = 0
n = 1
for Ji in J:
  if True:
    basename = 'start_ML1_%s.sh'%n
    fp = open(basename, 'w')
    print 'bsub -W 2:00 sh '+basename
    n += 1
  i += 1
  for ML in [1]:
    for L in reversed([16,24,32,40,48]):
     Jx = delta*Ji
     Jy = -.5*log(delta*gamma)
     fp.write('../ising %s %s %s %s %s\n' % (ML*L, L, Jy, Jx, nsweeps[L]))

