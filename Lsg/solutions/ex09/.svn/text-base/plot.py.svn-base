import sys, os

import numpy as np
import matplotlib.pyplot as plt
import pyalps
from pyalps.plot import plot

files = pyalps.getResultFiles(dirname='data')
data = pyalps.loadMeasurements(files , ['|m|','m^2', 'Connected Susceptibility', 'Binder Cumulant U2'])

for d in pyalps.flatten(data):
    d.props['M/L'] = d.props['M'] / d.props['L']

m = pyalps.collectXY(data, 'Jx', '|m|', foreach=['L', 'M'])
chi = pyalps.collectXY(data, 'Jx', 'Connected Susceptibility', foreach=['L', 'M'])
binder = pyalps.collectXY(data, 'Jx', 'Binder Cumulant U2', foreach=['L', 'M'])


for d in pyalps.flatten(m):
    d.x = np.exp(2.*d.props['Jy'])*d.x
plt.figure()
plot(m)
plt.xlabel('$J/\\Gamma$')
plt.ylabel('magnetization')
plt.legend(loc='best', frameon=False)


for d in pyalps.flatten(chi):
    d.x = np.exp(2.*d.props['Jy'])*d.x
plt.figure()
plot(chi)
plt.xlabel('$J/\\Gamma$')
plt.ylabel('Connected Susceptibility, $\chi = [<m^2> -<|m|>^2] / L / M$')
plt.legend(loc='best', frameon=False)


for d in pyalps.flatten(binder):
    d.x = np.exp(2.*d.props['Jy'])*d.x
plt.figure()
plot(binder)
plt.axvspan(0.96, 1.02, color=(.4,.4,.4, .3)) # by zooming into the figure
plt.axvline(0.99, color=(1,.2,.2))
plt.xlabel('$J/\\Gamma$')
plt.ylabel('Binder Cumulant, $U_2 = <m^2> / <|m|>^2$')
plt.legend(loc='best', frameon=False)


plt.show()
