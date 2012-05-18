# Plot binning analysis from all runs of the C++ program pimc in the current directory.

import numpy as np
import matplotlib.pyplot as plt
import pyalps
import pyalps.plot
import pyalps.load

runfiles = pyalps.getResultFiles(prefix='*.run')

loader = pyalps.load.Hdf5Loader()
ebinning = pyalps.flatten(loader.ReadBinningAnalysis(runfiles,measurements=['Energy'],respath='/simulation/realizations/0/clones/0/results'))
tbinning = pyalps.flatten(loader.ReadBinningAnalysis(runfiles,measurements=['KineticEnergy'],respath='/simulation/realizations/0/clones/0/results'))
vbinning = pyalps.flatten(loader.ReadBinningAnalysis(runfiles,measurements=['PotentialEnergy'],respath='/simulation/realizations/0/clones/0/results'))

for o in (ebinning,tbinning,vbinning):
    for d in o:
        d.props['label'] = str(d.props['SWEEPS'])+' sweeps'
    plt.figure()
    plt.title(o[0].props['observable'])
    plt.xlabel('binning level')
    plt.ylabel('error estimate')
    pyalps.plot.plot(o)
    plt.legend()

plt.show()