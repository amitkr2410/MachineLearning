from scipy.stats import qmc
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt #
import os
mpl.rcParams['savefig.format'] = "png"

def DesignPoint(column):
    #LogGamma, C [Linear, RBF, Sigmoid]
    sampler = qmc.LatinHypercube(d=2)
    sample = sampler.random(n=5)
    l_bounds = [0, 0]
    u_bounds = [2, 2]
    logans = qmc.scale(sample, l_bounds, u_bounds)
    ans = np.power(10, logans)

    figd, axd = plt.subplots(1,1,figsize=(5,5))
    plt.scatter(ans[:,0], ans[:, 1])
    plt.ylabel('C (Hyperparameter)')
    plt.xlabel('gamma (Hyperparameter)')
    plt.tight_layout()
    OutputFilename='DesignPoints_' + column +'.png'
    plt.savefig(OutputFilename)
    Command="open " + " "+OutputFilename
    os.system(Command)

    return ans
