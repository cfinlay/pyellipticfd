import numpy as np
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import matplotlib.ticker as ticker

from pyellipticfd import monge_ampere
from pyellipticfd.tests import setup_discs

Grid = setup_discs.disc_mesh(0.075,np.pi/4)

def Utrue(x):
    X, Y = x.T
    norm2 = X**2 + Y**2
    val = np.exp(norm2/2)
    return val

def neumann(x,n):
    X, Y = x.T

    return np.full_like(X,np.exp(1/2))

# Forcing function
def forcing(x):
    X, Y = x.T
    norm2 = X**2 + Y**2
    return -(1+norm2)*np.exp(norm2/2)


## Setup a plotter
#tri = Grid.triangulation
#x,y = Grid.points.T
#triang = mtri.Triangulation(x,y,tri)
#
#plt.ion()
#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#
#def plot_sol(U):
#    ax.cla()
#    ax.plot_trisurf(triang,U,cmap = plt.cm.CMRmap)
#    plt.draw()
#    plt.pause(0.001)
def main():

    Uma, diff, iters, t = monge_ampere.solve(Grid,forcing,
                            dirichlet = Utrue,verbose=True,
                            solver="newton",solution_tol=1e-8,operator_tol=1e-8)

    Err = np.abs(Uma-Utrue(Grid.points)).max()
    print('Error: %1.4g, Time: %0.2g, iters: %3d'%(Err,t, iters))

if __name__=="__main__":
    main()
