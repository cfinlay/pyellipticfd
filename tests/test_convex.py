from context import solvers, utils
from solvers import convex_envelope
from utils import plot_utils

import numpy as np

# Set up computational domain 
Nx = 21                      #grid size
dx = 1./(Nx-1)               #grid resolution
x = np.linspace(0.,1.,num = Nx) #gridpoints in x coordinate
X, Y = np.meshgrid(x,x,sparse=False,indexing='ij') # create x & y gridpoints, with matrix indexing

# Define an obstacle
def obstacle_fcn(X,Y):
    return np.absolute((X-.25)*(X-.75))

# Calculate the obstacle and optionally plot it
G = obstacle_fcn(X,Y)
#plot_utils.plotter3d(X,Y,G)

U = convex_envelope.euler_step(G,dx)
plot_utils.plotter3d(X,Y,U)

#Utrue = np.zeros(U.size)
#Cases = (np.logical_or(X<=.25,X>=.75),
#         np.logical_and(X>.25,X<.75))
#
#Utrue[Cases[0]] = G[Cases[0]]
#Utrue[Cases[1]] = 0

