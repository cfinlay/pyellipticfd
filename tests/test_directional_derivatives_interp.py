from context import solvers, utils
from solvers import directional_derivatives_interp as ddi
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

U = obstacle_fcn(X,Y)
Lambda,Theta = ddi.d2eigs(U,dx)
