import numpy as np
from context import solvers, utils
from solvers import finite_difference_matrices as fdm

# Set up computational domain 
Nx = 5                       #grid size
dx = 2./(Nx-1)               #grid resolution
x = np.linspace(-1.,1.,num = Nx) #gridpoints in x coordinate
X, Y = np.meshgrid(x,x,sparse=False,indexing='ij') # create x & y gridpoints, with matrix indexing
Ny = Nx

stencil = np.array([[1,0],
                    [1,1],
                    [0,1],
                    [-1,1]])
shape = (Nx,Ny)

D2 = fdm.d2(shape, stencil, dx) 
D1 = fdm.d1(shape, stencil, dx) 
