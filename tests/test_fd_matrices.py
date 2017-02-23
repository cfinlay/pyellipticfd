import numpy as np
from context import solvers, utils
from solvers import finite_difference_matrices as fdm
from solvers import directional_derivatives_grid as ddg

# Set up computational domain
Nx = 5                       #grid size
dx = 2./(Nx-1)               #grid resolution
x = np.linspace(-1.,1.,num = Nx) #gridpoints in x coordinate
X, Y = np.meshgrid(x,x,sparse=False,indexing='ij') # create x & y gridpoints, with matrix indexing
Ny = Nx
G = X**2 - Y**2

stencil = np.array([[1,0],
                    [1,1],
                    [0,1],
                    [-1,1]])
shape = (Nx,Ny)

D2 = fdm.d2(shape, stencil, dx)

lambda1, Ix = ddg.d2min(G,dx,stencil=stencil)
D2min = fdm.d2(G.shape,stencil,dx,ix=Ix)
