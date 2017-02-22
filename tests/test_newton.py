from context import solvers, utils
from utils import plot_utils
import matplotlib.pyplot as plt

import itertools
import numpy as np
from solvers import finite_difference_matrices as fdm
from solvers import directional_derivatives_grid as ddg
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import warnings


# Set up computational domain
Nx = 41                      #grid size
dx = 2./(Nx-1)               #grid resolution
x = np.linspace(-1.,1.,num = Nx) #gridpoints in x coordinate
X, Y = np.meshgrid(x,x,sparse=False,indexing='ij') # create x & y gridpoints, with matrix indexing

#Pick an obstacle
#G = X**2 + Y**2
#G  = np.minimum(abs(X-0.5),abs(X+0.5))
#G = abs(np.sin(X**2+Y**2)-0.5)
#G = abs(np.sin(Y*np.pi)+np.cos(X*np.pi))
a = (-1/3,-2/3)
b = (1/3,2/3)
G = abs(np.minimum(np.sqrt((X+a[0])**2+(Y+a[1])**2),
                   np.sqrt((X+b[0])**2+(Y+b[1])**2)))

shape = (Nx,Nx)
Nx = shape[0]
Ny = shape[1]
N = np.prod(shape)

stencil = ddg.stencil

def operator(U):
    lambda1, Ix = ddg.d2min(U,dx)

    M = fdm.d2(U.shape, stencil, dx, Ix)

    b = -lambda1 > U[1:-1,1:-1]-G[1:-1,1:-1]
    Fu = U[1:-1,1:-1]-G[1:-1,1:-1]
    Fu[b] = -lambda1[b]

    b = np.reshape(b,(Nx-2)*(Ny-2))
    Fu = np.reshape(Fu,(Nx-2)*(Ny-2))

    ix_int = fdm.domain_indices(shape,1)[0]
    Grad = sparse.eye(N).tolil()
    Grad = Grad[ix_int,:]
    Grad[b,:] = -M[b,:]
    Grad = Grad[:,ix_int]
    Grad = Grad.tocsr()

    return Fu, Grad

def newton(U0,operator,solution_tol=1e-4,max_iters=1e2):
    U = np.copy(U0)
    Nx, Ny = U0.shape

    for i in itertools.count(1):
        Fu, Grad = operator(U)
        d = spsolve(Grad,-Fu)
        U_interior = U[1:-1,1:-1] + np.reshape(d,(Nx-2,Ny-2))

        diff = np.amax(np.absolute(U[1:-1,1:-1] - U_interior))
        U[1:-1,1:-1] = U_interior

        if diff < solution_tol:
            return U, i, diff
        elif i >= max_iters:
            warnings.warn("Maximum iterations reached")
            return U, i, diff

U = newton(G,operator,max_iters=50)
