import numpy as np
from context import solvers, utils
from solvers import utils
from solvers.utils import hessian_eigenvals

# Set up computational domain 
Nx = 21                      #grid size
dx = 1./(Nx-1)               #grid resolution
x = np.linspace(0.,1.,num = Nx) #gridpoints in x coordinate
X, Y = np.meshgrid(x,x,sparse=False,indexing='ij') # create x & y gridpoints, with matrix indexing

U = 10000*X*(X-.25)*(X-.5)*(X-1)*Y*(Y-.5)*(Y-.75)*(Y-1)

lambda_max = solvers.utils.hessian_eigenvals.max(U,dx)
lambda_min = solvers.utils.hessian_eigenvals.min(U,dx)
