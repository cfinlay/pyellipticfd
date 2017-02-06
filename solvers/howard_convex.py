import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import utils.plot_utils
import numpy as np
import directional_derivatives_interp as ddi
import warnings

# Set up computational domain 
Nx = 41                      #grid size
dx = 2./(Nx-1)               #grid resolution
x = np.linspace(-1.,1.,num = Nx) #gridpoints in x coordinate
X, Y = np.meshgrid(x,x,sparse=False,indexing='ij') # create x & y gridpoints, with matrix indexing

# Define an obstacle
a = (-1/3,-2/3)
b = (1/3,2/3)
def obstacle_fcn(X,Y):
    return np.minimum(np.sqrt((X-a[0])**2 + (Y-a[1])**2),
                      np.sqrt((X-b[0])**2 + (Y-b[1])**2))

# Calculate the obstacle and optionally plot it
G = obstacle_fcn(X,Y)

# initialize policy 
La0, Th0 = ddi.d2min(G,dx)

jmax = 5*1e2
imax = 1e4
euler_tol = 1e-2
solution_tol = 1e-4
policy_tol = 1e-2
dt = dx**2
U = np.copy(G)
i, j = 0, 0

while (i <imax):
    while (j < jmax):
        U_interior = U[1:-1,1:-1] + dt * np.minimum(ddi.d2(U,Th0,dx),
                                                  G[1:-1,1:-1] - U[1:-1,1:-1])

        solution_diff = np.amax(np.absolute(U[1:-1,1:-1] - U_interior))
        U[1:-1,1:-1] = U_interior

        if solution_diff < euler_tol:
            break
       
        j = j + 1


    [La0, Th] = ddi.d2min(U,dx)
    policy_diff = np.amax(np.abs(Th - Th0))
    Th0 = Th

    if policy_diff < policy_tol and solution_diff < solution_tol:
        break

    i = i+1

if j >= jmax:
    warnings.warn("Maximum policy iterations reached")
utils.plot_utils.plotter3d(X,Y,U)
