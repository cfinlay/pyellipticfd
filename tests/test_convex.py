from context import solvers, utils
from solvers import convex_envelope
from utils import plot_utils
import matplotlib.pyplot as plt
import numpy as np
import time

plt.close('all')

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


times = dict()
st = time.time()
Eul_grid = convex_envelope.euler_step(G,dx, solution_tol=1e-6,method='grid')
times['Euler, grid'] = time.time()-st

st = time.time()
Eul_intp = convex_envelope.euler_step(G,dx, solution_tol=1e-6,method='interpolate')
times['Euler, interpolate'] = time.time()-st

st = time.time()
Pol_grid = convex_envelope.policy_iteration(G,dx,solution_tol=1e-6,max_euler_iters=15,method='grid')
times['Policy, grid'] = time.time()-st

st = time.time()
Pol_intp = convex_envelope.policy_iteration(G,dx,solution_tol=1e-6,max_euler_iters=15, method='interpolate')
times['Policy, interpolate'] = time.time()-st

st = time.time()
Newton = convex_envelope.newton_method(G,dx,solution_tol=1e-6)
times['Newton, grid'] = time.time()-st
