from context import solvers, utils
from solvers import convex_envelope
from utils import plot_utils
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas

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
A = np.array([-1/3,-2/3])
B = -A
G = np.minimum(np.sqrt((X+A[0])**2+(Y+A[1])**2),
                   np.sqrt((X+B[0])**2+(Y+B[1])**2))

# Gcvx is the true convex envelope
AB = B - A
AP = np.array([X-A[0],Y-A[1]])
t = (AP[0]*AB[0] +AP[1]*AB[1])/(AB[0]**2 + AB[1]**2)
Xbar = A[0] + t*AB[0]
Ybar = A[1] + t*AB[1]
Gcvx = np.sqrt((Xbar-X)**2 + (Ybar-Y)**2)
Gcvx[t<=0] = np.sqrt((X[t<=0]-A[0])**2 + (Y[t<=0]-A[1])**2)
Gcvx[t>=1] = np.sqrt((X[t>=1]-B[0])**2 + (Y[t>=1]-B[1])**2)


#stencil = np.array([[0,1],
#                    [1,0],
#                    [1,2],
#                    [2,1],
#                    [-1,2],
#                    [-2,1]])
#
#times = []
#iters = []
#diff = []
#names = []
#err = []
#
#st = time.time()
#Eul_grid = convex_envelope.euler_step(G,dx, solution_tol=1e-6,method='grid')
#times.append(time.time()-st)
#iters.append(Eul_grid[1])
#diff.append(Eul_grid[2])
#err.append(np.amax(np.abs(Eul_grid[0][1:-1,1:-1]-Gcvx[1:-1,1:-1])))
#names.append('Euler, grid')
#
#st = time.time()
#Eul_intp = convex_envelope.euler_step(G,dx, solution_tol=1e-6,method='interpolate')
#times.append(time.time()-st)
#iters.append(Eul_intp[1])
#diff.append(Eul_intp[2])
#err.append(np.amax(np.abs(Eul_intp[0][1:-1,1:-1]-Gcvx[1:-1,1:-1])))
#names.append('Euler, interp')
#
#st = time.time()
#Pol_grid = convex_envelope.policy_iteration(G,dx,solution_tol=1e-6,max_euler_iters=15,method='grid')
#times.append(time.time()-st)
#iters.append(Pol_grid[1])
#diff.append(Pol_grid[2])
#err.append(np.amax(np.abs(Pol_grid[0][1:-1,1:-1]-Gcvx[1:-1,1:-1])))
#names.append('Policy, grid')
#
#st = time.time()
#Pol_intp = convex_envelope.policy_iteration(G,dx,solution_tol=1e-6,max_euler_iters=15, method='interpolate')
#times.append( time.time()-st)
#iters.append(Pol_intp[1])
#diff.append(Pol_intp[2])
#err.append(np.amax(np.abs(Pol_intp[0][1:-1,1:-1]-Gcvx[1:-1,1:-1])))
#names.append('Policy, interp')
#
#st = time.time()
#Newton = convex_envelope.newton_method(G,dx,solution_tol=1e-6)
#times.append( time.time()-st)
#iters.append(Newton[1])
#diff.append(Newton[2])
#err.append(np.amax(np.abs(Newton[0][1:-1,1:-1]-Gcvx[1:-1,1:-1])))
#names.append('Newton, grid')
#
#st = time.time()
#LineSolver = convex_envelope.line_solver(G,stencil,solution_tol=1e-6)
#times.append( time.time()-st)
#iters.append(LineSolver[1])
#diff.append(LineSolver[2])
#err.append(np.amax(np.abs(LineSolver[0][1:-1,1:-1]-Gcvx[1:-1,1:-1])))
#names.append('Line Solver')
#
#d = {'Method': names, 'Time': times, 'Iters': iters, 'Diff': diff, 'Error': err}
#Stats = pandas.DataFrame(d)
#Stats = Stats[['Method','Time','Iters','Diff','Error']]
#
#print(Stats)
