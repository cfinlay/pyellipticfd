from context import solvers
from solvers import directional_derivatives_interp as ddi
#from solvers import directional_derivatives_grid as ddg
from solvers import gridtools

import numpy as np


# Set up computational domain
N = 2**5+1;
d = 2;
xi = [0,1]

shape = [N for i in range(d)]
bounds = np.array([xi for i in range(d)]).T
r = 2

G = gridtools.uniform_grid(shape,bounds,r)
X = G.vertices[:,0]
Y = G.vertices[:,1]
if d==3:
    Z = G.vertices[:,2]
else:
    Z = 0

U1 = X**2 - Y**2 + Z**2
U2 = X - Y

if d==2:
    th = np.array([np.cos(np.pi/8), np.sin(np.pi/8)])
else:
    th = np.array([1/3,1/4,1/7])
    th /= np.linalg.norm(th)

d2, M = ddi.d2(U1,G,[1,0])
