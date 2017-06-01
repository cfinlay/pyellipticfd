import numpy as np

from pyellipticfd import ddi
from pyellipticfd.fdclasses import FDRegularGrid

# Set up computational domain
N = 2**4-1;
d = 2;
xi = [0,1]

shape = [N for i in range(d)]
bounds = np.array([xi for i in range(d)]).T
r = 2

G = FDRegularGrid(shape,bounds,r)
X = G.vertices[:,0]
Y = G.vertices[:,1]
if d==3:
    Z = G.vertices[:,2]
else:
    Z = 0

U2 = X**2 - Y**2 + Z**2
U1 = X - Y

if d==2:
    th = np.array([np.cos(np.pi/8), np.sin(np.pi/8)])
else:
    th = np.array([1/3,1/4,1/7])
    th /= np.linalg.norm(th)

d1, M1 = ddi.d1(G,[1,0],U1, jacobian=True)
d2, M2 = ddi.d2(G,[1,0],U2, jacobian=True)
(d2min, M_min, v_min), (d2max, M_max, v_max) = ddi.d2eigs(G,U2, jacobian=True, control=True)
