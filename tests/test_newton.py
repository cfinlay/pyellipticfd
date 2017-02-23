import numpy as np
import matplotlib.pyplot as plt
import time

from context import solvers, utils
from utils import plot_utils
from solvers import convex_envelope as ce
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

st = time.time()
Newton = ce.newton_method(G,dx)
t = time.time() - st