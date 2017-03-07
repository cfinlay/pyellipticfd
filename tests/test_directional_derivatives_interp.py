from context import solvers, utils
from solvers import directional_derivatives_interp as ddi
from solvers import directional_derivatives_grid as ddg
from utils import plot_utils

import numpy as np

# Set up computational domain
Nx = 21                      #grid size
dx = 2./(Nx-1)               #grid resolution
x = np.linspace(-1.,1.,num = Nx) #gridpoints in x coordinate
X, Y = np.meshgrid(x,x,sparse=False,indexing='ij') # create x & y gridpoints, with matrix indexing
Xint = X[1:-1,1:-1]
Yint = Y[1:-1,1:-1]


U1 = X**2 - Y**2
U2 = X - Y

th = np.arctan(1/2)
C = np.cos(th)
S = np.sin(th)
U3 = X**2 * (C**2 - S**2) + 4*C*S*X*Y + Y**2 * (S**2 - C**2)

#Test on grid
#Lambda1, Control1 = ddi.d2eigs(U1,dx)

#Test off grid
Li3, Ci3 = ddi.d2eigs(U3,dx)
Lg3, Cg3 = ddg.d2eigs(U3,dx,stencil=ddi.stencil[0:4])

#class TestOnGrid:
#    def test_d2eigs(self):
#        Lambda,Theta = ddi.d2eigs(U,dx)
#        assert ((np.abs(Lambda[0]+2)<1e-13).all() and
#                (np.abs(Lambda[1]-2)<1e-13).all())
#        assert ((np.abs(Theta[0]-np.pi/2)<1e-13).all() and
#                (np.abs(Theta[1])<1e-13).all())
#
#    def test_d2(self):
#        Uxx = ddi.d2(U,0,dx)
#        Uyy = ddi.d2(U,np.pi/2,dx)
#        assert ((np.abs(Uxx-2) <1e-13).all() and
#                (np.abs(Uyy+2) <1e-13).all())
#
#    def test_d1(self):
#        Ux = ddi.d1(U2,0,dx)
#        Uy = ddi.d1(U2,np.pi/2,dx)
#        U_1_1 = ddi.d1(U2,np.pi*5/4,dx)
#        U1_1 = ddi.d1(U2,np.pi*7/4,dx)
#        assert ((np.abs(Ux-1) <1e-13).all() and
#                (np.abs(Uy+1) <1e-13).all() and
#                (np.abs(U1_1-np.sqrt(2)) <1e-13).all())
#    def test_d1da(self):
#        V, T = ddi.d1da(U2,dx)
#        assert ((np.abs(V[0]+np.sqrt(2))<1e-13).all() and
#                (np.abs(V[1]-np.sqrt(2)<1e-13).all()))
#        assert ((np.abs(T[0] - np.pi*3/4)<1e-13).all() and
#                (np.abs(T[1] - np.pi*7/4)<1e-13).all())
#
#class TestOffGrid:
#    def test_d2eigs(self):
#        Lambda,Theta = ddi.d2eigs(U3,dx)
#        assert ((np.abs(Lambda[0]+2)<.25).all() and
#                (np.abs(Lambda[1]-2)<.25).all())
#        assert ((np.abs(Theta[0]-th-np.pi/2)<0.1).all() and
#                (np.abs(Theta[1]-th)<0.1).all())
#
#    def test_d2(self):
#        Uvv = ddi.d2(U3,th,dx)
#        Uww = ddi.d2(U3,th+np.pi/2,dx)
#        assert ((np.abs(Uvv-2) <.25).all() and
#                (np.abs(Uww+2) <.25).all())
