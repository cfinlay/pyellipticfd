from context import solvers, utils
from solvers import directional_derivatives_interp as ddi
from utils import plot_utils

import numpy as np

# Set up computational domain 
Nx = 21                      #grid size
dx = 2./(Nx-1)               #grid resolution
x = np.linspace(-1.,1.,num = Nx) #gridpoints in x coordinate
X, Y = np.meshgrid(x,x,sparse=False,indexing='ij') # create x & y gridpoints, with matrix indexing

class TestOnGrid:
    def __init__(self):
        self.U = X**2 - Y**2
        self.U2 = X - Y

    def test_d2eigs(self):
        Lambda,Theta = ddi.d2eigs(self.U,dx)
        assert ((np.abs(Lambda[0]+2)<1e-13).all() and
                (np.abs(Lambda[1]-2)<1e-13).all())
        assert ((np.abs(Theta[0])<1e-13).all() and
                (np.abs(Theta[1]-np.pi/2)<1e-13).all())

    def test_d2(self):
        Uxx = ddi.d2(self.U,0,dx)
        Uyy = ddi.d2(self.U,np.pi/2,dx)
        assert ((np.abs(Uxx-2) <1e-13).all() and 
                (np.abs(Uyy+2) <1e-13).all())

    def test_d1(self):
        Ux = ddi.d1(self.U2,0,dx)
        Uy = ddi.d1(self.U2,np.pi/2,dx)
        assert ((np.abs(Ux-1) <1e-13).all() and 
                (np.abs(Uy+1) <1e-13).all())

class TestOffGrid:
    def __init__(self):
        self.th = np.arctan(1/2)
        C = np.cos(self.th)
        S = np.sin(self.th)
        self.U = X**2 * (C**2 - S**2) + 4*C*S*X*Y + Y**2 * (S**2 - C**2)

    def test_d2eigs(self):
        Lambda,Theta = ddi.d2eigs(self.U,dx)
        assert ((np.abs(Lambda[0]+2)<.25).all() and
                (np.abs(Lambda[1]-2)<.25).all())
        assert ((np.abs(Theta[0]-self.th-np.pi/2)<0.1).all() and
                (np.abs(Theta[1]-self.th)<0.1).all())

    def test_d2(self):
        Uvv = ddi.d2(self.U,self.th,dx)
        Uww = ddi.d2(self.U,self.th+np.pi/2,dx)
        assert ((np.abs(Uvv-2) <.25).all() and 
                (np.abs(Uww+2) <.25).all())
