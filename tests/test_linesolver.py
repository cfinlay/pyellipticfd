#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 15:13:22 2017

@author: bilalabbasi
"""
from context import utils
from utils import plot_utils
import numpy as np
import matplotlib.pyplot as plt
import linesolver as ls

plt.close('all')

Nx = 2**7
Ny = 2**7
x = np.linspace(-1,1,Nx)
y = np.linspace(-1,1,Ny)
x,y = np.meshgrid(x,y)
stencil = np.array([[0,1],
                    [1,0],
                    [1,2],
                    [2,1],
                    [-1,2],
                    [-2,1]])

##Obstacle
#g = np.minimum(np.sqrt((x-0.5)**2+y**2),
#               np.sqrt((x+0.5)**2+(y-0.2)**2))

g = np.minimum(np.sqrt((x-0.5)**2+y**2),
               np.sqrt((x+0.5)**2+y**2)-np.pi/10)

#g = np.minimum(np.sqrt((x-0.5)**2+y**2),
#               np.sqrt((x+0.5)**2+y**2),
#                np.sin(x**2+y**2))

##Iterative procedure
tol = 10**-6
err = 10**6
max_its = 1000
U = np.copy(g)
its = 0
while err > tol and its < max_its:
    Uold = U
    U = ls.convex_linesolver(U,stencil)
    err = np.max(np.abs(U-Uold))
    its+=1

p = 'Convergence in {} iterations with a residual of {}.'
print(p.format(its,err))
plot_utils.plotter3d(x,y,U)

plt.figure()
plt.contour(x,y,U)
plt.show()

