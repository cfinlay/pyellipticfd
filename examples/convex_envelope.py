"""Solve for the convex envelope of two cones"""

import numpy as np

from pyellipticfd import  convex_envelope
from pyellipticfd.tests import setup_discs

r = 3/7
n = 2
rot = np.pi/5

C = r*np.array([[np.cos(2*i*np.pi/n+rot),np.sin(2*i*np.pi/n+rot)] for i in range(n)])

def obstacle(X):
    x,y = X.T
    v = np.array([np.sqrt((x + c[0])**2 + (y+c[1])**2) for c in C])

    return v.min(axis=0)

def Utrue(X):
    v = C[0] - C[1]
    X = X - C[1]
    s = X.dot(v)/np.linalg.norm(v)

    dist = np.zeros(X.shape[0])

    m  = np.logical_and(s>=0,s<=np.linalg.norm(v))
    t = np.linalg.norm(X.T-v[:,None]*s/np.linalg.norm(v),axis=0)
    dist[m] = t[m]


    m = np.logical_not(m)
    dist[m] = obstacle(X[m]+C[1])

    return dist

Grid, plotter = setup_discs.disc(0.05)


Uce,diff , iters, t = convex_envelope.solve(Grid,obstacle,
                    solver='newton',fdmethod='interpolate', solution_tol=1e-10)

Err = np.abs(Uce-Utrue(Grid.points)).max()
