from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import matplotlib.ticker as ticker

import numpy as np
import distmesh as dm
from scipy.spatial import ConvexHull, Delaunay

from pyellipticfd.pointclasses import FDTriMesh

# Uniform grid on unit circle
# ---------------------------
def disc_mesh(h0, angular_resolution):
    hB = h0/2*np.tan(angular_resolution/2)
    c = np.max([hB*(-1 + 2/np.sin(angular_resolution/2)),
                2*hB/np.tan(angular_resolution/2)])
    fd = lambda p : np.sqrt((p**2).sum(1))-(1.0-c)
    p, _ = dm.distmesh2d(fd, dm.huniform, h0, (-1,-1,1,1),fig=None)

    # add a fine boundary resolution
    hB = h0/2*np.tan(angular_resolution/2)
    th = np.arange(0,2*np.pi,hB)
    if th[-2] == np.pi*2:
        th = th[0:-1]
    p = np.concatenate([p,np.array([np.cos(th),np.sin(th)]).T])
    boundary = np.arange(p.shape[0]-th.size,p.shape[0])
    interior = np.arange(p.shape[0]-th.size)
    dly = Delaunay(p)
    tri = dly.simplices

    Grid = FDTriMesh(p, tri,  num_interior=interior.size,
                   angular_resolution=angular_resolution,
                   bdry_normals=p[boundary])

    return Grid

def disc(h):

    # Uniform grid on unit circle
    # ---------------------------
    Grid = disc_mesh(h, np.pi*h**(1/3))
    tri = Grid.triangulation


    # Plotting
    # --------
    def plot_sol(U):
        x,y = Grid.points.T
        triang = mtri.Triangulation(x,y,tri)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(triang,U,cmap = plt.cm.CMRmap)

        tick_spacing = .5
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.show()

    return Grid, plot_sol
