from pathlib import Path
import pickle

import generate_grids

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as mtri
import matplotlib.ticker as ticker

def discs(n):
    ud = Path("unit_discs.p")

    if not ud.is_file():
        generate_grids.main()

    # Uniform grid on unit circle
    # ---------------------------
    Grids = pickle.load(open(ud,"rb"))
    Grid = Grids[n]['Grid']
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
