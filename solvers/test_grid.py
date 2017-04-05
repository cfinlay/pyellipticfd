from gridtools import uniform_grid
import numpy as np
from scipy.spatial import ConvexHull


N = 2**4+1;
d = 3;
xi = [0,1]

shape = [N for i in range(d)]
bounds = np.array([xi for i in range(d)]).T
r = 2

G = uniform_grid(shape,bounds,r)
