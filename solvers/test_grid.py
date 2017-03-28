from grids import *
from stencils import *
import numpy as np

N = 2**9 + 1;
d = 2;
xi = [0,1]

shape = [N for i in range(d)]
bounds = np.array([xi for i in range(d)]).T
r = 2

G = Grid(shape,bounds,r)
