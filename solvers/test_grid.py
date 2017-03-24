from grids import *
import numpy as np

#def main():
N = 32;
d = 3;
xi = [0,1]

shape = [N for i in range(d)]
bounds = np.array([xi for i in range(d)])
r = 3

G = Grid(shape,bounds,r)

#if __name__ == '__main__':
    #main()
