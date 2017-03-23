import numpy as np
import stencils


def create(shape,bounds,stencil_radius=2):
    """
    Create a uniform grid.
    """

    dim=len(shape)

    points = []
    for i in range(dim):
        points.append(np.linspace(bounds[i,0],bounds[i,1],num = shape[i])) # gridpoints in x coordinate

    # meshgrid, with matrix indexing
    X = np.meshgrid(*points,sparse=False,indexing='ij')

    # define appropriate stencil
    stcl = stencils.create(stencil_radius,dim)

    # grid indices, restrict to interior only
    grid = np.indices(shape)

    window = [slice(None)];
    for i in range(dim):
        window.append(slice(1,-1))

    grid = grid[window]


    # compute neighbour indices
    desired_shape = np.concatenate([[stcl.shape[1]],np.ones(dim,dtype=np.intp),[stcl.shape[0]]])
    neighbours = np.reshape(stcl.T,desired_shape) + np.expand_dims(grid,axis=dim+1)

    # Mask of domain neighbours
    small_enough = []
    for i in range(dim):
        small_enough.append(neighbours[i] < shape[i])
    small_enough = np.array(small_enough)
    small_enough = small_enough.all(axis=0)

    mask = np.logical_and(neighbours>=0,small_enough)
    mask = mask.all(axis=0)

    grid_ix = np.ravel_multi_index(grid,shape) # get raveled index

    # raveled index of centre stencil point
    I = np.tile(np.expand_dims(grid_ix,axis=dim+1),stcl.shape[0])
    I = I[mask]

    # raveled index of neighbour
    J = np.ravel_multi_index(neighbours[:,mask],shape)

    Rix = np.array([I,J])
    X = np.reshape(X,(dim,np.prod(shape)))

    return X, Rix
