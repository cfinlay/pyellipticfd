import grids
import numpy as np
import stencils

def uniform_grid(shape, bounds, stencil_radius):

    dim = len(shape)
    npts = np.prod(shape)

    # define appropriate stencil for calculating neighbours
    stcl = stencils.create(stencil_radius,dim)


    # --- Vertices ---
    points = [np.linspace(bounds[0,i],bounds[1,i],num = shape[i]) for i in range(dim)]

    # grid indices
    grid = np.indices(shape)

    # meshgrid, with matrix indexing
    X = np.meshgrid(*points,sparse=False,indexing='ij')
    vertices = np.reshape(X,(dim,np.prod(shape))).T



    # --- Interior indices ---
    mask_int = [slice(None)];
    for i in range(dim):
        mask_int.append(slice(1,-1))

    interior = grid[mask_int]

    interior = np.reshape(np.ravel_multi_index(interior,shape),-1)



    # --- Boundary indices ---
    big_enough = [grid[i] == shape[i]-1 for i in range(dim)]
    big_enough = np.array(big_enough)
    big_enough = big_enough.any(axis=0)

    mask_bdry = np.logical_or((grid==0).any(0),big_enough)
    boundary = grid[:, mask_bdry]

    boundary = np.ravel_multi_index(boundary,shape)



    # --- Compute neighbour indices ---

    # neighbours
    desired_shape = np.concatenate([[stcl.shape[1]],
                                    np.ones(dim,dtype=np.intp),
                                    [stcl.shape[0]]])
    neighbours = np.reshape(stcl.T,desired_shape) + np.expand_dims(grid,axis=dim+1)

    # mask of domain neighbours
    small_enough = [neighbours[i] < shape[i] for i in range(dim)]
    small_enough = np.array(small_enough)
    small_enough = small_enough.all(axis=0)

    mask_domain = np.logical_and(neighbours>=0,small_enough)
    mask_domain = mask_domain.all(axis=0)

    # raveled index of central grid stencil point
    grid_ix = np.ravel_multi_index(grid,shape)
    I = np.tile(np.expand_dims(grid_ix,axis=dim+1),stcl.shape[0])
    I = I[mask_domain]

    # raveled index of neighbour
    J = np.ravel_multi_index(neighbours[:,mask_domain],shape)

    neighbours = np.array([I,J]).T

    return grids.FDGraph(vertices,**{'interior': interior, 'boundary': boundary,
        'neighbours': neighbours, 'depth' : stencil_radius, 'get_pairs': True})
