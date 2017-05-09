import grids
import numpy as np
import stencils

def uniform_grid(shape, bounds, stencil_radius):
    """Create a FDGraph object for a uniform grid."""

    dim = len(shape)
    npts = np.prod(shape)
    
    rectangle_shape = (bounds[1,:] - bounds[0,:])/(np.array(shape)-1)
    min_edge_length = np.amin(rectangle_shape)
    resolution = np.linalg.norm(rectangle_shape)/2

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

    #TODO: minimum search radius, angular resolution


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
        'neighbours': neighbours, 'depth' : stencil_radius, 'get_pairs': True,
        'min_edge_length' : min_edge_length,
        'resolution' : resolution})


def process_v(G,v,domain="interior"):
    """Utility function to process direction vector into correct format."""

    if domain=="interior":
        N = G.num_interior
    elif domain=="boundary":
        N = G.num_boundary
    elif domain=="all":
        N = G.num_nodes

    # v must be an array of vectors, a direction for each interior point
    v = np.array(v)
    if (v.size==1) & (G.dim==2):
        # v is a constant spherical coordinate, convert to vector for each point
        v = np.broadcast_to([np.cos(v), np.sin(v)], (N, G.dim))

    elif (v.size==2) & (G.dim==3):
        v = np.broadcast_to([np.sin(v[0])*np.cos(v[1]), np.sin(v[0])*np.sin(v[1]), np.cos(v[1])],
                (N, G.dim))

    elif v.size==G.dim:
        # v is already a vector, but constant for each point.
        # Broadcast to everypoint
        norm = np.linalg.norm(v)
        v = v/norm
        v = np.broadcast_to(v, (N, G.dim))

    elif (v.size==N) & (G.dim==2):
        # v is in spherical coordinates, convert to vector
        v = np.array([np.cos(v),np.sin(v)]).T

    elif (v.shape==(N,2)) & (G.dim==3):
        v = np.array([np.sin(v[:,0])*np.cos(v[:,1]),
            np.sin(v[:,0])*np.sin(v[:,1]),
            np.cos(v[:,1])]).T

    elif v.shape==(N,G.dim):
        #then v is a vector for each point, normalize
        norm = np.linalg.norm(v,axis=1)
        v = v/norm[:,None]

    return v
