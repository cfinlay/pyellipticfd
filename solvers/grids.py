import numpy as np
import stencils


class Grid(object):
    """A uniform grid."""

    def __init__(self,shape,bounds,stencil_radius=2):
        dim=len(shape)

        # --- Gridpoints in x coordinate ---
        points = [np.linspace(bounds[i,0],bounds[i,1],num = shape[i]) for i in range(dim)]

        # grid indices
        grid = np.indices(shape)

        # meshgrid, with matrix indexing
        X = np.meshgrid(*points,sparse=False,indexing='ij')
        self.x = np.reshape(X,(dim,np.prod(shape))).T



        # --- Interior indices ---
        mask_int = [slice(None)];
        for i in range(dim):
            mask_int.append(slice(1,-1))

        interior = grid[mask_int]

        self.int = np.reshape(np.ravel_multi_index(interior,shape),-1)
        self.nint = self.int.size



        # --- Boundary indices ---
        big_enough = [grid[i] == shape[i]-1 for i in range(dim)]
        big_enough = np.array(big_enough)
        big_enough = big_enough.any(axis=0)

        mask_bdry = np.logical_or((grid==0).any(0),big_enough)

        self.bdry = np.ravel_multi_index(grid[:,mask_bdry],shape)
        self.nbdry = self.bdry.size



        # --- Compute neighbour indices ---

        # define appropriate stencil
        stcl = stencils.create(stencil_radius,dim)

        # neighbours
        desired_shape = np.concatenate([[stcl.shape[1]],np.ones(dim,dtype=np.intp),[stcl.shape[0]]])
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

        mask_nbs_bdry = np.in1d(I,self.bdry)
        mask_nbs_int = np.logical_not(mask_nbs_bdry)

        nbs = np.array([I,J]).T

        self.int_nbs = nbs[mask_nbs_int,:]
        self.bdry_nbs = nbs[mask_nbs_bdry,:]
