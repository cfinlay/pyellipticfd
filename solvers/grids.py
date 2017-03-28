from collections import defaultdict
import numpy as np
import stencils


class FDMesh(object):
    """A mesh for finite differences."""
    def __init__(self):
        self.dim = 0
        self.vertices = np.empty(shape=(0,0))
        self.bbox = np.empty(shape=(0,0))
        self.num_vertices = 0
        self.interior = np.empty(shape=0,dtype=np.intp)
        self.num_interior = 0
        self.boundary = np.empty(shape=0,dtype=np.intp)
        self.num_boundary = 0
        self.interior_neighbours = np.empty(shape=(0,0), dtype = np.intp)
        self.boundary_neighbours = np.empty(shape=(0,0), dtype = np.intp)

    @property
    def nodes(self):
        return self.vertices

    @nodes.setter
    def nodes(self, vertices):
        self.vertices = vertices

    @property
    def num_nodes(self):
        return self.num_vertices

class Grid(FDMesh):
    """A rectangular grid for finite differences."""

    def __init__(self, shape, bounds, stencil_radius):
        super().__init__()

        self.dim = len(shape)
        self.bbox = bounds

        self.stencil_radius = stencil_radius

        # define appropriate stencil for calculating neighbours
        stcl = stencils.create(self.stencil_radius,self.dim)



        # --- Vertices ---
        points = [np.linspace(bounds[0,i],bounds[1,i],num = shape[i]) for i in range(self.dim)]

        # grid indices
        grid = np.indices(shape)

        # meshgrid, with matrix indexing
        X = np.meshgrid(*points,sparse=False,indexing='ij')
        self.vertices = np.reshape(X,(self.dim,np.prod(shape))).T
        self.num_vertices = self.vertices.shape[0]



        # --- Interior indices ---
        mask_int = [slice(None)];
        for i in range(self.dim):
            mask_int.append(slice(1,-1))

        interior = grid[mask_int]

        self.interior = np.reshape(np.ravel_multi_index(interior,shape),-1)
        self.num_interior = self.interior.size



        # --- Boundary indices ---
        big_enough = [grid[i] == shape[i]-1 for i in range(self.dim)]
        big_enough = np.array(big_enough)
        big_enough = big_enough.any(axis=0)

        mask_bdry = np.logical_or((grid==0).any(0),big_enough)
        boundary = grid[:, mask_bdry]

        self.boundary = np.ravel_multi_index(boundary,shape)
        self.num_boundary = self.boundary.size



        # --- Compute interior neighbour indices ---

        # neighbours
        desired_shape = np.concatenate([[stcl.shape[1]],
                                        np.ones(self.dim,dtype=np.intp),
                                        [stcl.shape[0]]])
        neighbours = np.reshape(stcl.T,desired_shape) + np.expand_dims(interior,axis=self.dim+1)

        # mask of domain neighbours
        small_enough = [neighbours[i] < shape[i] for i in range(self.dim)]
        small_enough = np.array(small_enough)
        small_enough = small_enough.all(axis=0)

        mask_domain = np.logical_and(neighbours>=0,small_enough)
        mask_domain = mask_domain.all(axis=0)

        # raveled index of central interior stencil point
        interior_ix = np.ravel_multi_index(interior,shape)
        I = np.tile(np.expand_dims(interior_ix,axis=self.dim+1),stcl.shape[0])
        I = I[mask_domain]

        # raveled index of neighbour
        J = np.ravel_multi_index(neighbours[:,mask_domain],shape)

        self.interior_neighbours = np.array([I,J]).T



        # --- Compute boundary neighbour indices ---

        # neighbours
        desired_shape = np.concatenate([[stcl.shape[1]],
                                        [1],
                                        [stcl.shape[0]]])
        neighbours = np.reshape(stcl.T,desired_shape) + np.expand_dims(boundary,axis=self.dim+1)

        # mask of domain neighbours
        small_enough = [neighbours[i] < shape[i] for i in range(self.dim)]
        small_enough = np.array(small_enough)
        small_enough = small_enough.all(axis=0)

        mask_domain = np.logical_and(neighbours>=0,small_enough)
        mask_domain = mask_domain.all(axis=0)

        # raveled index of central boundary stencil point
        boundary_ix = np.ravel_multi_index(boundary,shape)
        I = np.tile(np.expand_dims(boundary_ix,axis=self.dim+1),stcl.shape[0])
        I = I[mask_domain]

        # raveled index of neighbour
        J = np.ravel_multi_index(neighbours[:,mask_domain],shape)

        self.boundary_neighbours = np.array([I,J]).T

    @property
    def stencil_radius(self):
        return self.__stencil_radius

    @stencil_radius.setter
    def stencil_radius(self, r):
        if r<=0:
            raise ValueError('stencil radius must be positive.')
        if type(r)!=int:
            raise TypeError('stencil radius must be integer valued.')
        self.__stencil_radius = r
