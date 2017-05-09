import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull


class FDGraph(object):
    """A graph for finite differences."""

    def __init__(self,vertices,angular_resolution=np.pi/3,
            get_pairs = False, colinear_tol = 1e-4,
            **kwargs):
        """Create a FDGraph object.

        Parameters
        ----------
        vertices : array
            An NxD array, listing N points in D dimensions.
        angular_resolution : float
            The angular resolution of the graph. Defaults to pi/3
        get_pairs : bool, optional
            Whether to search for antipodal pairs. Use this option
            only when working on symmetric grids. Defaults to False.
        colinear_tol : float, optional
            Tolerance for detecting colinear points. Defaults to 1e-4.

        **kwargs
            interior : array
                Indices for interior points.
            boundary : array
                Indices for boundary points.
            edges : array
                An array of graph edges, with two columns. Each row
                is a pair of indices specifying an edge.
            adjacency : scipy.sparse
                An NxN adjacency matrix.
            neighbours : array
                An array of neighbours, with two columns.
                The first column gives the index of the centre stencil point;
                the second column gives the index of a neighbour point.
            min_edge_length : float
                Minimum edge length between graph points.
                Only needed if creating the graph with 'neighbours'.
            resolution : float
                The spatial resolution of the graph. Every ball with radius 'resolution'
                contains at least one point. Required when creating the graph with
                'neighbours'. If not provided when creating the graph with 'edges'
                or 'adjacency', a heuristic guess is made.

        Notes
        -----
        - You must specify one of 'neighbours', 'adjacency', and 'edges'.
        - You must specify at least one of 'interior' and 'boundary'.
        """

        self.vertices = vertices
        self.angular_resolution = angular_resolution


        try:
            self.interior = kwargs['interior']
            try:
                self.boundary = kwargs['boundary']
            except KeyError:
                mask = np.in1d(self.indices,self.interior,invert=True)
                self.boundary = self.indices[mask]
        except KeyError:
            try:
                self.boundary = kwargs['boundary']
                mask = np.in1d(self.indices,self.boundary,invert=True)
                self.interior = self.indices[mask]
            except KeyError:
                raise TypeError("""Please provide either an array of boundary or interior indices""")


        if not ('edges' in kwargs or 'adjacency' in kwargs or 'neighbours' in kwargs):
            raise TypeError('One of "adjacency", "edges" or "neighbours" must be provided')

        if ('edges' in kwargs and 'adjacency' in kwargs):
            raise TypeError('Specify only one of "adjacency", "edges", or "neighbours"')

        if ('edges' in kwargs or 'adjacency' in kwargs) and 'neighbours' in kwargs:
            raise TypeError('Specify only one of "adjacency", "edges", or "neighbours"')


        if ('neighbours' not in kwargs):
            try:
                A = kwargs['adjacency'].tocoo()
                I = A.row
                J = A.col
                edges = np.array([I,J]).T

            except KeyError:
                edges = kwargs['edges']
                I = edges[:,0]
                J = edges[:,1]
                A = coo_matrix((np.ones(I.size, dtype=np.intp),(I,J)),
                        shape=(self.num_nodes,self.num_nodes), dtype = np.intp)

            # Convert a directed graph into an undirected graph
            if (A != A.T).nnz != 0:
                A = A + A.T
                A = coo_matrix((np.ones(A.nnz, dtype=np.intp),(A.rows,A.cols)),
                        shape=A.shape, dtype=np.intp)

                I = A.row
                J = A.col
                edges = np.array([I,J]).T

            adjacency = A.tocsr()

            try:
                self.min_edge_length = kwargs['min_edge_length']
            except KeyError:
                V = self.vertices[I]-self.vertices[J]
                Dist = np.linalg.norm(V,axis=1)
                self.min_edge_length = Dist.min()

            try:
                self.resolution = kwargs['resolution']
            except KeyError:
                print('keyerror')
                V = self.vertices[I]-self.vertices[J]
                Dist = np.linalg.norm(V,axis=1)
                self.resolution = Dist.max()

            if self.depth > 1:
                # Find all vertices 'depth' away from current point.
                A_pows = [adjacency]
                for k in range(1,self.depth):
                    A_pows.append( A_pows[k-1].dot(adjacency) )
                S = sum(A_pows)
                S = S.tocoo(copy=False)
                self.neighbours = np.array([S.row, S.col]).T

                # Only use neighbours within search radius
                self._limit_search_radius()

                # Remove any redundant stencil directions
                self._remove_colinear_neighbours(colinear_tol)
            else:
                self.neighbours = edges

        else:
            self.min_edge_length = kwargs['min_edge_length']
            self.resolution = kwargs['resolution']
            self.neighbours = kwargs['neighbours']

        # Compute finite difference simplices
        self._compute_simplices()

        # TODO: don't limit with min search radius
        if get_pairs:
            self._compute_pairs(colinear_tol)


    def __repr__(self):
        return ("FDGraph in {0.dim}D with {0.num_vertices} vertices "
                "and spatial resolution {0.resolution:.3g}").format(self)

    def _limit_search_radius(self):
        I, J = self.neighbours[:,0], self.neighbours[:,1] # index of point and its neighbours

        V = self.vertices[I] - self.vertices[J] # stencil vectors
        Dist = np.linalg.norm(V,axis=1)         # length of stencil vector

        mask = np.logical_and(Dist <= self.max_radius, Dist >= self.min_radius)
        I, J = I[mask], J[mask]
        self.neighbours = np.array([I, J]).T



    def _compute_simplices(self):
        """Compute finite difference simplices from neighbours."""

        I, J = self.neighbours[:,0], self.neighbours[:,1] # index of point and its neighbours

        V = self.vertices[I] - self.vertices[J] # stencil vectors
        Dist = np.linalg.norm(V,axis=1)         # length of stencil vector
        Vs = V/Dist[:,None]                     # stencil vector, normalized

        def get_simplices(i):
            mask = I==i
            nb_ix = J[mask]
            vs = Vs[mask]

            hull = ConvexHull(vs)

            simplex = nb_ix[hull.simplices]
            i_array = np.full((simplex.shape[0],1),i)


            return np.concatenate( [i_array, simplex], -1)

        self.simplices = np.concatenate([get_simplices(i) for i in range(self.num_nodes)])

    def _remove_colinear_neighbours(self, colinear_tol):

        I, J = self.neighbours[:,0], self.neighbours[:,1] # index of point and its neighbours
        mask = I!=J
        I, J = I[mask], J[mask]                           # centre point is not a neighbour

        V = self.vertices[I] - self.vertices[J] # stencil vectors
        Dist = np.linalg.norm(V,axis=1)         # length of stencil vector
        Vs = V/Dist[:,None]                     # stencil vector, normalized

        # Strip redundant directions
        def strip_neighbours(i):
            mask = I==i
            nb_ix = J[mask]
            d = Dist[mask]
            vs = Vs[mask]

            cos = vs.dot(vs.T)
            check = cos > 1 - colinear_tol
            check = np.triu(check)

            ix = np.arange(nb_ix.size)

            keep = list({ix[r][np.argmin(d[r])] for r in check})

            i_array = np.full((len(keep),1),i)
            return np.concatenate([ i_array, nb_ix[keep,None] ], -1)

        self.neighbours = np.concatenate([strip_neighbours(i) for i in range(self.num_nodes)])


    def _compute_pairs(self,colinear_tol):

        I, J = self.neighbours[:,0], self.neighbours[:,1] # index of point and its neighbours

        V = self.vertices[I] - self.vertices[J] # stencil vectors
        Dist = np.linalg.norm(V,axis=1)         # length of stencil vector
        Vs = V/Dist[:,None]                     # stencil vector, normalized

        def get_pairs(i):
            mask = I==i
            nb_ix = J[mask]
            vs = Vs[mask]

            cos = vs.dot(vs.T)
            check = cos < -1 + colinear_tol
            check = np.triu(check)
            if not check.any():
                raise TypeError("Point {0} has no pairs".format(i))

            ix = np.indices(check.shape)
            pairs = ix[:, check].T

            i_array = np.full((len(pairs),1),i)
            return  np.concatenate([ i_array, nb_ix[pairs] ], -1)

        self.pairs = np.concatenate([get_pairs(i) for i in self.interior])

    @property
    def dim(self):
        return self.vertices.shape[1]

    @property
    def Cd(self):
        if self.dim==2:
            return 2
        elif self.dim==3:
            return 1 + 2/np.sqrt(3)

    @property
    def num_vertices(self):
        return self.vertices.shape[0]

    @property
    def indices(self):
        return np.arange(self.num_vertices)

    @property
    def nodes(self):
        return self.vertices

    @property
    def num_nodes(self):
        return self.num_vertices

    @property
    def num_interior(self):
        return self.interior.size

    @property
    def num_boundary(self):
        return self.boundary.size

    @property
    def min_edge_length(self):
        return self._l

    @min_edge_length.setter
    def min_edge_length(self,l):
        if l>0:
            self._l = l
        else:
            raise TypeError("min_edge_length must be greater than 0")

    @property
    def resolution(self):
        return self._h

    @resolution.setter
    def resolution(self,h):
        if h>0:
            self._h = h
        else:
            raise TypeError("resolution must be greater than 0")

    @property
    def angular_resolution(self):
        return self._dtheta

    @angular_resolution.setter
    def angular_resolution(self,dtheta):
        if dtheta>0 and dtheta < np.pi:
            self._dtheta = dtheta
        else:
            raise TypeError("angular_resolution must be greater than 0 and less than pi")

    @property
    def max_radius(self):
        h = self.resolution
        th = self.angular_resolution

        return self.Cd * h * (1+ np.cos(th/2)/np.tan(th/2) + np.sin(th/2))

    @property
    def min_radius(self):
        h = self.resolution

        return self.max_radius - 2*self.Cd*h

    @property
    def depth(self):
        return int(np.ceil(self.max_radius/self.min_edge_length))

    @property
    def bbox(self):
        return np.array([np.amin(self.vertices,0), np.amax(self.vertices,0)])
