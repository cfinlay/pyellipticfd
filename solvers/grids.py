#from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull


class FDGraph(object):
    """A graph for finite differences."""

    def __init__(self,vertices,**kwargs):
        """Create a FDGraph object.

        Parameters
        ----------
        vertices : array (double)
            An NxD array, listing N points in D dimensions.

        **kwargs
            get_pairs : bool
                Whether to search for antipodal pairs. Use this option
                only when working on symmetric grids. Defaults to False.
            radius : float
                Maximum neighbour search radius.
            colinear_tol : float
                Tolerance for detecting colinear points. Defaults to 1e-4.
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
            depth : int
                Maximum neighbour graph distance.

        Notes
        -----
        - You must specify one of 'neighbours', 'adjacency', or 'edges'.
        """

        self.vertices = vertices # array of points

        self.bbox = np.array([np.amin(self.vertices,0),
            np.amax(self.vertices,0)]) # bounding box

        try:
            get_pairs = kwargs['get_pairs']
        except KeyError:
            get_pairs = False

        try:
            self.radius = kwargs['radius']
        except KeyError:
            self.radius = None

        try:
            colinear_tol = kwargs['colinear_tol']
        except KeyError:
            colinear_tol = 1e-4


        try:
            self.interior = kwargs['interior']
        except KeyError:
            self.interior = np.empty(shape=0,dtype=np.intp)

        try:
            self.boundary = kwargs['boundary']
        except KeyError:
            self.boundary = np.empty(shape=0,dtype=np.intp)


        if not ('edges' in kwargs or 'adjacency' in kwargs or 'neighbours' in kwargs):
            raise TypeError('One of "adjacency", "edges" or "neighbours" must be provided')

        if ('edges' in kwargs and 'adjacency' in kwargs):
            raise TypeError('Specify only one of "adjacency", "edges", or "neighbours"')

        if ('edges' in kwargs or 'adjacency' in kwargs) and 'neighbours' in kwargs:
            raise TypeError('Specify only one of "adjacency", "edges", or "neighbours"')

        if ('neighbours' not in kwargs):
            try:
                adjacency = kwargs['adjacency'].tocsr()
            except KeyError:
                #TODO: turn directed graphs into undirected
                edges = kwargs['edges']
                I = edges[:,0]
                J = edges[:,1]
                adjacency = coo_matrix((np.ones(I.size, dtype=np.intp),(I,J)),
                        shape=(self.num_nodes,self.num_nodes), dtype = np.intp)
                adjacency = adjacency.tocsr()

            try:
                self.depth = kwargs['depth']
            except KeyError:
                self.depth = 1

            if self.depth > 1:
                # Find all vertices 'depth' away from current point.
                A_pows = [adjacency]
                for k in range(1,self.depth):
                    A_pows.append( A_pows[k-1].dot(adjacency) )
                S = sum(A_pows)
                S = S.tocoo(copy=False)
                self.neighbours = np.array([S.row, S.col]).T

                # Remove any redundant stencil directions
                self._remove_colinear_neighbours(colinear_tol)
            else:
                self.neighbours = edges
        else:
            try:
                self.depth = kwargs['depth']
            except KeyError:
                self.depth = None

            self.neighbours = kwargs['neighbours']

        # Only use neighbours within search radius
        if not self.radius==None:
            self._limit_search_radius()

        # Compute finite difference simplices
        self._compute_simplices()

        if get_pairs and self.interior.size!=0:
            self._compute_pairs(colinear_tol)
        elif get_pairs and self.interior.size==0:
            raise TypeError("Provide interior indices to compute finite difference pairs.")


    def __repr__(self):
        return ("FDGraph in {0.dim}D with {0.num_vertices} vertices "
                "and spatial resolution {0.resolution}").format(self)

    def _limit_search_radius(self):
        I, J = self.neighbours[:,0], self.neighbours[:,1] # index of point and its neighbours

        V = self.vertices[I] - self.vertices[J] # stencil vectors
        Dist = np.linalg.norm(V,axis=1)         # length of stencil vector

        mask = Dist <= radius
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

        self.resolution = max([np.amin(Dist[I==i]) for i in range(self.num_nodes)])

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
    def num_vertices(self):
        return self.vertices.shape[0]

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
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, d):
        if (type(d)==int and d>=1) or d==None:
            self._depth = d
        else:
            raise TypeError("depth must be integer valued and greater than 0")

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
