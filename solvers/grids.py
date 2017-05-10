"""Classes to structure data for finite differences."""

import itertools
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

class FDPointCloud(object):
    """Base class for finite differences on point clouds"""

    def __repr__(self):
        return ("FDPointCloud in {0.dim}D with {0.num_vertices} vertices").format(self)


    def __init__(self, vertices, angular_resolution=None,
            spatial_resolution=None, neighbours=None,
            boundary_resolution=None, dist_to_boundary=None,
            interior=None, boundary=None):
        """Initialize a finite difference point cloud.

        Parameters
        ----------
        vertices : array_like
            An NxD array, listing N points in D dimensions.
        angular_resolution : float
            The desired angular resolution.
        spatial_resolution : float
            The spatial resolution of the graph. Every ball with radius
            'spatial_resolution' contains at least one point.
        boundary_resolution : float
            The spatial resolution of the graph on the boundary. Every ball
            centered on the boundary, with radius 'boundary_resolution',
            contains at least onei boundary point.
        neighbours : array_like
            An array of neighbours, with two columns.
            The first column gives the index of the centre stencil point;
            the second column gives the index of a neighbour point.
        dist_to_boundary : float
            The minimum distance between interior points and boundary points.
        interior : array_like
            Indices for interior points.
        boundary : array_like
            Indices for boundary points.
        """

        self.vertices = vertices
        self.angular_resolution = angular_resolution
        self.spatial_resolution = spatial_resolution
        self.boundary_resolution = boundary_resolution
        self.dist_to_boundary = dist_to_boundary
        self.neighbours = neighbours

        if not interior is None:
            self.interior = interior
            if not boundary is None:
                self.boundary = boundary
            else:
                mask = np.in1d(self.indices,self.interior,invert=True)
                self.boundary = self.indices[mask]
        elif not boundary  is None:
            self.boundary = boundary
            mask = np.in1d(self.indices,self.boundary,invert=True)
            self.interior = self.indices[mask]
        else:
            raise TypeError("Please provide either boundary or interior indices")


    @property
    def dim(self):
        return self.vertices.shape[1]

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
    def bbox(self):
        return np.array([np.amin(self.vertices,0), np.amax(self.vertices,0)])

    @property
    def spatial_resolution(self):
        return self._h

    @spatial_resolution.setter
    def spatial_resolution(self,h):
        if h==None:
            self._h = h
        elif h>0:
            self._h = h
        else:
            raise TypeError("spatial resolution must be greater than 0")

    @property
    def boundary_resolution(self):
        return self._hb

    @boundary_resolution.setter
    def boundary_resolution(self,hb):
        if hb==None:
            self._hb = hb
        elif hb>0:
            self._hb = hb
        else:
            raise TypeError("boundary resolution must be greater than 0")

    @property
    def angular_resolution(self):
        return self._dtheta

    @angular_resolution.setter
    def angular_resolution(self,dtheta):
        if dtheta==None:
            self._dtheta = dtheta
        elif dtheta>0 and dtheta <= np.pi/2:
            self._dtheta = dtheta
        else:
            raise TypeError("angular_resolution must be strictly greater than 0 and less than pi/2")

    @property
    def dist_to_boundary(self):
        return self._delta

    @dist_to_boundary.setter
    def dist_to_boundary(self,delta):
        if delta==None:
            self._delta = delta
        elif delta>0:
            self._delta = delta
        else:
            raise TypeError("dist_to_boundary must be greater than 0")

    @property
    def _I(self):
        return self.neighbours[:,0]

    @property
    def _J(self):
        return self.neighbours[:,1]

    @property
    def _V(self):
        return self.vertices[self._J]-self.vertices[self._I]

    @property
    def _VDist(self):
        return np.linalg.norm(self._V, axis=1)

    @property
    def _Vs(self):
        return self._V/self._VDist[:,None]

class FDTriMesh(FDPointCloud):
    """Class for finite differences on triangular meshes."""

    def __init__(self, vertices, t, angular_resolution=np.pi/4,
            colinear_tol=1e-4, **kwargs):
        """Create a FDTriMesh object.

        Parameters
        ----------
        vertices : array
            An NxD array, listing N points in D dimensions.
        t : array
            An Nt x (D+1) array, listing Nt triangles (or tetrahedra in 3d).
        angular_resolution : float
            The desired angular resolution.
        colinear_tol : float
            Tolerance for detecting colinear points. Defaults to 1e-4.
            Set to False if you don't want this safety check.
        interior : array
            Indices for interior points.
        boundary : array
            Indices for boundary points.
        """

        super().__init__(vertices, angular_resolution=angular_resolution, **kwargs)

        # Compute circumcenter, get radius of circumcircle for each triangle.
        # This is the spatial resolution.
        if not self.spatial_resolution:
            def circumcircle_radius(i):
                X = vertices[t[i]]
                V = X[1:]-X[0]
                s = np.linalg.solve(V,.5*np.diag(V.dot(V.T)))
                return np.linalg.norm(s)
            self.spatial_resolution = max([circumcircle_radius(i) for i in range(t.shape[0])])


        # Compute minimum distance between interior and boundary vertices
        if not self.dist_to_boundary:
            b = np.reshape(np.in1d(t,self.boundary),t.shape).any(axis=1)
            tb = t[b]
            b = np.reshape(np.in1d(tb,self.boundary,invert=True),tb.shape)

            self.dist_to_boundary = min([cdist(vertices[ix,:][np.logical_not(i)],
                vertices[ix,:][i]).min() for (ix,i) in zip(tb,b) ])


        # Get edge from list of triangles
        edges = [np.array([v1,v2]).transpose() for v1,v2 in itertools.permutations(t.transpose(),2)]
        self.edges = np.concatenate(edges)
        self.neighbours = self.edges

        # Create adjacency matrix
        A = coo_matrix((np.ones(self._I.size, dtype=np.intp),(self._I,self._J)),
                shape=(self.num_nodes,self.num_nodes), dtype = np.intp)
        self.adjacency = A.tocsr()

        # Compute minimum edge length
        self.min_edge_length = self._VDist.min()

        # Find all neighbours 'depth' away (in graph distance) from each vertex
        if self.depth > 1:
            A_pows = [self.adjacency]
            for k in range(1,self.depth):
                A_pows.append( A_pows[k-1].dot(self.adjacency) )
            S = sum(A_pows)
            S = S.tocoo(copy=False)
            self.neighbours = np.array([S.row, S.col]).T

        # Remove neighbours that are outside the search radii of each vertex
        D = self._VDist
        mask = np.logical_and(D <= self.max_radius, D>=self.min_radius)
        self.neighbours = self.neighbours[mask,:]

        # If neighbours are colinear, remove them
        if colinear_tol:
            self._remove_colinear_neighbours(colinear_tol)

        # Get simplices
        self._compute_simplices()


    def __repr__(self):
        return ("FDTriMesh in {0.dim}D with {0.num_vertices} vertices, "
                "spatial resolution {0.spatial_resolution:.3g}, "
                "and angular resolution {0.angular_resolution:.3g}").format(self)

    def _compute_simplices(self):
        Vs = self._Vs

        def get_simplices(i):
            mask = self._I==i
            nb_ix = self._J[mask]
            vs = Vs[mask]

            hull = ConvexHull(vs)

            simplex = nb_ix[hull.simplices]
            i_array = np.full((simplex.shape[0],1),i)

            return np.concatenate( [i_array, simplex], -1)

        self.simplices = np.concatenate([get_simplices(i) for i in range(self.num_nodes)])

    def _remove_colinear_neighbours(self, colinear_tol):
        D = self._VDist
        Vs = self._V/D[:,None]

        # Strip redundant directions
        def strip_neighbours(i):
            mask = self._I==i
            nb_ix = self._J[mask]
            d = D[mask]
            vs = Vs[mask]

            cos = vs.dot(vs.T)
            check = cos > 1 - colinear_tol
            check = np.triu(check)

            ix = np.arange(nb_ix.size)

            keep = list({ix[r][np.argmin(d[r])] for r in check})

            i_array = np.full((len(keep),1),i)
            return np.concatenate([ i_array, nb_ix[keep,None] ], -1)

        self.neighbours = np.concatenate([strip_neighbours(i) for i in range(self.num_nodes)])


    @property
    def Cd(self):
        if self.dim==2:
            return 2
        elif self.dim==3:
            return 1 + 2/np.sqrt(3)
        else:
            raise TypeError("Dimensions other than two and three not supported")

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
    def max_radius(self):
        h = self.spatial_resolution
        th = self.angular_resolution

        return self.Cd * h * (1+ np.cos(th/2)/np.tan(th/2) + np.sin(th/2))

    @property
    def min_radius(self):
        h = self.spatial_resolution

        return self.max_radius - 2*self.Cd*h

    @property
    def depth(self):
        return int(np.ceil(self.max_radius/self.min_edge_length))


#class FDUniformGrid(FDPointCloud):
#        def __init__(self):
#
#        # TODO: don't limit with min search radius
#        if get_pairs:
#            self._compute_pairs(colinear_tol)
#    def _compute_pairs(self,colinear_tol):
#
#        I, J = self.neighbours[:,0], self.neighbours[:,1] # index of point and its neighbours
#
#        V = self.vertices[I] - self.vertices[J] # stencil vectors
#        Dist = np.linalg.norm(V,axis=1)         # length of stencil vector
#        Vs = V/Dist[:,None]                     # stencil vector, normalized
#
#        def get_pairs(i):
#            mask = I==i
#            nb_ix = J[mask]
#            vs = Vs[mask]
#
#            cos = vs.dot(vs.T)
#            check = cos < -1 + colinear_tol
#            check = np.triu(check)
#            if not check.any():
#                raise TypeError("Point {0} has no pairs".format(i))
#
#            ix = np.indices(check.shape)
#            pairs = ix[:, check].T
#
#            i_array = np.full((len(pairs),1),i)
#            return  np.concatenate([ i_array, nb_ix[pairs] ], -1)
#
#        self.pairs = np.concatenate([get_pairs(i) for i in self.interior])
