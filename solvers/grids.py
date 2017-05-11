"""Classes to structure data for finite differences."""

import warnings
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

        if angular_resolution is None:
            self._dtheta = angular_resolution
        elif angular_resolution>0 and angular_resolution <= np.pi/2:
            self._dtheta = angular_resolution
        else:
            raise TypeError("angular_resolution must be strictly greater than 0 and less than pi/2")

        self.vertices = vertices
        self._h = spatial_resolution
        self._hb = boundary_resolution
        self._delta = dist_to_boundary
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
    def points(self):
        return self.vertices

    @property
    def num_points(self):
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

    @property
    def boundary_resolution(self):
        return self._hb

    @property
    def angular_resolution(self):
        return self._dtheta


    @property
    def dist_to_boundary(self):
        return self._delta


    @property
    def _I(self):
        """Index of centre point in stencil vectors"""
        return self.neighbours[:,0]

    @property
    def _J(self):
        """Index of neighbour point in stencil vectors"""
        return self.neighbours[:,1]

    @property
    def _V(self):
        """Stencil vectors"""
        return self.vertices[self._J]-self.vertices[self._I]

    @property
    def _VDist(self):
        """Length of stencil vectors"""
        return np.linalg.norm(self._V, axis=1)

    @property
    def _Vs(self):
        """Stencil vectors, normalized"""
        return self._V/self._VDist[:,None]

class FDTriMesh(FDPointCloud):
    """Class for finite differences on triangular meshes."""
    # Only for interpolating methods, not Froese's
    #TODO: Froese's framework

    def __init__(self, p, t, angular_resolution=np.pi/4,
            colinear_tol=1e-4, **kwargs):
        """Create a FDTriMesh object.

        Parameters
        ----------
        p : array
            An NxD array, listing N points in D dimensions.
        t : array
            An Nt x (D+1) array of point indices, listing Nt triangles (or
            tetrahedra in 3d).
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

        super().__init__(p, angular_resolution=angular_resolution, **kwargs)

        # function to compute a simplex's circumcircle's radius
        def circumcircle_radius(ix):
            X = p[ix]
            V = X[1:]-X[0]
            if ix.size > 2:
                s = np.linalg.solve(V,.5*np.diag(V.dot(V.T)))
                return np.linalg.norm(s)
            elif ix.size==2:
                return np.linalg.norm(V)/2

        # Compute circumcenter, get radius of circumcircle for each triangle.
        # This is the spatial resolution.
        if not self.spatial_resolution:
            self._h = max([circumcircle_radius(ix) for ix in t])


        if (not self.dist_to_boundary) or (not self.boundary_resolution):
            b = np.reshape(np.in1d(t,self.boundary),t.shape)

        # Compute minimum distance between interior and boundary vertices
        if not self.dist_to_boundary:
            b1 = b.any(axis=1)
            tb = t[b1]
            b1 = np.reshape(np.in1d(tb,self.boundary,invert=True),tb.shape)

            self._delta = min([cdist(p[ix,:][np.logical_not(i)],
                p[ix,:][i]).min() for (ix,i) in zip(tb,b1) ])

        # Compute the boundary resolution
        if not self.boundary_resolution:
            tb = t[b.sum(axis=1)==2]

            # boundary faces
            fb = np.reshape(tb.flatten()[np.in1d(tb,self.boundary)],(tb.shape[0],self.dim))

            self._hb = max([circumcircle_radius(ix) for ix in fb])

        needed_hb = self._delta*np.tan(self._dtheta/2)/self.Cd
        if self._hb > needed_hb:
            raise TypeError ("The boundary resolution {0._hb:.3g} is not small enough "
                "to satisfy the desired angular resolution."
                "\nNeed boundary resolution less than {1:.3g}").format(self,needed_hb)


        # Get edge from list of triangles
        edges = [np.array([v1,v2]).transpose() for v1,v2 in itertools.permutations(t.transpose(),2)]
        self.edges = np.concatenate(edges)
        self.neighbours = self.edges

        # Create adjacency matrix
        A = coo_matrix((np.ones(self._I.size, dtype=np.intp),(self._I,self._J)),
                shape=(self.num_points,self.num_points), dtype = np.intp)
        self.adjacency = A.tocsr()

        # Compute minimum edge length
        self._l = self._VDist.min()

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

        self.simplices = np.concatenate([get_simplices(i) for i in range(self.num_points)])

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

        self.neighbours = np.concatenate([strip_neighbours(i) for i in range(self.num_points)])


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
