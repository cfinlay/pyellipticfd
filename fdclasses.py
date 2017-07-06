"""Classes to structure data for finite differences."""

import warnings
import itertools
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, pdist, squareform

from pyellipticfd import stencils

class FDPointCloud(object):
    """Base class for finite differences on point clouds"""

    def __repr__(self):
        return ("FDPointCloud in {0.dim}D with {0.num_vertices} vertices").format(self)


    def __init__(self, vertices, angular_resolution=None,
            spatial_resolution=None, neighbours=None,
            boundary_resolution=None, dist_to_boundary=None,
            interior=None, boundary=None, boundary_normals=None):
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
        boundary_normals : array_like
            Array of outward pointing normals on the boundary.
        """

        if angular_resolution is None:
            self._dtheta = angular_resolution
        elif angular_resolution>0 and angular_resolution <= np.pi:
            self._dtheta = angular_resolution
        else:
            raise TypeError("angular_resolution must be strictly greater than 0 and less than pi")

        self._pts = vertices
        self._h = spatial_resolution
        self._hb = boundary_resolution
        self._delta = dist_to_boundary
        self._nbs = neighbours
        self._simplices = None
        self._normals = boundary_normals

        if not interior is None:
            self.interior = interior
            if not boundary is None:
                self.boundary = boundary
            else:
                mask = np.in1d(self.indices,self.interior,invert=True)
                self.boundary = self.indices[mask]
        elif not boundary is None:
            self.boundary = boundary
            mask = np.in1d(self.indices,self.boundary,invert=True)
            self.interior = self.indices[mask]
        else:
            raise TypeError("Please provide either boundary or interior indices")
    @property
    def vertices(self):
        return self._pts

    @property
    def neighbours(self):
        return self._nbs

    @property
    def simplices(self):
        return self._simplices

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
    def nodes(self):
        return self.vertices

    @property
    def num_nodes(self):
        return self.num_vertices

    @property
    def interior_points(self):
        return self._pts[self.interior]

    @property
    def num_interior(self):
        return self.interior.size

    @property
    def boundary_points(self):
        return self._pts[self.boundary]

    @property
    def num_boundary(self):
        return self.boundary.size

    @property
    def boundary_normals(self):
        return self._normals

    @property
    def bbox(self):
        return np.array([np.amin(self._pts,0), np.amax(self._pts,0)])

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
        return self._pts[self._J]-self._pts[self._I]

    @property
    def _VDist(self):
        """Length of stencil vectors"""
        return np.linalg.norm(self._V, axis=1)

    @property
    def _Vs(self):
        """Stencil vectors, normalized"""
        return self._V/self._VDist[:,None]

    @property
    def adjacency(self):
        A = coo_matrix((np.ones(self._I.size, dtype=np.intp),(self._I,self._J)),
                shape=(self.num_points,self.num_points), dtype = np.intp)
        return A.tocsr()

    def _remove_colinear_neighbours(self, colinear_tol):
        V = self._V
        D = np.linalg.norm(V,axis=1)

        # Strip redundant directions
        def strip_neighbours(i):
            mask = self._I==i
            nb_ix = self._J[mask]
            nv = nb_ix.size
            v = V[mask]
            d = D[mask]

            Dc = pdist(v,'cosine')
            check = Dc < colinear_tol
            if np.sum(check) > 0:
                # pairwise combinations of stencil points
                ij = np.array(list(itertools.combinations(range(nv),2)))

                # those points that fail the colinearity test
                doubles = ij[check]

                # of each pair, which if furtherst from the stencil
                arg = np.argmax(d[doubles],axis=1)

                # points to discard
                nd = doubles.shape[0]
                ix = np.arange(nd)
                discard = doubles[ix,arg]

                # mask for those that should remain
                mask =  np.in1d(np.arange(nv),discard,invert=True)

                keep = np.arange(nv)[mask] # points that pass colinear test

                # Need to deal with boundary points.
                # Want to include boundary points with
                # high angular resolution, but not those that
                # fail the colinear test with interior points.
                # --------------------------------------------
                mask_boundary = np.reshape(np.in1d(nb_ix[doubles],self.boundary),
                                           doubles.shape)
                bdry_int_pair = np.logical_xor.reduce(mask_boundary,axis=1)
                I, J = ix[bdry_int_pair], arg[bdry_int_pair]
                bdry_discard = doubles[I,J]

                boundary_mask = np.in1d(nb_ix,self.boundary)
                boundary = np.arange(nv)[boundary_mask]
                mask_boundary_keep = np.in1d(boundary, bdry_discard, invert=True)
                boundary_keep = boundary[mask_boundary_keep]

                #b_ix = np.in1d(nb_ix,self.boundary)
                #boundary_keep = np.arange(nv)[b_ix]

                keep = np.unique(np.concatenate([boundary_keep,keep]))

            else:
                keep = np.arange(nv)

            i_array = np.full((len(keep),1),i)
            return  np.concatenate([ i_array, nb_ix[keep,None] ], -1)

        self._nbs = np.concatenate([strip_neighbours(i) for i in range(self.num_points)])

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

        self._simplices = np.concatenate([get_simplices(i) for i in range(self.num_points)])

class FDTriMesh(FDPointCloud):
    """Class for finite differences on triangular meshes."""

    def __init__(self, p, t, angular_resolution=np.pi/4,
              interpolation=True, **kwargs):
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
        interpolation : bool
            If True, create simplices for interpolating finite differences.
            If False, use Froese's finite difference method for point clouds.
        interior : array
            Indices for interior points.
        boundary : array
            Indices for boundary points.
        boundary_normals : array_like
            Array of outward pointing normals on the boundary.
        """

        super().__init__(p, angular_resolution=angular_resolution, **kwargs)

        if interpolation: # TODO: Check this is correct in 3D
            colinear_tol = 1-np.cos(angular_resolution/4)
        else:
            colinear_tol = 1-np.cos(angular_resolution/2)

        self._interp = interpolation
        self._T = t

        if not interpolation:
            min_search=False
        else:
            min_search=True
        self._min_search = min_search

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

        if self.dist_to_boundary < self.min_boundary_radius:
            raise TypeError (
                ("The interior grid points' minimum distance to the boundary"
                " ({0.dist_to_boundary:.3g}) "
                "\nmust be greater than the minimum boundary search radius"
                "({0.min_boundary_radius:.3g})."
                "\nTry increasing the angular resolution,"
                "\nor deleting points close to the boundary.").format(self))

        if self._hb > self._max_hb:
            raise TypeError (("The boundary resolution ({0._hb:.3g}) is not small enough "
                "\nto satisfy the desired angular resolution."
                "\nNeed boundary resolution less than ({0._max_hb:.3g})").format(self))


        # Get edge from list of triangles
        edges = [np.array([v1,v2]).transpose() for v1,v2 in itertools.permutations(t.transpose(),2)]
        self._e = np.concatenate(edges)
        self._nbs = self.edges

        # Compute minimum interior edge length
        self._l = (self._VDist[np.in1d(self._I, self.interior)]).min()

        # Find all neighbours 'depth' away (in graph distance) from each vertex
        A = self.adjacency
        if self.depth > 1:
            A_pows = [A]
            for k in range(1,self.depth):
                A_pows.append( A_pows[k-1].dot(A) )
            S = sum(A_pows)
            S = S.tocoo(copy=False)
            self._nbs = np.array([S.row, S.col]).T

        # Remove neighbours that are outside the search radii of each vertex
        D = self._VDist

        mask = np.logical_and.reduce((D <= self.max_radius,
                                      np.logical_or(D>=self.min_radius,
                                                    np.in1d(self._J, self.boundary)),
                                      self._I!=self._J))

        self._nbs = self.neighbours[mask,:]

        # If neighbours are colinear, remove them
        self._remove_colinear_neighbours(colinear_tol)

        # Get simplices
        if interpolation:
            self._compute_simplices()

    def __repr__(self):
        return ("FDTriMesh in {0.dim}D with {0.num_vertices} vertices, "
                "spatial resolution {0.spatial_resolution:.3g}, "
                "and angular resolution {0.angular_resolution:.3g}").format(self)

    @property
    def triangulation(self):
        return self._T

    @property
    def edges(self):
        return self._e

    @property
    def min_edge_length(self):
        """Minimum interior edge length."""
        return self._l

    @property
    def Cd(self):
        if self._interp:
            if self.dim==2:
                return 2
            elif self.dim==3:
                return 1 + 2/np.sqrt(3)
            else:
                raise TypeError("Dimensions other than two and three not supported")
        else:
            if self.dim==2:
                return 1
            else:
                raise TypeError("Dimensions other than two not supported")

    @property
    def max_radius(self):
        h = self.spatial_resolution
        th = self.angular_resolution

        return  h * (1+ self.Cd / np.sin(th/2))

    @property
    def min_radius(self):
        if self._min_search:
            h = self.spatial_resolution
            th = self.angular_resolution
            return h * (-1+ self.Cd/ np.sin(th/2))
        else:
            return 0.0

    @property
    def min_boundary_radius(self):
        if self._min_search:
            h = self.boundary_resolution
            th = self.angular_resolution
            return  h * (-1+ self.Cd /  np.sin(th/2))
        else:
            return 0.0

    @property
    def depth(self):
        return int(np.ceil(self.max_radius/self.min_edge_length))

    @property
    def _max_hb(self):
        if self._interp:
            return self._delta*np.tan(self._dtheta/2)/self.Cd
        else:
            return 2*self._delta*np.tan(self._dtheta/2)


class FDRegularGrid(FDPointCloud):
    """Class for finite differences on rectangular grids."""
    # TODO : implement Bresnham's algorithm, or something similar

    def __repr__(self):
        return ("FDRegularGrid in {0.dim}D with {0.num_vertices} vertices, "
                "spatial resolution {0.spatial_resolution:.3g}, "
                "and angular resolution {0.angular_resolution:.3g}").format(self)

    def __init__(self, interior_shape, bounds, stencil_radius, get_pairs=True,
            interpolation=True, colinear_tol=1e-3):
        """Create a FDRegularGrid.

        Parameters
        ----------
        interior_shape : array
            Number of interior points per axis.
        bounds : array
            A 2 x D array of the bounding box. The first row is the minimum
            point, the second the maximal point.
        stencil_radius : int
            Maximum integer distance to search for stencil points on a lattice.
        get_pairs : bool
            Whether to search for directions with exact derivatives on
            quadratic functions.
        interpolation : bool
            If True, create simplices for interpolating finite differences.
        colinear_tol : float
            Tolerance for detecting colinear points. Defaults to 1e-3.
            Set to False if you don't want this safety check.
        """
        #TODO need minimum search radius to guarantee converence for interpolation method...

        self._r = stencil_radius
        self._interp = interpolation

        dim = len(interior_shape)
        interior_shape = np.array(interior_shape)
        bounds = np.array(bounds)

        rect = bounds/(interior_shape+1)
        self._h = np.linalg.norm(rect)/2

        cardinal_points = [np.arange(1,n+1) for n in interior_shape]
        Xint = np.meshgrid(*cardinal_points,sparse=False,indexing='ij')
        Xint = np.reshape(Xint,(dim,np.prod(interior_shape))).T


        # define appropriate stencil for calculating neighbours
        stcl = stencils.create(stencil_radius,dim)
        stcl_l1 = stcl/np.max(np.abs(stcl),axis=1)[:,None] # normalize by maximum side length
        stcl_l1 = np.concatenate([[[0,0]],stcl_l1])


        # Compute neighbours, for boundary
        mask = np.stack([Xint[:,i] == n for i, n in enumerate(interior_shape)])
        mask = np.logical_or( (Xint==1).any(axis=1),
                            mask.any(axis=0))

        X = Xint[mask,:,None] + stcl_l1.T[None,:,:]
        sh = X.shape
        X = np.reshape(np.rollaxis(X,2), (sh[0]*sh[2],sh[1]))

        mask = [np.logical_or(X[:,i]==0, X[:,i]==n+1) for i, n in enumerate(interior_shape)]
        mask = np.stack(mask)
        mask = mask.any(axis=0)
        X = X[mask]

        Xb = np.vstack({tuple(row) for row in X})

        self._pts = np.concatenate([Xint,Xb])
        self.interior = np.arange(Xint.shape[0])
        self.boundary = np.arange(Xint.shape[0],Xint.shape[0]+Xb.shape[0])

        #TODO: normals

        # Function to compute maximum side length
        d = lambda u, v : np.max(np.abs(u-v))
        Dc = pdist(self._pts, d)
        D = squareform(Dc, checks=False)

        if (stencil_radius==1 and self._interp) or (not self._interp):
            Nb =np.logical_and(D <= stencil_radius, D>0)
        elif stencil_radius > 1 and self._interp:
            Nb =np.logical_and(D <= stencil_radius, D>= (stencil_radius-1) )


        self._nbs = np.concatenate([np.stack([np.full(Nb[i].sum(),i), self.indices[Nb[i]]]).T
            for i in self.indices])

        # If neighbours are colinear, remove them
        if colinear_tol:
            self._remove_colinear_neighbours(colinear_tol)

        self._pairs = None
        if get_pairs:
            self._compute_pairs(colinear_tol)

        # Get simplices
        self._simplices = None
        if self._interp:
            self._compute_simplices()

        # Scale points
        scaling = (bounds[1]-bounds[0])/(interior_shape+1)
        self._pts = self._pts*scaling + bounds[0]

        # angular resolution
        stcl = stcl*scaling
        stcl = stcl[np.logical_not(stcl==0).all(axis=1)]
        def dtheta(i):
            e = np.zeros(dim)
            e[i] = 1
            vs = stcl/np.linalg.norm(stcl,axis=1)[:,None]
            c = vs.dot(e)
            th = np.arccos(c)
            return th[th>0].min()

        self._dtheta = max([dtheta(i) for i in range(dim)])

        # distance to boundary
        self._delta = scaling.min()

        # boundary resolution
        b_rect = np.full(dim,1/stencil_radius)
        b_rect = b_rect*scaling
        self._hb = max([np.linalg.norm(v)/2 for v in itertools.combinations(b_rect,dim-1)])

    def _compute_pairs(self,colinear_tol):

        I, J = self._I, self._J # index of point and its neighbours
        Vs = self._Vs           # stencil vector, normalized

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

        self._pairs = np.concatenate([get_pairs(i) for i in self.interior])

    @property
    def pairs(self):
        return self._pairs

    @property
    def stencil_radius(self):
        """Stencil radius on integer lattice"""
        return self._r

    @property
    def depth(self):
        """Stencil radius on integer lattice"""
        return self._r
