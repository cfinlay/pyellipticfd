"""Classes to structure data for finite differences."""

import warnings
import itertools
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, pdist, squareform

from pyellipticfd import stencils

def compute_pairs(PointCloud,colinear_tol):
    """Compute antipodal pairs in each points' stencil on a regular grid."""

    I, J = PointCloud._I, PointCloud._J # index of point and its neighbours
    Vs = PointCloud._Vs                 # stencil vector, normalized

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

    PointCloud.pairs = np.concatenate([get_pairs(i) for i in PointCloud.interior])

def remove_colinear_neighbours(PointCloud, colinear_tol,prefer="min"):
    """Remove colinear neighbours for each stencil."""
    V = PointCloud._V
    D = np.linalg.norm(V,axis=1)

    # Strip redundant directions
    def strip_neighbours(i):
        mask = PointCloud._I==i
        nb_ix = PointCloud._J[mask]
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

            # of each pair, which is furthest from the stencil
            if prefer=="min":
                arg = np.argmax(d[doubles],axis=1)
            elif prefer=="max":
                arg = np.argmin(d[doubles],axis=1)

            # points to discard
            nd = doubles.shape[0]
            ix = np.arange(nd)
            discard = doubles[ix,arg]

            # mask for those that should remain. Keep closer neighbours
            mask =  np.in1d(np.arange(nv),discard,invert=True)

            keep = np.arange(nv)[mask] # points that pass colinear test

            # Need to deal with boundary points.
            # Want to include boundary points with
            # high angular resolution, but not those that
            # fail the colinear test with interior points.
            # --------------------------------------------
            mask_boundary = np.reshape(np.in1d(nb_ix[doubles],PointCloud.bdry),
                                       doubles.shape)
            bdry_int_pair = np.logical_xor.reduce(mask_boundary,axis=1)
            I, J = ix[bdry_int_pair], arg[bdry_int_pair]
            bdry_discard = doubles[I,J]

            boundary_mask = np.in1d(nb_ix,PointCloud.bdry)
            boundary = np.arange(nv)[boundary_mask]
            mask_boundary_keep = np.in1d(boundary, bdry_discard, invert=True)
            boundary_keep = boundary[mask_boundary_keep]

            #b_ix = np.in1d(nb_ix,PointCloud.bdry)
            #boundary_keep = np.arange(nv)[b_ix]

            keep = np.unique(np.concatenate([boundary_keep,keep]))

        else:
            keep = np.arange(nv)

        i_array = np.full((len(keep),1),i)
        return  np.concatenate([ i_array, nb_ix[keep,None] ], -1)

    PointCloud.neighbours = np.concatenate(
                                [strip_neighbours(i) for i in range(PointCloud.num_points)])

def compute_simplices(PointCloud):
    """Compute simplices for interpolating finite differences."""
    Vs = PointCloud._Vs

    def get_simplices(i):
        mask = PointCloud._I==i
        nb_ix = PointCloud._J[mask]
        vs = Vs[mask]

        hull = ConvexHull(vs)

        simplex = nb_ix[hull.simplices]
        i_array = np.full((simplex.shape[0],1),i)

        return np.concatenate( [i_array, simplex], -1)

    PointCloud.simplices = np.concatenate([get_simplices(i) for i in range(PointCloud.num_points)])

class FDPointCloud(object):
    """Base class for finite differences on point clouds"""

    def __repr__(self):
        return ("FDPointCloud in {0.dim}D with {0.num_points} points").format(self)


    #TODO: generate neighbours, simplices without triangulation
    def __init__(self, points, angular_resolution=None,
            spatial_resolution=None, neighbours=None,
            bdry_resolution=None, dist_to_bdry=None,
            interior=None, boundary=None, bdry_normals=None):
        """Initialize a finite difference point cloud.

        Parameters
        ----------
        points : array_like
            An NxD array, listing N points in D dimensions.
        angular_resolution : float
            The desired angular resolution.
        spatial_resolution : float
            The spatial resolution of the graph. Every ball with radius
            'spatial_resolution' contains at least one point.
        bdry_resolution : float
            The spatial resolution of the graph on the boundary. Every ball
            centered on the boundary, with radius 'bdry_resolution',
            contains at least onei boundary point.
        dist_to_bdry : float
            The minimum distance between interior points and boundary points.
        interior : array_like
            Indices for interior points.
        boundary : array_like
            Indices for boundary points.
        bdry_normals : array_like
            Array of outward pointing normals on the boundary.
        """

        if angular_resolution is None:
            self.angular_resolution = angular_resolution
        elif angular_resolution>0 and angular_resolution <= np.pi:
            self.angular_resolution = angular_resolution
        else:
            raise TypeError("angular_resolution must be strictly greater than 0 and less than pi")

        self.spatial_resolution = spatial_resolution
        self.bdry_resolution = bdry_resolution
        self.dist_to_bdry = dist_to_bdry
        self.neighbours = None
        self.simplices = None
        self.pairs = None
        self.bdry_normals = bdry_normals
        self._d1cache = None
        self._d2cache = None

        if not interior is None:
            if boundary is None:
                mask = np.in1d(self.indices,interior,invert=True)
                boundary = self.indices[mask]
        elif not boundary is None:
            mask = np.in1d(self.indices,boundary,invert=True)
            interior = self.indices[mask]
        else:
            raise TypeError("Please provide either boundary or interior indices")

        self.num_interior = interior.size
        self.num_bdry = boundary.size
        self.points = np.concatenate([points[interior],points[boundary]], axis=0)

    @property
    def num_points(self):
        return self.points.shape[0]

    @property
    def dim(self):
        return self.points.shape[1]

    @property
    def indices(self):
        return np.arange(self.num_points)

    @property
    def interior_points(self):
        return self.points[self.interior]

    @property
    def interior(self):
        return np.arange(self.num_interior)

    @property
    def bdry_points(self):
        return self.points[self.bdry]

    @property
    def bdry(self):
        return np.arange(self.num_interior,self.num_interior+self.num_bdry)

    @property
    def bbox(self):
        return np.array([np.amin(self.points,0), np.amax(self.points,0)])

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
        return self.points[self._J]-self.points[self._I]

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
        """Adjacency matrix of each point and its neighbours."""
        A = coo_matrix((np.ones(self._I.size, dtype=np.intp),(self._I,self._J)),
                shape=(self.num_points,self.num_points), dtype = np.intp)
        return A.tocsr()


    @property
    def Cd(self):
        """Spatial constant for guarnteeing existence of simplices."""
        if self.dim==2:
            return 2
        elif self.dim==3:
            return 1 + 2/np.sqrt(3)
        else:
            raise TypeError("Dimensions other than two and three not supported")

    @property
    def max_radius(self):
        """Maximum search radius for neighbours."""
        h = self.spatial_resolution
        th = self.angular_resolution
        return  h * (1+ self.Cd / np.sin(th/2))

    @property
    def min_radius(self):
        """Minimum search radius for neighbours."""
        h = self.spatial_resolution
        th = self.angular_resolution
        return h * (-1+ self.Cd/ np.sin(th/2))

    @property
    def min_bdry_radius(self):
        """Minimum allowable search radius for boundary neighbours."""
        h = self.bdry_resolution
        th = self.angular_resolution
        return  h * (-1+ self.Cd /  np.sin(th/2))

    @property
    def maximum_bdry_res(self):
        """Maximum allowable boundary resolution, given angular resolution."""
        return self.dist_to_bdry*np.tan(self.angular_resolution/2)/self.Cd

    @property
    def min_interior_nb_dist(self):
        """Minimum interior distance between neighbours."""
        return (self._VDist[np.in1d(self._I, self.interior)]).min()



class FDTriMesh(FDPointCloud):
    """Class for finite differences on triangular meshes."""

    def __repr__(self):
        return ("FDTriMesh in {0.dim}D with {0.num_points} points, "
                "spatial resolution {0.spatial_resolution:.3g}, "
                "and angular resolution {0.angular_resolution:.3g}").format(self)

    def __init__(self, p, t, angular_resolution=np.pi/4, **kwargs):
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
        interior : array
            Indices for interior points.
        boundary : array
            Indices for boundary points.
        bdry_normals : array_like
            Array of outward pointing normals on the boundary.
        """

        super().__init__(p, angular_resolution=angular_resolution, **kwargs)

        # TODO: sort triangulation to reflect interior/boundary order

        colinear_tol = 1-np.cos(angular_resolution/4)

        self.triangulation = t

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
            self.spatial_resolution = max([circumcircle_radius(ix) for ix in t])


        if (not self.dist_to_bdry) or (not self.bdry_resolution):
            b = np.reshape(np.in1d(t,self.bdry),t.shape)

        # Compute minimum distance between interior and boundary points
        if not self.dist_to_bdry:
            b1 = b.any(axis=1)
            tb = t[b1]
            b1 = np.reshape(np.in1d(tb,self.bdry,invert=True),tb.shape)

            self.dist_to_bdry = min([cdist(p[ix,:][np.logical_not(i)],
                p[ix,:][i]).min() for (ix,i) in zip(tb,b1) ])


        # Compute the boundary resolution
        if not self.bdry_resolution:
            tb = t[b.sum(axis=1)==2]

            # boundary faces
            fb = np.reshape(tb.flatten()[np.in1d(tb,self.bdry)],(tb.shape[0],self.dim))

            self.bdry_resolution = max([circumcircle_radius(ix) for ix in fb])

        if self.dist_to_bdry < self.min_bdry_radius:
            raise TypeError (
                ("The interior grid points' minimum distance to the boundary"
                " ({0.dist_to_bdry:.3g}) "
                "\nmust be greater than the minimum boundary search radius"
                "({0.min_bdry_radius:.3g})."
                "\nTry increasing the angular resolution,"
                "\nor deleting points close to the boundary.").format(self))

        if self.bdry_resolution > self.maximum_bdry_res:
            raise TypeError (("The boundary resolution ({0.bdry_resolution:.3g}) is not small enough "
                "\nto satisfy the desired angular resolution."
                "\nNeed boundary resolution less than ({0.maximum_bdry_res:.3g})").format(self))


        # Get edge from list of triangles
        edges = [np.array([v1,v2]).transpose() for v1,v2 in itertools.permutations(t.transpose(),2)]
        self.edges = np.concatenate(edges)
        self.neighbours = self.edges

        # Find all neighbours 'depth' away (in graph distance) from each vertex
        depth = int(np.ceil(self.max_radius/self.min_interior_nb_dist))
        A = self.adjacency
        if depth > 1:
            A_pows = [A]
            for k in range(1,depth):
                A_pows.append( A_pows[k-1].dot(A) )
            S = sum(A_pows)
            S = S.tocoo(copy=False)
            self.neighbours = np.array([S.row, S.col]).T

        # Remove neighbours that are outside the search radii of each vertex
        D = self._VDist

        mask = np.logical_and.reduce((D <= self.max_radius,
                                      np.logical_or(D>=self.min_radius,
                                                    np.in1d(self._J, self.bdry)),
                                      self._I!=self._J))

        self.neighbours = self.neighbours[mask,:]

        # If neighbours are colinear, remove them
        remove_colinear_neighbours(self,colinear_tol)

        # Create the simplices
        compute_simplices(self)




class FDRegularGrid(FDPointCloud):
    """Class for finite differences on rectangular grids."""

    def __repr__(self):
        return ("FDRegularGrid in {0.dim}D with {0.num_points} points, "
                "spatial resolution {0.spatial_resolution:.3g}, "
                "and angular resolution {0.angular_resolution:.3g}").format(self)

    def __init__(self, interior_shape, bounds, stencil_radius,
            interpolation=True):
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
        interpolation : bool
            If True, create simplices for interpolating finite differences.
            If False, create pairs of antipodal points, for finite differences only
            in grid directions.
        """
        #TODO need minimum search radius to guarantee converence for interpolation method...

        self.stencil_radius = stencil_radius

        dim = len(interior_shape)
        interior_shape = np.array(interior_shape)
        bounds = np.array(bounds)

        rect = bounds/(interior_shape+1)
        self.spatial_resolution = np.linalg.norm(rect)/2

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

        self.points = np.concatenate([Xint,Xb])
        self.num_interior = Xint.shape[0]
        self.num_bdry = Xb.shape[0]

        #TODO: normals

        # Function to compute maximum side length
        d = lambda u, v : np.max(np.abs(u-v))
        Dc = pdist(self.points, d)
        D = squareform(Dc, checks=False)

        Nb =np.logical_and(D <= stencil_radius, D>0)

        self.neighbours = np.concatenate([np.stack([np.full(Nb[i].sum(),i), self.indices[Nb[i]]]).T
            for i in self.indices])

        # Scale points
        scaling = (bounds[1]-bounds[0])/(interior_shape+1)
        self.points = self.points*scaling + bounds[0]

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

        self.angular_resolution = max([dtheta(i) for i in range(dim)])

        # distance to boundary
        self.dist_to_bdry = scaling.min()

        # boundary resolution
        b_rect = np.full(dim,1/stencil_radius)
        b_rect = b_rect*scaling
        self.bdry_resolution = max([np.linalg.norm(v)/2 for v in itertools.combinations(b_rect,dim-1)])

        if not interpolation:
            colinear_tol = 1e-3
            remove_colinear_neighbours(self,colinear_tol)
            compute_pairs(self,colinear_tol)
            self.simplices = None
        else:
            colinear_tol = 1-np.cos(self.angular_resolution/4)
            remove_colinear_neighbours(self,colinear_tol,prefer="max")
            compute_simplices(self)
            self.pairs = None

        self._d1cache = None
        self._d2cache = None

    # TODO redifine min_search, max_search etc
