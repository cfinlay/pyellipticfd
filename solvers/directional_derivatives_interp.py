import numpy as np
import itertools
import stencils

#def d1da(U,dx,direction="both"):
#    """
#    Calculate descent and ascent directions respectively minimizing and
#    maximizing
#        <grad u, p>, st ||p|| = 1.
#    """
#
#def d1descent(U,dx):
#    """
#    Compute the minimal value of <grad u, p> st ||p||=1.
#    Equivalent to calling d2da(u,dx,direction="descent")
#    """
#
#def d1ascent(U,dx):
#    """
#    Compute the maximal value of <grad u, p> st ||p||=1.
#    Equivalent to calling d2da(u,dx,direction="ascent")
#    """
#
#def d1(U,theta,dx):
#    """
#    Compute the directional derivative of u, in direction
#    v = [cos(theta), sin(theta)].
#    """

def d2(U,X,Ix,v):
    """
    Compute the second directional derivative of u, in direction v.

    Parameters
    ----------
    U : array_like
        Function values at grid points.
    dx : scalar
        Uniform grid resolution.
    stencil: array_like
        An array of stencil vectors. Must be listed in counterclockwise order.
    control : tuple
        Angle of direction of derivative. The first element of the tuple is the index
        specifying a pair of stencil vectors, given by the index to the
        clockwise element of the pair. The second element is the convex
        combination parameter.

    Returns
    -------
    d2u : array_like
        Second directional derivative.
    """
    domain_shape = np.array(U.shape)
    ndims = domain_shape.size
    D2vv = np.zeros(domain_shape-2)


    # We want v to be an array of vectors, a direction for each point
    v = np.array(v)
    if v.size==1:
        # v is a constant angle, convert to vector for each point
        v = np.array([np.broadcast_to(np.cos(v),domain_shape-2),
                      np.broadcast_to(np.sin(v),domain_shape-2)])

    elif v.size==2 & ndims==3:
        v = np.array([np.broadcast_to(np.sin(v[0])*np.cos(v[1]),domain_shape-2),
                      np.broadcast_to(np.sin(v[0])*np.sin(v[1]),domain_shape-2),
                      np.broadcast_to(np.cos(v[1]),domain_shape-2)])

    elif v.size==ndims:
        # v is already a vector, but constant for each point.
        # Broadcast to everypoint
        v /= np.linalg.norm(v)
        if ndims==2:
            v = np.broadcast_to(v[:,None,None],
                                np.append(ndims,domain_shape-2))
        elif ndims==3:
            v = np.broadcast_to(v[:,None,None,None],
                                np.append(ndims,domain_shape-2))

    elif (v.shape==domain_shape-2).all() & ndims==2:
        # v is an angle, convert to vector
        v = np.array([np.cos(v),np.sin(v)])

    elif v.shape==np.append(ndims-1,domain_shape-2).all() & ndims==3:
        #
        v = np.array([np.sin(v[0,:,:])*np.cos(v[1,:,:]),
                      np.sin(v[0,:,:])*np.sin(v[1,:,:]),
                      np.cos(v[1,:,:])])

    elif (v.shape==np.append(ndims,domain_shape-2)).all():
        #then v is a vector for each point, normalize
        norm_v = np.linalg.norm(v,axis=0)
        v /=  norm_v

## TODO: -deal with points near boundary
#def d2eigs(U,dx,stencil=stencil,eigs="both"):
#    """
#    Compute the maximum and minimum eigenvalues of the Hessian of U.
#
#    Parameters
#    ----------
#    u : array_like
#        Function values at grid points.
#    dx : scalar
#        Uniform grid resolution.
#    eigs : string
#        Specify which eigenvalue to retrieve: "min", "max", or "both".
#
#    Returns
#    -------
#    Lambda : a tuple, or an array
#        If eigs="both", a tuple containing the minimal and maximal eigenvalues,
#        with the minimal eigenvalue first.
#        If eigs!="both", then an array of the specified eigenvalue.
#    Control : a list of controls
#        If eigs="both", a tuple containing the controls of the minimal and maximal eigenvalues,
#        minimal eigenvalue first.
#        If eigs!="both", then the control of the specified eigenvalue.
#    """
#    Nx, Ny = U.shape
#    width = np.abs(stencil).max()
#    nvectors = int(stencil.shape[0]/2) #TODO: verify this actually an integer
#
#    if eigs=="both" or eigs=="min":
#        Dvv_min = np.zeros((nvectors,Nx-2*width,Ny-2*width))
#        t_min = np.zeros((nvectors,Nx-2*width,Ny-2*width))
#    if eigs=="both" or eigs=="max":
#        Dvv_max = np.zeros((nvectors,Nx-2*width,Ny-2*width))
#        t_max = np.zeros((nvectors,Nx-2*width,Ny-2*width))
#
#    #Vector indices excluding the boundary
#    I = np.arange(width,Nx-width,dtype=np.intp)
#    J = np.arange(width,Ny-width,dtype=np.intp)
#
#    A = U[width:-width,width:-width]
#
#    #Block indices, interior only
#    [Iint, Jint] = np.indices(A.shape)
#
#    for k, (v,w) in enumerate(zip(stencil[0:nvectors],stencil[1:(nvectors+1)])):
#        vdotw = np.dot(v,w)
#        diff = w-v
#        norm_diff2 = np.dot(diff,diff)
#        norm_v2 = np.dot(v,v)
#
#        # z is the vector defined by the convex combination of v and w
#        def norm_z2(t):
#            z0 = v[0] +t*diff[0]
#            z1 = v[1] + t*diff[1]
#            return z0**2 + z1**2
#
#        # Sum of antipodal function values in the stencil
#        B = U[np.ix_(I+v[0],J+v[1])] + U[np.ix_(I-v[0],J-v[1])]
#        C = U[np.ix_(I+w[0],J+w[1])] + U[np.ix_(I-w[0],J-w[1])]
#
#        def D2(t):
#            return (-2*A + (1-t)*B + t*C)/(norm_z2(t)*dx**2)
#
#        tm, tp = np.zeros(A.shape), np.zeros(A.shape)
#
#        a = (B-C)*norm_diff2
#        b = 2*(2*A-B)*norm_diff2
#        c = (C+B-4*A)*norm_v2 + 2*(2*A-B)*vdotw
#
#        delta = b**2-4*a*c
#        ix = np.logical_and(B != C, delta>=0)
#        if ix.any():
#            aix, bix = a[ix], b[ix]
#            sqdelix = np.sqrt(delta[ix])
#
#            tm[ix], tp[ix] = (-bix-sqdelix)/(2*aix), (-bix+sqdelix)/(2*aix)
#
#        ix = np.logical_and(B==C, 2*A!=B)
#        if ix.any():
#            tm[ix] = -c[ix]/b[ix]
#            tp[ix] = tm[ix]
#
#        tm[np.logical_or(tm<0,tm>1)] = 0
#        tp[np.logical_or(tp<0,tp>1)] = 0
#        t = np.stack((np.zeros(A.shape),tm,tp,np.ones(A.shape)))
#
#        Dzz = D2(t)
#        if eigs=="both" or eigs=="min":
#            l_min = Dzz.argmin(0)
#            t_min[k,:,:] = t[l_min,Iint,Jint]
#            Dvv_min[k,:,:] = Dzz[l_min,Iint,Jint]
#
#        if eigs=="both" or eigs=="max":
#            l_max = Dzz.argmax(0)
#            t_max[k,:,:] = t[l_max,Iint,Jint]
#            Dvv_max[k,:,:] = Dzz[l_max,Iint,Jint]
#
#
#    if eigs=="both":
#        kmin = Dvv_min.argmin(0)
#        lambda_min = Dvv_min[kmin,Iint,Jint]
#        tmin = t_min[kmin,Iint,Jint]
#
#        kmax = Dvv_max.argmax(0)
#        lambda_max = Dvv_max[kmax,Iint,Jint]
#        tmax = t_max[kmax,Iint,Jint]
#
#        Lambda = lambda_min, lambda_max
#        Control = (kmin, tmin), (kmax, tmax)
#
#        return Lambda, Control
#    elif eigs=="min":
#        kmin = Dvv_min.argmin(0)
#        lambda_min = Dvv_min[kmin,Iint,Jint]
#        tmin = t_min[kmin,Iint,Jint]
#
#        return lambda_min, (kmin, tmin)
#    elif eigs=="max":
#        kmax = Dvv_max.argmax(0)
#        lambda_max = Dvv_max[kmax,Iint,Jint]
#        tmax = t_max[kmax,Iint,Jint]
#
#        return lambda_max, (kmax, tmax)
#
#def d2min(U,dx,**kwargs):
#    """
#    Compute the minimum eigenvalues of the Hessian of U.
#    Equivalent to calling d2eigs(u,dx,eigs="min")
#    """
#    return d2eigs(U,dx,**kwargs,eigs="min")
#
#def d2max(U,dx,**kwargs):
#    """
#    Compute the maximum eigenvalues of the Hessian of U.
#    Equivalent to calling d2eigs(u,dx,eigs="max")
#    """
#    return d2eigs(U,dx,**kwargs,eigs="max")
