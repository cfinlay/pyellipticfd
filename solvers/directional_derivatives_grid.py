import numpy as np

#TODO: stencil type should agree with ddi
stencil = np.array([[  1,  0],
                      [  1,  1],
                      [  0,  1],
                      [ -1,  1]], dtype=np.intp)

def d2(U,dx,stencil,ix=np.array([0])):
    """
    Returns second derivative in the direction v using a centered difference.
    """
    Nx, Ny = U.shape
    if ix.size==1:
        if stencil.ndim==1:
            v = stencil
        else:
            v = stencil[ix]

        w = np.max(np.abs(v))       #width of vector
        norm_v = np.linalg.norm(v)  #length of the vector

        #recover interior index appropriate for stencil of width w
        ind_x = np.arange(w,Nx-w,dtype = np.intp)
        ind_y = np.arange(w,Ny-w,dtype = np.intp)

        c = np.ix_(ind_x,ind_y)              #center index
        f = np.ix_(ind_x-v[0],ind_y-v[1])    #forward
        b = np.ix_(ind_x+v[0],ind_y+v[1])    #backward

        uvv = U[f] + U[b] - 2 * U[c]
        uvv = uvv/(norm_v*dx)**2
        return uvv
    else:
        v = stencil[ix,:]
        normv2 = stencil[ix,0]**2 + stencil[ix,1]**2
        minw = np.min(np.abs(stencil).max(1))

        I, J =  np.indices(U.shape)
        I = I[minw:Nx-minw,minw:Ny-minw]
        J = J[minw:Nx-minw,minw:Ny-minw]

        C = U[I,J]
        F = U[I+v[:,:,0],J+v[:,:,1]]
        B = U[I-v[:,:,0],J-v[:,:,1]]

        return (-2*C + F + B)/(dx**2 * normv2)




def d1abs(U,dx,v):
    """
    Returns (absolute) directional directive in direction v using
    forward/backward differences.

    Caution: this difference is not monotone!
    """
    Nx, Ny = U.shape
    w = np.max(np.abs(v))       #width of vectors
    norm_v = np.linalg.norm(v)  #length of the vector

    #recover interior index appropriate for stencil of width w
    ind_x = np.arange(w,Nx-w,dtype = np.intp)
    ind_y = np.arange(w,Ny-w,dtype = np.intp)

    c = np.ix_(ind_x,ind_y)              #center index
    f = np.ix_(ind_x-v[0],ind_y-v[1])    #forward
    b = np.ix_(ind_x+v[0],ind_y+v[1])    #backward

    uv = np.maximum(U[f],U[b]) - U[c]
    uv = uv/(dx*norm_v)
    return uv

def d2eigs(U,dx,stencil=stencil,eigs="both"):
    """
    Compute the maximum and minimum eigenvalues of the Hessian of U.

    Parameters
    ----------
    u : array_like
        Function values at grid points.
    dx : scalar
        Uniform grid resolution.
    stencil : list
        A list of k stencil directions, with shape (k,2).
    eigs : string
        Specify which eigenvalue to retrieve: "min", "max", or "both".

    Returns
    -------
    Lambda : a list, or an array
        If eigs="both", a list containing the minimal and maximal eigenvalues,
        with the minimal eigenvalue first.
        If eigs!="both", then an array of the specified eigenvalue.
    """
    Nx, Ny = U.shape

    widths = np.abs(stencil).max(1)

    # Assume first vector in stencil has width 1
    # TODO: otherwise throw Exception
    if eigs=="both" or eigs=="min":
        lambda_min = d2(U,dx,stencil[0])
        ix_min = np.zeros(lambda_min.shape, dtype=np.intp)
    if eigs=="both" or eigs=="max":
        lambda_max = d2(U,dx,stencil[0])
        ix_max = np.zeros(lambda_max.shape, dtype=np.intp)

    for k, (v, w) in enumerate(zip(stencil[1:], widths[1:]),1):
        Dvv = d2(U,dx,v)
        if eigs=="both" or eigs=="min":
            l = lambda_min[(w-1):(Nx-1-w),(w-1):(Ny-1-w)]
            subix = ix_min[(w-1):(Nx-1-w),(w-1):(Ny-1-w)]

            bl = Dvv < l
            l[bl] = Dvv[bl]
            subix[bl] = k
        if eigs=="both" or eigs=="max":
            l = lambda_max[(w-1):(Nx-1-w),(w-1):(Ny-1-w)]
            subix = ix_max[(w-1):(Nx-1-w),(w-1):(Ny-1-w)]

            bl = Dvv > l
            l[bl] = Dvv[bl]
            subix[bl] = k

    if eigs=="both":
        L = (lambda_min, lambda_max)
        Ix = (ix_min, ix_max)
        return L, Ix
    elif eigs=="min":
        return lambda_min, ix_min
    elif eigs=="max":
        return lambda_max, ix_max

def d2min(U,dx,**kwargs):
    """
    Compute the minimum eigenvalues of the Hessian of U.
    Equivalent to calling d2eigs(u,dx,eigs="min")
    """
    return d2eigs(U,dx,eigs="min",**kwargs)

def d2max(U,dx,**kwargs):
    """
    Compute the maximum eigenvalues of the Hessian of U.
    Equivalent to calling d2eigs(u,dx,eigs="max")
    """
    return d2eigs(U,dx,eigs="max",**kwargs)
