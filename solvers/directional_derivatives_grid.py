import numpy as np

__stencil = np.array([[0,1],
                [1,1],
                [1,0],
                [-1,1],
                [1,2],
                [2,1],
                [-2,1],
                [-1,2]])

def d2(U,v=[1,0],dx):
    """
    d2(U,v=[1,0],dx)

    Returns second derivative in the direction v using a centered difference.
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

    uvv = U[f] + U[b] - 2 * U[c]
    uvv = uvv/(norm_v*dx)**2
    return uvv
    
def d1abs(U,v=[1,0],dx):
    """
    d1abs(U,v=[1,0],dx)

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

def d2eigs(U,dx,stencil=__stencil,eigs="both")
    """
    d2eigs(U,dx,stencil=default_stencil,eigs="both") 
    
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

    widths = np.linalg.norm(stencil,axis=1,ord=np.inf)
    ix = np.argsort(widths))
    stencil = stencil[ix]
    widths = widths[ix]

    if eigs=="both" or eigs=="min":
        lambda_min = d2(U,stencil[0],dx)
    if eigs=="both" or eigs=="max":
        lambda_max = d2(U,stencil[0],dx)

    for (v, w) in zip(stencil[1:], widths[1:]):
        Dvv = d2(U,v,dx)
        if eigs=="both" or eigs=="min":
            lmn = lambda_min[(w-1):(Nx-1-w),(w-1):(Ny-1-w)]
            lambda_min[(w-1):(Nx-1-w),(w-1):(Ny-1-w)] = np.minimum(Dvv,lmn)
        if eigs=="both" or eigs=="max":
            lmx = lambda_min[(w-1):(Nx-1-w),(w-1):(Ny-1-w)]
            lambda_max[(w-1):(Nx-1-w),(w-1):(Ny-1-w)] = np.maximum(Dvv,lmx)

    if eigs=="both":
        return lambda_min, lambda_max
    elif eigs=="min":
        return lambda_min
    elif eigs=="max":
        return lambda_max

def d2min(U,dx,stencil=__stencil):
    """
    d2min(u,dx,stencil=default_stencil) 
    
    Compute the minimum eigenvalues of the Hessian of U.
    Equivalent to calling d2eigs(u,dx,eigs="min")
    """
    Lambda = d2eigs(U,dx,eigs="min",stencil)
    return Lambda

def d2max(U,dx,stencil=__stencil):
    """
    d2max(u,dx,stencil=default_stencil) 
    
    Compute the maximum eigenvalues of the Hessian of U.
    Equivalent to calling d2eigs(u,dx,eigs="max")
    """
    Lambda = d2eigs(U,dx,eigs="max",stencil)
    return Lambda