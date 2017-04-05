from context import solvers, utils
#from solvers import directional_derivatives_interp as ddi
#from solvers import directional_derivatives_grid as ddg
from solvers import gridtools

import numpy as np


# Set up computational domain
N = 2**5+1;
d = 2;
xi = [0,1]

shape = [N for i in range(d)]
bounds = np.array([xi for i in range(d)]).T
r = 2

G = gridtools.uniform_grid(shape,bounds,r)
X = G.vertices[:,0]
Y = G.vertices[:,1]
if d==3:
    Z = G.vertices[:,2]
else:
    Z = 0

U1 = X**2 - Y**2 + Z**2
U2 = X - Y

if d==2:
    th = np.array([np.cos(np.pi/8), np.sin(np.pi/8)])
else:
    th = np.array([1/3,1/4,1/7])
    th /= np.linalg.norm(th)

# ------------------
mask = np.in1d(G.simplices[:,0], G.interior)
interior_simplices = G.simplices[mask]

I, S = interior_simplices[:,0], interior_simplices[:,1:]
V = G.vertices[S] - G.vertices[I,None]
V = np.swapaxes(V,1,2) # transpose last two axis before matrix inversion

Xi = np.linalg.solve(V,th[None,:,None]) # coordinates in the cone spanned by stencil vectors


def compute_d2(k):
    mask = interior_simplices[:,0]==k
    xi = Xi[mask]
    Vk = V[mask]
    ix = interior_simplices[mask]

    # Forward direction
    mask_cone_f = np.squeeze((xi>=0).all(axis=1)) # Farkas' lemma: coordinates must be non-negative
    V_f = Vk[mask_cone_f][0]
    xi_f =  xi[mask_cone_f][0]
    t_f = 1/np.sum(xi_f)

    # Backward direction
    mask_cone_b = np.squeeze((xi<=0).all(axis=1))
    V_b = Vk[mask_cone_b][0]
    xi_b =  xi[mask_cone_b][0]
    t_b = -1/np.sum(xi_b)

    t = np.min([t_f, t_b])

    xi_f = np.append(1-np.sum(t*xi_f),t*xi_f)
    xi_b = np.append(1-np.sum(-t*xi_b),-t*xi_b)

    i_f = ix[mask_cone_f][0]
    i_b = ix[mask_cone_b][0]

    u_f = U1[i_f].dot(xi_f)
    u_b = U1[i_b].dot(xi_b)

    return (-2*U1[k]+u_f+u_b)/t**2

d2u = np.array([compute_d2(k) for k in G.interior])
