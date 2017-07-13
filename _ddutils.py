"""Utility functions for directional derivative routines."""

import numpy as np


def process_v(G,v,domain="interior"):
    """Utility function to process direction vector into correct format."""

    if domain=="interior":
        N = G.num_interior
    elif domain=="boundary":
        N = G.num_bdry
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
