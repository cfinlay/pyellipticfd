#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 14:39:48 2017
Contains functions to compute first and second order directional derivaties.
@author: bilalabbasi
"""
import numpy as np

def Uvv(U,v=[1,0]):
    #Returns second derivative in the direction v using a centered difference.
    Nx = U.shape[0]
    Ny = U.shape[1]
    w = np.linalg.norm(v,np.inf)              #width of vectors
    ind_x = np.arange(w,Nx-w,dtype = np.intp) #recover interior index appropriate for stencil of width W
    ind_y = np.arange(w,Ny-w,dtype = np.intp)
    c = np.ix_(ind_x,ind_y)                       #center index
    f = np.ix_(ind_x-v[0],ind_y-v[1]) 
    b = np.ix_(ind_x+v[0],ind_y+v[1])
    uvv = U[f] + U[b] - 2 * U[c]
    return uvv
    
def Uv(U,v=[1,0]):
    #Returns (absolute) directional directive in direction v using forward/backward differences
    #Caution: this difference is not monotone!
    n = U.shape[0]
    W = np.linalg.norm(v,np.inf).astype(int)   #width of vectors
    ind = np.arange(W,n-W)                     #recover interior index appropriate for stencil of width W
    c = np.ix_(ind,ind)                        #center index
    f = np.ix_(ind-v[0],ind-v[1]) 
    b = np.ix_(ind+v[0],ind+v[1])
    uv = np.maximum(U[f],U[b]) - U[c]
    return uv