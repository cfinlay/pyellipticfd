#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 13:19:39 2017

@author: bilalabbasi
"""
import numpy as np

def convex_linesolver1D(U):
    #Computes the convex envelope along a 1D line.
    ind = 0
    U_ce = np.copy(U)
    N = np.shape(U)[0] #Number of grid points
    x = np.arange(N) # Index
    while ind < len(U)-1:
        slope = (U_ce[ind+1:] - U_ce[ind])/(x[ind+1:]-x[ind]) #Calculate slopes from reference point
        m = np.min(slope) #Retrieve the smallest one
        i = np.argmin(slope)+ind+1 #Update location of reference point
        U_ce[ind:i] = m*(x[ind:i]-x[ind]) + U_ce[ind] #Create line from (old) reference point and where smallest slope occurs
        ind = i
    return U_ce
    
def convex_linesolver_line(Nx,Ny,U,v,i=0,j=0):
    #Computes the convex envelope of U along the line v, starting at the index (i,j)
    U_ce = np.copy(U)
    #If either increment in (v0,v1) is 0, have to manually define index vector
    if v[0] == 0:
        ind_x = np.ones(Nx) * i
        ind_y = np.arange(j,Ny,v[1],dtype=np.intp)
    elif v[1]==0:
        ind_x = np.arange(i,Nx,v[0],dtype=np.intp)
        ind_y = np.ones(Ny) * j
    else:
        ind_x = np.arange(i,Nx,v[0],dtype=np.intp)
        ind_y = np.arange(j,Ny,v[1],dtype=np.intp)
    
    #Make sure you only go as far as you can in BOTH indices
    max_ind = np.min([len(ind_x),len(ind_y)])
    ind_x = ind_x[:max_ind].astype(np.intp)
    ind_y = ind_y[:max_ind].astype(np.intp)
    
    #Now apply the line solver
    u_v = U_ce[ind_x,ind_y]           #Retrieve obstacle along line
    u_v_CE = convex_linesolver1D(u_v) #Apply 1D solver to line
    U_ce[ind_x,ind_y] = u_v_CE        #Insert 1D solution back in

    return U_ce
    
def convex_linesolver(U,stencil=np.array([[0,1],[1,0]])):
    #Computes the convex envelope by iterating the 1D linesolver along the lines with direction v.
    #Solution propagates from the left and bottom boundaries.
    U_ce = np.copy(U)
    if len(np.shape(U)) == 1:
        #If array is one dimensional, apply 1D solver directly.
        U_ce = convex_linesolver1D(U)
    else:
        #Else iterate line solver from the boundary.
        Nx,Ny = np.shape(U)
        for v in stencil:
            for i in range(0,Nx):
                U_ce = convex_linesolver_line(Nx,Ny,U_ce,v,i,0)
            for i in range(1,Ny):
                U_ce = convex_linesolver_line(Nx,Ny,U_ce,v,0,i)
    return U_ce

