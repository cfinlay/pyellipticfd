"""Solvers for arbitrary finite difference schemes."""

import numpy as np
import itertools
import warnings
import time
from scipy.sparse.linalg import spsolve, lsmr


def euler(U,operator,solution_tol=1e-4,max_iters=1e5,
            timeout=None,zeromean=False,zeromax=False,plotter=None):
    """
    Solve F[U] = 0 by iterating Euler steps until the
    stopping criteria |U^n+1 - U^n| < solution_tol.

    Parameters
    ----------
    U0 : array_like
        The initial condition.
    operator : function
        An function returning a tuple : first, the operator value F(U), including
        points on the boundary; and second, the CFL condition.
    solution_tol : scalar
        Stopping criterion, in the infinity norm.
    max_iters : scalar
        Maximum number of iterations.
    zeromean : boolean
        If the operator is only unique up to a constant, then setting zeromean
        to True tells the solver to choose the solution with zero mean, where
        each point is weighted equally.
    zeromean : boolean
        If the operator is only unique up to a constant, then setting zeromax
        to True tells the solver to choose the solution with zero max.
    plotter : function
        If provided, this function plots the solution every iteration.

    Returns
    -------
    U : array_like
        The solution.
    diff : scalar
        The maximum absolute difference between the solution and the previous
        iterate.
    i : scalar
        Number of iterations taken.
    time : scalar
        CPU time spent computing solution.
    """

    t0 = time.time()
    if not timeout is None:
        timeout = time.time()+timeout

    for i in itertools.count(1):
        FU, dt = operator(U)

        U_new = U - dt * FU

        if plotter:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plotter(U_new)

        diff = np.amax(np.absolute(U - U_new))
        if zeromean and zeromax:
            ValueError('Only one of zeromean and zeromax may be True')
        if zeromean:
            U = U_new - np.mean(U_new)
        elif zeromax:
            U = U_new - np.max(U_new)
        else:
            U = U_new

        if diff < solution_tol:
            return U, diff, i, time.time()-t0
        elif i >= max_iters:
            warnings.warn("Maximum iterations reached")
            return U, diff, i, time.time()-t0
        elif (not timeout is None) and (time.time() > timeout):
            warnings.warn("Maximum computation time reached")
            return U, diff, i, time.time()-t0

def newton(U,operator,solution_tol=1e-4,max_iters=1e2,
        euler_ratio=1, max_euler_iters=None, scipysolver = "spsolve",
        plotter=None):
    """
    Use semismooth Newton's method to find the steady state F[U]=0.

    Parameters
    ----------
    U : array_like
        The initial condition.
    operator : function
        An function returning a tuple of: the operator value F(U), the Jacobian, and
        the CFL condition.  The operator must return values for the boundary
        conditions as well.  The operator must have a boolean parameter
        'jacobian', which specifies whether to calculate the Jacobian matrix.
        If False then the Jacobian must be set to None.
    solution_tol : scalar
        Stopping criterion, in the infinity norm.
    max_iters : scalar
        Maximum number of iterations.
    euler_ratio : scalar
        In between Newton steps, the method perform Euler steps. The scalar
        euler_ratio gives the ratio of the time spent on Euler over the time
        spent doing a Newton step.  Defaults to 1, ie 50% of CPU time is
        spent on Euler steps.
    max_euler_iters : scalar
        Maximum allowable Euler iterations. Defaults to the number of grid points.
    scipysolver : string
        The scipy solver to use. Either 'spsolve' or 'lsmr'.
        Use 'lsmr' if the Jacobian has deficient rank.
    plotter : function
        If provided, this function plots the solution every iteration.

    Returns
    -------
    U : array_like
        The solution.
    diff : scalar
        The maximum absolute difference between the solution
        and the previous iterate.
    i : scalar
        Number of iterations taken.
    time : scalar
        CPU time spent computing solution.
    """
    t0 = time.time()

    # Operator for the Euler step
    def G(U):
        tup = operator(U,jacobian=False)
        return (tup[0], tup[2])

    # Set max number of Euler steps if not given
    if max_euler_iters is None:
        max_euler_iters=U.size

    for i in itertools.count(1):
        if euler_ratio is not None:
            tstart = time.time()

        Fu, Grad, _ = operator(U,jacobian=True)

        if scipysolver == 'spsolve':
            d = spsolve(Grad, -Fu)
        elif scipysolver == 'lsmr':
            d = lsmr(Grad, -Fu)[0]


        U_new = U + np.reshape(d,U.shape)
        diff = np.amax(np.absolute(U - U_new))
        U = U_new
        if scipysolver=='lsmr':
            U_new = U_new - U_new.max()

        if plotter:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plotter(U_new)

        if euler_ratio ==0:
            timeout = None
        elif euler_ratio is not None:
            NewtonTime = time.time()-tstart
            timeout = euler_ratio*NewtonTime
        else:
            timeout = None

        if (max_euler_iters is not 0) and (timeout is not None):
            zeromax=False
            if scipysolver=='lsmr':
                zeromax=True
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                U, diff, _, _ = euler(U, G,
                                    solution_tol=solution_tol,
                                    max_iters=max_euler_iters,
                                    timeout=timeout,zeromax=zeromax)

        if diff < solution_tol:
            return U, diff, i, time.time()-t0
        elif i+1 >= max_iters:
            warnings.warn("Maximum iterations reached")
            return U, diff, i, time.time()-t0
