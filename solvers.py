"""Solvers for arbitrary finite difference schemes."""

import numpy as np
import itertools
import warnings
from warnings import warn
import time
from scipy.sparse import diags
from scipy.sparse.linalg import  bicgstab
from scipy.sparse.linalg.dsolve.linsolve import MatrixRankWarning


def euler(U,operator,solution_tol=1e-6,operator_tol=1e-6,max_iters=None,
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
        Stopping criterion of the solution, in the infinity norm.
    operator_tol : scalar
        Stopping critera of the residual, ie the infinity norm.
    max_iters : scalar
        Maximum number of iterations.
    timeout : scalar
        Exit if computation time exceeds this value.
    zeromean : boolean
        If the operator is only unique up to a constant, then setting zeromean
        to True tells the solver to choose the solution with zero mean, where
        each point is weighted equally.
    zeromax : boolean
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
    if max_iters is None:
        max_iters = U.size

    t0 = time.time()
    if not timeout is None:
        timeout = time.time()+timeout

    for i in itertools.count(1):
        try:
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

            if (diff < solution_tol) and np.abs(FU).max() < operator_tol:
                return U, diff, i, time.time()-t0
            elif i >= max_iters:
                warn("Maximum iterations reached")
                return U, diff, i, time.time()-t0
            elif (not timeout is None) and (time.time() > timeout):
                warn("Maximum computation time reached")
                return U, diff, i, time.time()-t0
        except KeyboardInterrupt:
            return U, diff, i, time.time()-t0

class NewtonDecreaseWarning(UserWarning):
    pass

def NewtonEulerLS(U,operator,solution_tol=1e-6,operator_tol = 1e-6, max_iters=1e2,
        euler_ratio=1,plotter=None, max_euler_iters=None,
        verbose = False, force_euler=False):
    """
    Use semismooth Newton's method to find the steady state F[U]=0. If an
    insufficient decrease is detected in the Newton step, switch to Euler
    steps. 

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
        Stopping criterion of the solution, in the infinity norm.
    operator_tol : scalar
        Stopping critera of the residual, ie the infinity norm.
    max_iters : scalar
        Maximum number of iterations.
    euler_ratio : scalar
        If a Newton step fails, the algorithm switches to performing Euler
        steps.  The scalar euler_ratio gives the proportion of time spent on
        Euler steps relative to one Newton step. Defaults to 1.
    plotter : function
        If provided, this function plots the solution every iteration.
    max_euler_iters : scalar
        Maximum number of iterations during Euler steps.
    verbose : boolean
        Whether to print diagnostics.
    force_euler : boolean
        If True, then always perform Euler steps after each Newton step.

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

    # Safety checks
    if euler_ratio <=0:
        raise ValueError('euler_ratio must be positive')

    newtontime = 1.0 # Default time to perform newton step, in case Jacobian
                     # is singular at U0


    # Sufficient decrease parameters
    rho, p = 0.5, 2

    # Operator for the Euler step
    def G(U):
        tup = operator(U,jacobian=False)
        return (tup[0], tup[2])

    # Set max number of Euler steps if not given
    max_euler_iters=10*U.size

    for i in itertools.count(0):


        euler_flag = False
        try: # TODO simplify warning and exception handling
            with warnings.catch_warnings():
                warnings.simplefilter('error')

                tstart = time.time()
                Gu, Jac, _ = operator(U,jacobian=True)

                P = diags(1/Jac.diagonal(),format='csr') # Preconditioner

                d, info = bicgstab(Jac, -Gu, M=P, tol=np.min([operator_tol,solution_tol]))

                newtontime = time.time()-tstart
                timeout = euler_ratio*newtontime

                if info > 0:
                    warn('Failed to invert Jacobian', MatrixRankWarning)

                U_new = U + d

                # Check for sufficient decrease
                gradTheta = Jac.transpose().dot(Gu)
                normp = np.linalg.norm(d)**p

                if gradTheta.dot(d) > - rho * normp:
                    warn('Insufficient decrease in Newton step',
                         NewtonDecreaseWarning)

                diff = np.amax(np.absolute(U - U_new))
                U = U_new
        except (MatrixRankWarning,NewtonDecreaseWarning) as e:
            if verbose:
                print('At iteratate',i,':',e,'; switching to Euler')
            euler_flag=True


            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                U, diff, _, _ = euler(U, G,timeout = timeout,
                                    solution_tol=solution_tol,
                                    operator_tol=operator_tol,
                                    max_iters=max_euler_iters)

        if force_euler and not euler_flag:
            warnings.simplefilter("ignore")
            U, diff, _, _ = euler(U, G,timeout = timeout,
                                solution_tol=solution_tol,
                                operator_tol=operator_tol,
                                max_iters=max_euler_iters)
            Gu,_,_ = operator(U,jacobian=False) 


        if plotter:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plotter(U)


        if (diff < solution_tol) and np.abs(Gu).max() < operator_tol:
            return U, diff, i+1, time.time()-t0
        elif i >= max_iters:
            warn("Maximum iterations reached")
            return U, diff, i+1, time.time()-t0
