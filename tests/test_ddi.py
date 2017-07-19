"""pytest ddi functions"""
import pytest

import numpy as np

from pyellipticfd import ddi
from setup_discs import disc


# Test on circular domain
@pytest.fixture(scope="module")
def Grid():
    h = 0.03
    Grid, plot_sol = disc(h)

    return Grid

def test_d1(Grid):
    X, Y = Grid.points.T
    U1 = X - Y
    d1x, M1x = ddi.d1(Grid,[1,0],U1, jacobian=True)
    d1y, M1y = ddi.d1(Grid,[0,1],U1, jacobian=True)

    assert np.abs(d1x-1).max() < 1e-4
    assert np.abs(M1x.dot(U1) - d1x).max() < 1e-12
    assert np.abs(d1y+1).max() < 1e-4
    assert np.abs(M1y.dot(U1) - d1y).max() < 1e-12

def test_grad(Grid):
    X, Y = Grid.points.T
    U1 = X - Y
    dgrad, Mgrad, vgrad = ddi.d1grad(Grid,U1,jacobian=True,control=True)

    assert np.abs(dgrad-np.sqrt(2)).max() < 1e-2
    assert np.abs(Mgrad.dot(U1) - dgrad).max() < 1e-12
    assert np.abs(vgrad.dot(1/np.sqrt(2)*np.array([1,-1])) - 1).max() < 1e-2

def test_normal(Grid):
    Un = np.linalg.norm(Grid.points, axis=1)
    d1n, M_d1n = ddi.d1n(Grid, u=Un, jacobian=True)

    assert np.abs(d1n+1).max() < 1e-2
    assert np.abs(M_d1n.dot(Un) - d1n).max() < 1e-12

def test_d2(Grid):
    X, Y = Grid.points.T
    U2 = X**2 - Y**2
    d2x, M2x = ddi.d2(Grid,[1,0],U2, jacobian=True)
    d2y, M2y = ddi.d2(Grid,[0,1],U2, jacobian=True)

    assert np.abs(d2x-2).max() < 0.15
    assert np.abs(M2x.dot(U2) - d2x).max() < 1e-12
    assert np.abs(d2y+2).max() < 0.15
    assert np.abs(M2y.dot(U2) - d2y).max() < 1e-12

def test_d2eigs1(Grid):
    X, Y = Grid.points.T
    U2 = X**2 - Y**2

    (d2min, M_min, v_min), (d2max, M_max, v_max) = ddi.d2eigs(Grid,U2, jacobian=True, control=True)

    assert np.abs(d2min +2).max() < 0.3
    assert np.abs(d2max -2).max() < 0.3
    assert np.abs(M_min.dot(U2) - d2min).max() < 1e-12
    assert np.abs(M_max.dot(U2) - d2max).max() < 1e-12
    assert np.abs((v_max.dot(np.array([1,0])))**2 - 1).max() < 0.025
    assert np.abs((v_min.dot(np.array([0,1])))**2 - 1).max() < 0.025

def test_d2eigs2(Grid):
    # This will use the cache
    X, Y = Grid.points.T
    U2 = X**2 - Y**2

    (d2min, M_min, v_min), (d2max, M_max, v_max) = ddi.d2eigs(Grid,U2, jacobian=True, control=True)

    assert np.abs(d2min +2).max() < 0.3
    assert np.abs(d2max -2).max() < 0.3
    assert np.abs(M_min.dot(U2) - d2min).max() < 1e-12
    assert np.abs(M_max.dot(U2) - d2max).max() < 1e-12
    assert np.abs((v_max.dot(np.array([1,0])))**2 - 1).max() < 0.025
    assert np.abs((v_min.dot(np.array([0,1])))**2 - 1).max() < 0.025
