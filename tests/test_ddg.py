import pytest
import numpy as np

from pyellipticfd import ddg
from pyellipticfd.pointclasses import FDRegularGrid, FDTriMesh
from setup_discs import disc


# Test on circular domain
@pytest.fixture(scope="module")
def Grid():
    h = 0.075
    Grid, plot_sol = disc(h)

    return Grid

@pytest.fixture(scope="module")
def RegGrid():
    # Set up computational domain
    N = 9;
    d = 2;
    xi = [0,1]

    shape = [N for i in range(d)]
    bounds = np.array([xi for i in range(d)]).T
    r = 2

    G = FDRegularGrid(shape,bounds,r,interpolation=False)
    return G

def test_d1(RegGrid):
    X, Y = RegGrid.points.T
    U1 = X - Y
    d1x, M1x = ddg.d1(RegGrid,[1,0],U1, jacobian=True)
    d1y, M1y = ddg.d1(RegGrid,[0,1],U1, jacobian=True)

    assert np.abs(d1x-1).max() < 1e-8
    assert np.abs(M1x.dot(U1) - d1x).max() < 1e-12
    assert np.abs(d1y+1).max() < 1e-8
    assert np.abs(M1y.dot(U1) - d1y).max() < 1e-12

def test_grad(RegGrid):
    X, Y = RegGrid.points.T
    U1 = X - Y
    dgrad, Mgrad, vgrad = ddg.d1grad(RegGrid,U1,jacobian=True,control=True)

    assert np.abs(dgrad-np.sqrt(2)).max() < 1e-8
    assert np.abs(Mgrad.dot(U1) - dgrad).max() < 1e-12
    assert np.abs(vgrad.dot(1/np.sqrt(2)*np.array([1,-1])) - 1).max() < 1e-8

def test_normal(Grid):
    Un = np.linalg.norm(Grid.points, axis=1)
    d1n, M_d1n = ddg.d1n(Grid, u=Un, jacobian=True)

    assert np.abs(d1n+1).max() < 5e-2
    assert np.abs(M_d1n.dot(Un) - d1n).max() < 1e-12

def test_d2(RegGrid):
    X, Y = RegGrid.points.T
    U2 = X**2 - Y**2
    d2x, M2x = ddg.d2(RegGrid,[1,0],U2, jacobian=True)
    d2y, M2y = ddg.d2(RegGrid,[0,1],U2, jacobian=True)

    assert np.abs(d2x-2).max() < 1e-8
    assert np.abs(M2x.dot(U2) - d2x).max() < 1e-12
    assert np.abs(d2y+2).max() < 1e-8
    assert np.abs(M2y.dot(U2) - d2y).max() < 1e-12

def test_d2eigs(RegGrid):
    X, Y = RegGrid.points.T
    U2 = X**2 - Y**2

    (d2min, M_min, v_min), (d2max, M_max, v_max) = ddg.d2eigs(RegGrid,U2, jacobian=True, control=True)

    assert np.abs(d2min +2).max() < 1e-8
    assert np.abs(d2max -2).max() < 1e-8
    assert np.abs(M_min.dot(U2) - d2min).max() < 1e-12
    assert np.abs(M_max.dot(U2) - d2max).max() < 1e-12
    assert np.abs((v_max.dot(np.array([1,0])))**2 - 1).max() < 1e-8
    assert np.abs((v_min.dot(np.array([0,1])))**2 - 1).max() < 1e-8
