"""pytest ddf functions"""
import pytest

import numpy as np
import distmesh as dm

from pyellipticfd import ddf
from setup_discs import disc


# Test on circular domain
@pytest.fixture(scope="module")
def get_disc():
    h = 0.03
    Grid, plot_sol = disc(h)

    X, Y = Grid.points.T
    U2 = X**2 - Y**2

    return Grid, U2

def test_d2(get_disc):
    Grid, U2 = get_disc
    d2f, M2f = ddf.d2(Grid,[1,0],U2, jacobian=True)

    assert np.abs(d2f-2).max() < 0.2

    assert np.abs(M2f.dot(U2) - d2f).max() < 1e-12

def test_d2eigs1(get_disc):
    # Should be close to 2

    Grid, U2 = get_disc
    (d2min, M_min, v_min), (d2max, M_max, v_max) = ddf.d2eigs(Grid,U2, jacobian=True, control=True)


    assert np.abs(d2min +2).max() < 0.3
    assert np.abs(d2max -2).max() < 0.3
    assert np.abs(M_min.dot(U2) - d2min).max() < 1e-12
    assert np.abs(M_max.dot(U2) - d2max).max() < 1e-12
    assert np.abs((v_max.dot(np.array([1,0])))**2 - 1).max() < 0.05
    assert np.abs((v_min.dot(np.array([0,1])))**2 - 1).max() < 0.05

def test_d2eigs2(get_disc):
    # This will use the cache
    Grid, U2 = get_disc
    (d2min, M_min, v_min), (d2max, M_max, v_max) = ddf.d2eigs(Grid,U2, jacobian=True, control=True)


    assert np.abs(d2min +2).max() < 0.3
    assert np.abs(d2max -2).max() < 0.3
    assert np.abs(M_min.dot(U2) - d2min).max() < 1e-12
    assert np.abs(M_max.dot(U2) - d2max).max() < 1e-12
    assert np.abs((v_max.dot(np.array([1,0])))**2 - 1).max() < 0.05
    assert np.abs((v_min.dot(np.array([0,1])))**2 - 1).max() < 0.05
