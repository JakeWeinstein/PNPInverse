"""Graded mesh utilities for the Butler-Volmer PNP solver."""

from __future__ import annotations

import firedrake as fd


# ---------------------------------------------------------------------------
# Graded mesh utilities
# ---------------------------------------------------------------------------

def make_graded_interval_mesh(N: int = 300, beta: float = 2.0) -> fd.Mesh:
    """Create a 1D interval mesh on [0, 1] with power-law grading.

    Points are placed at ``x_i = (i/N)^beta`` for ``i = 0, ..., N``.
    ``beta > 1`` clusters nodes near ``x = 0`` (electrode).

    Boundary markers:
        1 = left  (x=0, electrode)
        2 = right (x=1, bulk)

    Parameters
    ----------
    N : int
        Number of cells (elements).
    beta : float
        Grading exponent.  beta=1 gives uniform spacing; beta=2 gives
        quadratic clustering near x=0.
    """
    mesh = fd.IntervalMesh(N, 1.0)
    coords = mesh.coordinates.dat.data
    # Apply power-law stretching: x -> x^beta
    coords[:] = coords[:] ** beta
    return mesh


def make_graded_rectangle_mesh(
    Nx: int = 8,
    Ny: int = 300,
    beta: float = 2.0,
) -> fd.Mesh:
    """Create a 2D rectangle mesh on [0, 1]^2 with power-law grading in y.

    The y-coordinate is stretched: ``y -> y^beta``.  ``beta > 1`` clusters
    nodes near ``y = 0`` (electrode / bottom).  The x-direction is uniform.

    Boundary markers (firedrake ``RectangleMesh`` convention):
        1 = left   (x=0, zero-flux)
        2 = right  (x=1, zero-flux)
        3 = bottom (y=0, electrode)
        4 = top    (y=1, bulk)

    Parameters
    ----------
    Nx : int
        Number of cells in x (tangential, uniform).
    Ny : int
        Number of cells in y (normal to electrode, graded).
    beta : float
        Grading exponent for the y-direction.
    """
    mesh = fd.RectangleMesh(Nx, Ny, 1.0, 1.0)
    coords = mesh.coordinates.dat.data
    coords[:, 1] = coords[:, 1] ** beta
    return mesh
