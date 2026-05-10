"""Graded mesh utilities for the Butler-Volmer PNP solver."""

from __future__ import annotations

import firedrake as fd


# ---------------------------------------------------------------------------
# Domain-height bounds
# ---------------------------------------------------------------------------
#
# ``domain_height_hat`` decouples the mesh y-extent from L_REF so the
# transport-domain physical height L_eff_m can be swept independently of
# the global nondim scale.  See ``.claude/plans/l-eff-transport-sweep.md``
# for the L_eff sweep design and ``Forward/bv_solver/forms_logc.py`` for
# how the IC routines normalize their outer linear interpolations by
# this factor.

_DOMAIN_HEIGHT_HAT_MIN = 1e-3   # 0.1 µm at L_REF = 100 µm
_DOMAIN_HEIGHT_HAT_MAX = 10.0   # 1 mm at L_REF = 100 µm


def _validate_domain_height_hat(domain_height_hat: float) -> float:
    """Validate ``domain_height_hat`` is positive and within sanity bounds."""
    val = float(domain_height_hat)
    if val <= 0.0:
        raise ValueError(
            f"domain_height_hat must be positive; got {val}."
        )
    if val < _DOMAIN_HEIGHT_HAT_MIN or val > _DOMAIN_HEIGHT_HAT_MAX:
        raise ValueError(
            f"domain_height_hat={val} out of sanity range "
            f"[{_DOMAIN_HEIGHT_HAT_MIN}, {_DOMAIN_HEIGHT_HAT_MAX}]; "
            f"if intentional, raise the bounds in mesh.py."
        )
    return val


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
    domain_height_hat: float = 1.0,
) -> fd.Mesh:
    """Create a 2D rectangle mesh on [0, 1] x [0, domain_height_hat].

    The y-coordinate is stretched: ``y -> y^beta`` (in unit-cube space) and
    then scaled by ``domain_height_hat``.  ``beta > 1`` clusters nodes near
    ``y = 0`` (electrode / bottom).  The x-direction is uniform.  When
    ``domain_height_hat == 1.0`` the mesh is byte-identical to the legacy
    ``[0, 1]^2`` graded rectangle.

    Boundary markers (firedrake ``RectangleMesh`` convention):
        1 = left   (x=0, zero-flux)
        2 = right  (x=1, zero-flux)
        3 = bottom (y=0, electrode)
        4 = top    (y=domain_height_hat, bulk)

    Parameters
    ----------
    Nx : int
        Number of cells in x (tangential, uniform).
    Ny : int
        Number of cells in y (normal to electrode, graded).
    beta : float
        Grading exponent for the y-direction.
    domain_height_hat : float
        Mesh y-extent in nondim coords (= ``L_eff_m / L_REF``).  Default
        ``1.0`` reproduces the legacy ``[0, 1]^2`` rectangle.  Used by the
        L_eff transport sweep to vary the proton-transport ceiling without
        rescaling the global L_REF.
    """
    domain_height_hat = _validate_domain_height_hat(domain_height_hat)
    mesh = fd.RectangleMesh(Nx, Ny, 1.0, 1.0)
    coords = mesh.coordinates.dat.data
    # Stretch y in unit-cube coords first so the relative clustering near
    # y=0 is preserved, then scale to the requested physical extent.
    coords[:, 1] = (coords[:, 1] ** beta) * domain_height_hat
    return mesh
