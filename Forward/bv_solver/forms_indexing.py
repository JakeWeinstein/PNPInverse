"""Mixed-function-space DOF indexing helper (Phase 6β v9 Gate 3A).

The bv_solver mixed space normally has shape ``[V_scalar]*n_species +
[V_scalar (phi)]``.  Phase 6β v9 Gate 3A extends it conditionally with an
R-space global scalar for the surface coverage ``Γ_MOH`` of the
cation-hydrolysis residual::

    legacy   :   [V_scalar]*n_species + [V_scalar (phi)]
    +Γ slot  :   [V_scalar]*n_species + [V_scalar (phi)] + [R_space (Γ)]

Form-build call-sites used to slice the mixed components with the
hard-coded patterns ``fd.split(U)[:-1]`` / ``fd.split(U)[-1]``.  Those
break silently when ``has_gamma=True`` because ``[-1]`` then points at
``Γ`` rather than ``φ``.

This helper centralises the slice / index logic so every call-site can
stay agnostic of the layout.  When ``has_gamma=False`` the indices match
the legacy patterns byte-for-byte; when ``has_gamma=True`` ``φ`` shifts
to position ``-2`` and ``Γ`` lands at ``-1``.

Pure-Python; no Firedrake import.  Lets the fast unit suite verify the
indexing without spinning up FE infrastructure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SpeciesPhiGammaIndices:
    """DOF-position indices for the species + φ (+ Γ) mixed space.

    Attributes
    ----------
    species_slice
        Python ``slice`` covering all dynamic-species components.  For
        both legacy and Γ-augmented layouts the species occupy indices
        ``0..n_species-1``; the slice expresses that via negative
        right-bound (``-1`` legacy, ``-2`` Γ-augmented) so it does not
        need to know ``n_species``.
    phi_index
        Position of the ``φ`` component using negative indexing
        (``-1`` legacy, ``-2`` Γ-augmented).  Negative form chosen so
        the same index works on both ``fd.split(U)`` (a tuple) and
        ``fd.TestFunctions(W)`` (also a tuple).
    gamma_index
        Position of the ``Γ_MOH`` R-space component (always ``-1`` when
        present).  ``None`` in legacy layout — callers should short-
        circuit on ``is None`` rather than catch IndexError.
    has_gamma
        Convenience flag mirroring the constructor argument.
    """

    species_slice: slice
    phi_index: int
    gamma_index: Optional[int]
    has_gamma: bool


def unpack_dof_indices(*, has_gamma: bool) -> SpeciesPhiGammaIndices:
    """Return DOF-position indices for the current mixed-space layout.

    Parameters
    ----------
    has_gamma
        ``True`` iff the form was built with
        ``enable_cation_hydrolysis=True`` (mixed space includes one
        extra R-space slot at the end for Γ_MOH).  ``False`` for the
        legacy layout.

    Returns
    -------
    SpeciesPhiGammaIndices
        Indices that bridge legacy and Γ-augmented layouts uniformly.
        Caller can use ``species_slice`` against ``fd.split(U)`` /
        ``fd.TestFunctions(W)`` for the species block, ``phi_index``
        for the potential, and ``gamma_index`` for ``Γ`` (when present).
    """
    if has_gamma:
        return SpeciesPhiGammaIndices(
            species_slice=slice(0, -2),
            phi_index=-2,
            gamma_index=-1,
            has_gamma=True,
        )
    return SpeciesPhiGammaIndices(
        species_slice=slice(0, -1),
        phi_index=-1,
        gamma_index=None,
        has_gamma=False,
    )


__all__ = [
    "SpeciesPhiGammaIndices",
    "unpack_dof_indices",
]
