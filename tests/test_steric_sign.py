"""Fast sign-sanity test for the Bikerman steric chemical potential.

Documents the sign convention spelled out symbolically: the variational
derivative of the lattice-gas entropy density `(1-Phi)*ln(1-Phi)` produces
a `-ln(1-Phi)` excess chemical potential (Borukhov-Andelman-Orland 1997
eq (3); Bazant-Kilic-Storey-Ajdari 2009 eq (20)).  Old sign in
`Forward/bv_solver/forms_logc.py:266` was `+ln(1-Phi)`, which gave a
non-physical attractive divergence at saturation.  Post-fix line is
`mu_steric = -fd.ln(packing)`.

This test does not exercise the Firedrake form; it pins the sign
convention so any future reader sees the convention encoded as an
assertion.
"""
from __future__ import annotations

import math


def test_mu_steric_sign_at_saturation():
    """At Phi = 0.95 (near saturation), mu_steric should be large and
    POSITIVE (repulsive), not negative.  The new sign is `-ln(1-Phi)`.
    """
    phi = 0.95
    mu_old_sign = math.log(1.0 - phi)
    mu_new_sign = -math.log(1.0 - phi)

    assert mu_new_sign > 0, (
        "Post-fix mu_steric = -ln(1-Phi) must be positive at saturation"
    )
    assert mu_old_sign < 0, (
        "Pre-fix mu_steric = +ln(1-Phi) was negative at saturation "
        "(non-physical attractive sign)"
    )

    # Saturation behaviour: as Phi -> 1, |mu_new_sign| diverges to +infty
    assert -math.log(1.0 - 0.999) > -math.log(1.0 - 0.95) > 0
