"""
Tolman-Oppenheimer-Volkoff (TOV) equations.

This module defines the right-hand side of the TOV
equations for a spherically symmetric neutron star.

We use geometrized units:
    G = c = 1
"""

import numpy as np

from .eos import energy_density_from_pressure


def tov_rhs(r, m, P):
    """
    Calculate the derivatives dm/dr and dP/dr.

    Parameters
    ----------
    r : float
        Radial coordinate.
    m : float
        Mass enclosed within radius r.
    P : float
        Pressure at radius r.

    Returns
    -------
    dm_dr : float
        Derivative of mass with respect to radius.
    dP_dr : float
        Derivative of pressure with respect to radius.
    """

    epsilon = energy_density_from_pressure(P)

    dm_dr = 4 * np.pi * r**2 * epsilon

    dP_dr = -(
        (epsilon + P)
        * (m + 4 * np.pi * r**3 * P)
        / (r * (r - 2 * m))
    )

    return dm_dr, dP_dr