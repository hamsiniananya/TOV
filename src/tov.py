"""
Tolman-Oppenheimer-Volkoff (TOV) equations.

This module defines the right-hand side of the TOV
equations for a spherically symmetric neutron star.

SI units:
    r   : radius [m]
    m   : enclosed mass [kg]
    P   : pressure [Pa]
    rho : mass density [kg/m^3]

Physical constants G and c are used explicitly.
"""

import numpy as np

from .constants import G, c
from .eos import density_from_pressure, energy_density_from_pressure


def tov_rhs(r, m, P, K, gamma):
    """
    Calculate the derivatives dm/dr and dP/dr.

    Parameters
    ----------
    r : float
        Radial coordinate [m].

    m : float
        Mass enclosed within radius r [kg].

    P : float
        Pressure at radius r [Pa].

    Returns
    -------
    dm_dr : float
        Derivative of mass with respect to radius [kg/m].

    dP_dr : float
        Derivative of pressure with respect to radius [Pa/m].
    """

    if P<=0:
        return 0.0, 0.0

    # Convert pressure to mass density using the EOS
    rho = density_from_pressure(P, K, gamma)

    epsilon = energy_density_from_pressure(P, K, gamma)
    rho_total = epsilon / c**2

    # Mass equation
    dm_dr = 4 * np.pi * r**2 * rho_total

    # TOV pressure equation
    dP_dr = -(
        G / r**2
        * (rho_total + P / c**2)
        * (m + 4 * np.pi * r**3 * P / c**2)
        / (1 - 2 * G * m / (r * c**2))
    )

    return dm_dr, dP_dr
