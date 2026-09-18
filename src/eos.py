"""
Polytropic Equation of State (EOS)
for neutron-star matter.

SI units:
    rho     : mass density [kg/m^3]
    P       : pressure [Pa]
    epsilon : energy density [J/m^3]

The EOS relates pressure, mass density,
and energy density.
"""

import numpy as np

from .constants import c


def pressure_from_density(rho, K, gamma):
    """
    Calculate pressure from mass density.

    P = K * rho^Gamma

    Parameters
    ----------
    rho : float
        Mass density [kg/m^3]

    Returns
    -------
    P : float
        Pressure [Pa]
    """

    return K * rho**gamma


def density_from_pressure(P, K, gamma):
    """
    Calculate mass density from pressure.

    rho = (P / K)^(1/Gamma)

    Parameters
    ----------
    P : float
        Pressure [Pa]

    Returns
    -------
    rho : float
        Mass density [kg/m^3]
    """

    return (P / K)**(1 / gamma)


def energy_density(rho, K, gamma):
    """
    Calculate total energy density.

    epsilon = rho*c^2 + P/(Gamma - 1)

    Parameters
    ----------
    rho : float
        Mass density [kg/m^3]

    Returns
    -------
    epsilon : float
        Energy density [J/m^3]
    """

    P = pressure_from_density(rho, K, gamma)

    return rho * c**2 + P / (gamma - 1)


def energy_density_from_pressure(P, K, gamma):
    """
    Calculate energy density directly from pressure.

    First:
        P -> rho

    Then:
        rho -> epsilon
    """

    rho = density_from_pressure(P, K, gamma)

    return energy_density(rho, K, gamma)
