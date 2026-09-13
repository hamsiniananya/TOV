# This file explains how pressure, density, and energy density are related 
"""
Polytropic Equation of State (EOS)
for neutron-star matter.

We use geometrized units:
    G = c = 1

The EOS relates pressure, density,
and energy density.
"""

# Polytropic parameters
from .constants import K, GAMMA


def pressure_from_density(rho):
    """
    Calculate pressure from mass density.

    P = K * rho^Gamma
    """
    return K * rho**GAMMA


def density_from_pressure(P):
    """
    Calculate mass density from pressure.

    rho = (P / K)^(1/Gamma)
    """
    return (P / K)**(1 / GAMMA)


def energy_density(rho):
    """
    Calculate total energy density.

    epsilon = rho + P / (Gamma - 1)

    Since we use geometrized units, c = 1.
    """
    P = pressure_from_density(rho)

    return rho + P / (GAMMA - 1)


def energy_density_from_pressure(P):
    """
    Calculate energy density directly from pressure.

    First convert:
        P -> rho

    Then:
        rho -> epsilon
    """
    rho = density_from_pressure(P)

    return energy_density(rho)