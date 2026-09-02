"""
Polytropic equation of state for neutron-star matter.

This module contains functions relating:

    - mass density
    - pressure
    - energy density
"""

# Polytropic parameters
K = 100.0
Gamma = 2.0


def pressure_from_density(rho):
    """
    Calculate pressure from mass density.

    P = K * rho^Gamma
    """
    return K * rho**Gamma


def energy_density_from_density(rho):
    """
    Calculate total energy density from mass density.

    epsilon = rho + P / (Gamma - 1)

    In our geometrized units, c = 1.
    """
    P = pressure_from_density(rho)
    return rho + P / (Gamma - 1)


def density_from_pressure(P):
    """
    Calculate mass density from pressure.

    rho = (P / K)^(1/Gamma)
    """
    return (P / K)**(1 / Gamma)

# Test the EOS
rho = 0.001

P = pressure_from_density(rho)
epsilon = energy_density_from_density(rho)
rho_check = density_from_pressure(P)

print("Density:", rho)
print("Pressure:", P)
print("Energy density:", epsilon)
print("Recovered density:", rho_check)