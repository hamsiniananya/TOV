"""
Pressure-Energy Density relation for neutron-star matter.

This script uses the polytropic equation of state defined
in eos.py and plots pressure against energy density.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.eos import pressure_from_density, energy_density


def generate_pressure_energy_data():
    """
    Generate pressure and energy-density values.

    Returns
    -------
    epsilon : numpy.ndarray
        Energy density [J/m^3].

    pressure : numpy.ndarray
        Pressure [Pa].
    """

    # Mass density range [kg/m^3]
    rho = np.logspace(14, 19, 500)

    # Pressure from EOS
    pressure = pressure_from_density(rho)

    # Energy density from EOS
    epsilon = energy_density(rho)

    return epsilon, pressure


def plot_pressure_energy_density():
    """
    Plot pressure against energy density.
    """

    epsilon, pressure = generate_pressure_energy_data()

    plt.figure(figsize=(8, 6))

    plt.loglog(
        epsilon,
        pressure,
        linewidth=2
    )

    plt.xlabel(r"Energy Density $\epsilon$ (J/m$^3$)")
    plt.ylabel("Pressure P (Pa)")
    plt.title("Pressure-Energy Density Relation")

    plt.grid(True, which="both")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_pressure_energy_density()