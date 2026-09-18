"""
Mass-Radius relation for neutron stars.

This script solves the TOV equations for a range of
central pressures and plots the resulting neutron-star
mass against radius.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.solver import solve_tov


# Solar mass in kg
SOLAR_MASS = 1.98847e30


def generate_mass_radius_curve():
    """
    Generate the Mass-Radius relation.

    Returns
    -------
    radii_km : numpy.ndarray
        Neutron-star radii in km.

    masses_solar : numpy.ndarray
        Neutron-star masses in solar masses.
    """

    # Central pressure values [Pa]
    central_pressures = np.logspace(32, 36, 30)

    radii_km = []
    masses_solar = []

    for P_c in central_pressures:

        # Solve TOV equations
        radii, masses, pressures = solve_tov(
            P_c=P_c,
            dr=10.0,
            r_max=30000.0
        )

        # Surface values
        radius = radii[-1]
        mass = masses[-1]

        # Convert units
        radius_km = radius / 1000.0
        mass_solar = mass / SOLAR_MASS

        radii_km.append(radius_km)
        masses_solar.append(mass_solar)

    return np.array(radii_km), np.array(masses_solar)


def plot_mass_radius():
    """
    Plot the Mass-Radius relation.
    """

    radii_km, masses_solar = generate_mass_radius_curve()

    plt.figure(figsize=(8, 6))

    plt.plot(
        radii_km,
        masses_solar,
        marker="o",
        markersize=4,
        linewidth=2
    )

    plt.xlabel("Radius (km)")
    plt.ylabel("Mass (Solar Masses)")
    plt.title("Mass-Radius Relation of a Neutron Star")

    plt.grid(True)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    plot_mass_radius()