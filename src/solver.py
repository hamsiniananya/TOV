"""
Numerical solver for the Tolman-Oppenheimer-Volkoff equations.

Uses the fourth-order Runge-Kutta (RK4) method
to integrate the TOV equations outward from
the center of the neutron star.

All quantities are in SI units:
    radius   : m
    mass     : kg
    pressure : Pa
"""

import numpy as np

from .tov import tov_rhs
from .eos import density_from_pressure


def solve_tov(P_c, dr=10.0, r_max=30000.0):
    """
    Solve the TOV equations for a given central pressure.

    Parameters
    ----------
    P_c : float
        Central pressure [Pa].

    dr : float
        Radial step size [m].

    r_max : float
        Maximum radius to integrate to [m].

    Returns
    -------
    radii : numpy.ndarray
        Radial coordinates [m].

    masses : numpy.ndarray
        Enclosed mass [kg].

    pressures : numpy.ndarray
        Pressure [Pa].
    """

    # Start slightly away from r = 0
    r = dr

    # Calculate central mass density
    rho_c = density_from_pressure(P_c)

    # Approximate mass near the center:
    # m(r) ≈ (4/3) * pi * r^3 * rho_c
    m = (4 / 3) * np.pi * r**3 * rho_c

    # Central pressure
    P = P_c

    # Store results
    radii = [r]
    masses = [m]
    pressures = [P]

    while P > 0 and r < r_max:

        # -------------------------
        # RK4: first evaluation
        # -------------------------
        k1_m, k1_P = tov_rhs(r, m, P)

        # -------------------------
        # RK4: second evaluation
        # -------------------------
        P2 = P + dr * k1_P / 2

        if P2 <= 0:
            fraction = -P / (dr * k1_P)

            r_surface = r + fraction * dr
            m_surface = m + fraction * dr * k1_m

            radii.append(r_surface)
            masses.append(m_surface)
            pressures.append(0.0)

            break

        k2_m, k2_P = tov_rhs(
            r + dr / 2,
            m + dr * k1_m / 2,
            P2
        )

        # -------------------------
        # RK4: third evaluation
        # -------------------------
        P3 = P + dr * k2_P / 2

        if P3 <= 0:
            fraction = -P / (dr * k2_P)

            r_surface = r + fraction * (dr / 2)
            m_surface = m + fraction * (dr / 2) * k2_m

            radii.append(r_surface)
            masses.append(m_surface)
            pressures.append(0.0)

            break

        k3_m, k3_P = tov_rhs(
            r + dr / 2,
            m + dr * k2_m / 2,
            P3
        )

        # -------------------------
        # RK4: fourth evaluation
        # -------------------------
        P4 = P + dr * k3_P

        if P4 <= 0:
            fraction = -P / (dr * k3_P)

            r_surface = r + fraction * dr
            m_surface = m + fraction * dr * k3_m

            radii.append(r_surface)
            masses.append(m_surface)
            pressures.append(0.0)

            break

        k4_m, k4_P = tov_rhs(
            r + dr,
            m + dr * k3_m,
            P4
        )

        # -------------------------
        # Save previous values
        # -------------------------
        r_previous = r
        m_previous = m
        P_previous = P

        # -------------------------
        # Update mass
        # -------------------------
        m += (dr / 6) * (
            k1_m
            + 2 * k2_m
            + 2 * k3_m
            + k4_m
        )

        # -------------------------
        # Update pressure
        # -------------------------
        P += (dr / 6) * (
            k1_P
            + 2 * k2_P
            + 2 * k3_P
            + k4_P
        )

        # Move outward
        r += dr

        # Store normal point
        radii.append(r)
        masses.append(m)
        pressures.append(P)

    # Return the complete solution
    return (
        np.array(radii),
        np.array(masses),
        np.array(pressures)
    )


# Test the solver
if __name__ == "__main__":

    # Central pressure [Pa]
    P_c = 1.0e34

    radii, masses, pressures = solve_tov(
        P_c=P_c,
        dr=10.0,
        r_max=30000.0
    )

    # Convert SI mass to solar masses for display
    solar_mass = 1.98847e30

    print("Number of points:", len(radii))
    print("Final radius:", radii[-1] / 1000, "km")
    print("Final mass:", masses[-1] / solar_mass, "solar masses")
    print("Final pressure:", pressures[-1], "Pa")