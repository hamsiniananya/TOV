"""
Numerical solver for the Tolman-Oppenheimer-Volkoff equations.

Uses the fourth-order Runge-Kutta (RK4) method
to integrate the TOV equations outward from
the center of the neutron star.
"""

import numpy as np

from .tov import tov_rhs
from .eos import energy_density_from_pressure


def solve_tov(P_c, dr=0.01, r_max=20.0):
    """
    Solve the TOV equations for a given central pressure.

    Parameters
    ----------
    P_c : float
        Central pressure.

    dr : float
        Radial step size.

    r_max : float
        Maximum radius to integrate to.

    Returns
    -------
    radii : numpy.ndarray
        Radial coordinates.

    masses : numpy.ndarray
        Enclosed mass at each radius.

    pressures : numpy.ndarray
        Pressure at each radius.
    """

    # Start slightly away from r = 0
    r = dr

    # Calculate central energy density
    epsilon_c = energy_density_from_pressure(P_c)

    # Approximate mass near the center:
    # m(r) ≈ (4/3) * pi * r^3 * epsilon_c
    m = (4 / 3) * np.pi * r**3 * epsilon_c

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

        # If pressure reaches zero inside this step,
        # stop instead of evaluating the EOS at negative pressure.
        if P2 <= 0:
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
            break

        k4_m, k4_P = tov_rhs(
            r + dr,
            m + dr * k3_m,
            P4
        )

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

        # Make sure pressure never becomes negative
        if P < 0:
            P = 0.0

        # Move outward
        r += dr

        # Store results
        radii.append(r)
        masses.append(m)
        pressures.append(P)

    return (
        np.array(radii),
        np.array(masses),
        np.array(pressures)
    )


# Test the solver
if __name__ == "__main__":

    radii, masses, pressures = solve_tov(
        P_c=0.001,
        dr=0.01,
        r_max=20.0
    )

    print("Number of points:", len(radii))
    print("Final radius:", radii[-1])
    print("Final mass:", masses[-1])
    print("Final pressure:", pressures[-1])