"""
TOV Solver

Solves the Tolman-Oppenheimer-Volkoff equations
using the fourth-order Runge-Kutta (RK4) method.

SI units:
    r   : radius [m]
    m   : enclosed mass [kg]
    P   : pressure [Pa]

K and gamma are supplied as EOS parameters.
"""

import numpy as np

from .constants import G, c
from .eos import (
    density_from_pressure,
    energy_density_from_pressure
)
from .tov import tov_rhs


def solve_tov(P_c, K, gamma, dr=10.0, r_max=30000.0):
    """
    Solve the TOV equations for a given central pressure.

    Parameters
    ----------
    P_c : float
        Central pressure [Pa]

    K : float
        Polytropic constant

    gamma : float
        Polytropic index

    dr : float
        Radial step size [m]

    r_max : float
        Maximum radius allowed [m]

    Returns
    -------
    radii : numpy.ndarray
        Radius values [m]

    masses : numpy.ndarray
        Enclosed mass values [kg]

    pressures : numpy.ndarray
        Pressure values [Pa]

    surface_reached : bool
        True if pressure reached zero before r_max.
    """

    # -----------------------------------------------------
    # Central density and energy density
    # -----------------------------------------------------

    rho_c = density_from_pressure(P_c, K, gamma)

    epsilon_c = energy_density_from_pressure(P_c, K, gamma)

    # Convert energy density [J/m^3] to mass-energy
    # density [kg/m^3]
    rho_energy_c = epsilon_c / c**2

    # -----------------------------------------------------
    # Initial radius
    # -----------------------------------------------------

    r = dr

    # Initial mass assuming approximately uniform
    # central energy density in the small central region
    m = (4.0 / 3.0) * np.pi * r**3 * rho_energy_c

    P = P_c

    # -----------------------------------------------------
    # Store solution
    # -----------------------------------------------------

    radii = [r]
    masses = [m]
    pressures = [P]

    # -----------------------------------------------------
    # Surface flag
    # -----------------------------------------------------

    surface_reached = False

    # -----------------------------------------------------
    # Main RK4 integration loop
    # -----------------------------------------------------

    while r < r_max and P > 0:

        # ---------------------------------------------
        # RK4: k1
        # ---------------------------------------------

        k1_m, k1_P = tov_rhs(
            r,
            m,
            P,
            K,
            gamma
        )

        # ---------------------------------------------
        # RK4: k2
        # ---------------------------------------------

        k2_m, k2_P = tov_rhs(
            r + dr / 2.0,
            m + dr * k1_m / 2.0,
            P + dr * k1_P / 2.0,
            K,
            gamma
        )

        # ---------------------------------------------
        # RK4: k3
        # ---------------------------------------------

        k3_m, k3_P = tov_rhs(
            r + dr / 2.0,
            m + dr * k2_m / 2.0,
            P + dr * k2_P / 2.0,
            K,
            gamma
        )

        # ---------------------------------------------
        # RK4: k4
        # ---------------------------------------------

        k4_m, k4_P = tov_rhs(
            r + dr,
            m + dr * k3_m,
            P + dr * k3_P,
            K,
            gamma
        )

        # ---------------------------------------------
        # Calculate next values
        # ---------------------------------------------

        new_m = m + (
            dr / 6.0
        ) * (
            k1_m
            + 2.0 * k2_m
            + 2.0 * k3_m
            + k4_m
        )

        new_P = P + (
            dr / 6.0
        ) * (
            k1_P
            + 2.0 * k2_P
            + 2.0 * k3_P
            + k4_P
        )

        new_r = r + dr

        # ---------------------------------------------
        # Check whether the pressure crossed zero
        # ---------------------------------------------

        if new_P <= 0:

            # Linear interpolation to estimate the
            # radius at which P = 0.
            if P != new_P:

                fraction = P / (P - new_P)

                surface_r = r + fraction * dr

                surface_m = m + fraction * (new_m - m)

            else:

                surface_r = new_r
                surface_m = new_m

            radii.append(surface_r)
            masses.append(surface_m)
            pressures.append(0.0)

            surface_reached = True

            break

        # ---------------------------------------------
        # Accept the new RK4 values
        # ---------------------------------------------

        r = new_r
        m = new_m
        P = new_P

        radii.append(r)
        masses.append(m)
        pressures.append(P)

    # -----------------------------------------------------
    # Return solution
    # -----------------------------------------------------

    return (
        np.array(radii),
        np.array(masses),
        np.array(pressures),
        surface_reached
    )


def generate_mass_radius_sequence(
    central_pressures,
    K,
    gamma,
    dr=10.0,
    r_max=30000.0
):
    """
    Generate a mass-radius sequence for multiple
    central pressures.

    Parameters
    ----------
    central_pressures : array-like
        Central pressures [Pa]

    K : float
        Polytropic constant

    gamma : float
        Polytropic index

    dr : float
        Radial step size [m]

    r_max : float
        Maximum radius [m]

    Returns
    -------
    results : list of dictionaries
        Results for each central pressure.
    """

    results = []

    for P_c in central_pressures:

        (
            radii,
            masses,
            pressures,
            surface_reached
        ) = solve_tov(
            P_c=P_c,
            K=K,
            gamma=gamma,
            dr=dr,
            r_max=r_max
        )

        # The physical radius is the final radius
        # if the surface was reached.
        radius = radii[-1]

        # Final enclosed mass
        mass = masses[-1]

        results.append(
            {
                "central_pressure": P_c,
                "radius": radius,
                "mass": mass,
                "radii": radii,
                "masses": masses,
                "pressures": pressures,
                "surface_reached": surface_reached
            }
        )

    return results


# =========================================================
# Test / baseline run
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # Baseline EOS parameters
    # -----------------------------------------------------

    K = 0.01
    gamma = 2.0

    # -----------------------------------------------------
    # Central pressure sequence
    # -----------------------------------------------------

    central_pressures = np.logspace(
        32,
        37,
        20
    )

    # -----------------------------------------------------
    # Generate sequence
    # -----------------------------------------------------

    results = generate_mass_radius_sequence(
        central_pressures=central_pressures,
        K=K,
        gamma=gamma,
        dr=10.0,
        r_max=30000.0
    )

    # -----------------------------------------------------
    # Solar mass
    # -----------------------------------------------------

    solar_mass = 1.98847e30

    # -----------------------------------------------------
    # Print results
    # -----------------------------------------------------

    print("\nMass-Radius Sequence")
    print("=" * 90)

    print(
        f"{'Pc (Pa)':>15} "
        f"{'R (km)':>12} "
        f"{'M (Msun)':>12} "
        f"{'Surface':>12}"
    )

    print("-" * 90)

    for result in results:

        P_c = result["central_pressure"]

        R = result["radius"]

        M = result["mass"]

        surface = (
            "YES"
            if result["surface_reached"]
            else "NO - r_max"
        )

        print(
            f"{P_c:>15.3e} "
            f"{R / 1000:>12.3f} "
            f"{M / solar_mass:>12.6f} "
            f"{surface:>12}"
        )