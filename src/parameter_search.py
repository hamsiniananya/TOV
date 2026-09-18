"""
TOV Parameter Study + CSV Export

Runs the TOV solver for different values of K and gamma
and exports the results in CSV format for plotting in Origin.
"""

import numpy as np
import csv
from pathlib import Path

from .solver import generate_mass_radius_sequence


# =========================================================
# PARAMETERS
# =========================================================

K_values = [
    0.035,
    0.0375,
    0.040,
    0.0425,
    0.045
]

gamma_values = [
    1.99,
    1.995,
    2.00,
    2.005,
    2.01
]

central_pressures = np.logspace(
    33,
    36,
    40
)

dr = 10.0
r_max = 100000.0

solar_mass = 1.98847e30


# =========================================================
# CREATE RESULTS DIRECTORY
# =========================================================

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)


# =========================================================
# STORAGE
# =========================================================

parameter_results = []

mass_radius_data = []

pressure_mass_data = []

tov_profiles = []


# =========================================================
# RUN PARAMETER STUDY
# =========================================================

for gamma in gamma_values:

    for K in K_values:

        print(
            f"Running K = {K:.4e}, "
            f"Gamma = {gamma:.3f}"
        )

        results = generate_mass_radius_sequence(
            central_pressures=central_pressures,
            K=K,
            gamma=gamma,
            dr=dr,
            r_max=r_max
        )

        # -------------------------------------------------
        # Store every model in the mass-radius dataset
        # -------------------------------------------------

        for result in results:

            if not result["surface_reached"]:
                continue

            P_c = result["central_pressure"]

            R = result["radius"] / 1000.0

            M = result["mass"] / solar_mass

            mass_radius_data.append(
                {
                    "K": K,
                    "Gamma": gamma,
                    "Central_Pressure_Pa": P_c,
                    "Radius_km": R,
                    "Mass_Msun": M
                }
            )

        # -------------------------------------------------
        # Find maximum mass
        # -------------------------------------------------

        valid_results = [
            result
            for result in results
            if result["surface_reached"]
        ]

        if len(valid_results) == 0:

            parameter_results.append(
                {
                    "K": K,
                    "Gamma": gamma,
                    "Max_Mass_Msun": np.nan,
                    "Radius_at_Max_Mass_km": np.nan,
                    "Pc_at_Max_Mass_Pa": np.nan,
                    "Surface_Reached": "NO"
                }
            )

            continue

        masses = np.array([
            result["mass"] / solar_mass
            for result in valid_results
        ])

        max_index = np.argmax(masses)

        max_result = valid_results[max_index]

        max_mass = (
            max_result["mass"] / solar_mass
        )

        max_radius = (
            max_result["radius"] / 1000.0
        )

        max_pressure = (
            max_result["central_pressure"]
        )

        parameter_results.append(
            {
                "K": K,
                "Gamma": gamma,
                "Max_Mass_Msun": max_mass,
                "Radius_at_Max_Mass_km": max_radius,
                "Pc_at_Max_Mass_Pa": max_pressure,
                "Surface_Reached": "YES"
            }
        )

        # -------------------------------------------------
        # Store TOV profiles for the maximum-mass star
        # -------------------------------------------------

        radii = max_result["radii"]

        masses_profile = max_result["masses"]

        pressures_profile = max_result["pressures"]

        for r, m, P in zip(
            radii,
            masses_profile,
            pressures_profile
        ):

            tov_profiles.append(
                {
                    "K": K,
                    "Gamma": gamma,
                    "Radius_km": r / 1000.0,
                    "Mass_Msun": m / solar_mass,
                    "Pressure_Pa": P
                }
            )


# =========================================================
# CSV WRITER FUNCTION
# =========================================================

def write_csv(filename, data):

    if not data:
        print(f"No data to write for {filename}")
        return

    filepath = results_dir / filename

    fieldnames = list(data[0].keys())

    with open(
        filepath,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(data)

    print(f"Created: {filepath}")


# =========================================================
# EXPORT CSV FILES
# =========================================================

write_csv(
    "parameter_study.csv",
    parameter_results
)

write_csv(
    "mass_radius_all.csv",
    mass_radius_data
)

write_csv(
    "tov_profiles_max_mass.csv",
    tov_profiles
)


# =========================================================
# PRINT SUMMARY
# =========================================================

print("\n")
print("=" * 90)
print("TOV PARAMETER STUDY")
print("=" * 90)

print(
    f"{'K':>10} "
    f"{'Gamma':>8} "
    f"{'Max Mass':>12} "
    f"{'Radius':>12} "
    f"{'Pc':>18} "
    f"{'Surface':>10}"
)

print("-" * 90)

for result in parameter_results:

    print(
        f"{result['K']:>10.4f} "
        f"{result['Gamma']:>8.3f} "
        f"{result['Max_Mass_Msun']:>12.4f} "
        f"{result['Radius_at_Max_Mass_km']:>12.3f} "
        f"{result['Pc_at_Max_Mass_Pa']:>18.3e} "
        f"{result['Surface_Reached']:>10}"
    )


print("\nCSV files created in:")
print(results_dir.resolve())