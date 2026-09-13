"""
Generate a mass-radius relation by solving
the TOV equations for multiple central pressures.
"""

import numpy as np
import matplotlib.pyplot as plt

from src.solver import solve_tov


# -------------------------
# Central pressure values
# -------------------------

central_pressures = np.logspace(
    -4,
    -2,
    100
)


# Lists to store results
radii = []
masses = []


# -------------------------
# Solve for each pressure
# -------------------------

for P_c in central_pressures:

    try:

        r, m, P = solve_tov(
            P_c=P_c,
            dr=0.01,
            r_max=20.0
        )

        # Surface values
        star_radius = r[-1]
        star_mass = m[-1]

        radii.append(star_radius)
        masses.append(star_mass)

        print(
            f"P_c = {P_c:.6e}   "
            f"R = {star_radius:.4f}   "
            f"M = {star_mass:.4f}"
        )

    except Exception as error:

        print(
            f"P_c = {P_c:.6e} failed: {error}"
        )


# Convert lists to NumPy arrays
radii = np.array(radii)
masses = np.array(masses)


# -------------------------
# Find maximum mass
# -------------------------

max_index = np.argmax(masses)

max_mass = masses[max_index]
max_radius = radii[max_index]
max_pressure = central_pressures[max_index]


print()
print("Maximum Mass Model")
print("------------------")
print(f"Central pressure : {max_pressure:.6e}")
print(f"Maximum mass     : {max_mass:.4f}")
print(f"Radius           : {max_radius:.4f}")


# -------------------------
# Mass-radius plot
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    radii,
    masses,
    marker="o",
    markersize=4
)

plt.scatter(
    max_radius,
    max_mass,
    s=80,
    label=f"Maximum mass = {max_mass:.3f}"
)

plt.xlabel("Radius")
plt.ylabel("Mass")
plt.title("Neutron Star Mass-Radius Relation")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()