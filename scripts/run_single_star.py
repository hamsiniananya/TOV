"""
Run the TOV solver for one neutron star
and visualize its pressure and mass profiles.
"""

import matplotlib.pyplot as plt

from src.solver import solve_tov


# -------------------------
# Choose central pressure
# -------------------------

P_c = 0.001


# -------------------------
# Solve the TOV equations
# -------------------------

radii, masses, pressures = solve_tov(
    P_c=P_c,
    dr=0.01,
    r_max=20.0
)


# -------------------------
# Get final star properties
# -------------------------

star_radius = radii[-1]
star_mass = masses[-1]


print()
print("Neutron Star")
print("------------")
print(f"Central pressure : {P_c}")
print(f"Radius           : {star_radius:.4f}")
print(f"Mass             : {star_mass:.4f}")
print(f"Final pressure   : {pressures[-1]:.4e}")


# -------------------------
# Pressure vs radius
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(radii, pressures)

plt.xlabel("Radius")
plt.ylabel("Pressure")
plt.title("Pressure vs Radius")

plt.axvline(
    star_radius,
    linestyle="--",
    label=f"Surface R = {star_radius:.2f}"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


# -------------------------
# Mass vs radius
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(radii, masses)

plt.xlabel("Radius")
plt.ylabel("Enclosed Mass")
plt.title("Enclosed Mass vs Radius")

plt.axvline(
    star_radius,
    linestyle="--",
    label=f"Surface R = {star_radius:.2f}"
)

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()