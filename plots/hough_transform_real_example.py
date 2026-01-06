import os
import math
import time
import array
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection



# Input points: (x, y, z)
points = {
"ALPIDE_0": (2.4057599999999995,-5.482500000000002,0.0),
"ALPIDE_1": (2.349326068753353,-4.909929243346539,20.000000000000014),
"ALPIDE_2": (2.2963656421789946,-4.311975590575699,40.000000000000014),
"ALPIDE_3": (2.2303777830521505,-3.7226842198429324,60.000000000000014),
"ALPIDE_4": (2.1791417249644915,-3.1225710136659472,80.00000000000001),
}

# Colors for each label
colors = {
    "ALPIDE_0": "blue",
    "ALPIDE_1": "orange",
    "ALPIDE_2": "green",
    "ALPIDE_3": "red",
    "ALPIDE_4": "purple",
}

# Angles from 0 to pi
theta = np.linspace(0, np.pi, 500)

# Bigger figure so pads are square
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

# Plot (z, x) waves
for label, (x, y, z) in points.items():
    rho_x = x * np.sin(theta) + z * np.cos(theta)
    ax1.plot(theta, rho_x, label=label, color=colors[label])

ax1.set_xlabel(r"$\theta_x$ [rad]", fontsize=18)
ax1.set_ylabel(r"$\rho_x$ [mm]", fontsize=18)
ax1.set_title(r"$\rho_x = x\sin\theta_x + z\cos\theta_x$", fontsize=20)
ax1.set_xlim(0, np.pi)
ax1.grid(True)
ax1.legend(fontsize=16)
ax1.tick_params(axis='both', labelsize=16) # Make tick labels bigger

# Plot (z, y) waves
for label, (x, y, z) in points.items():
    rho_y = y * np.sin(theta) + z * np.cos(theta)
    ax2.plot(theta, rho_y, label=label, color=colors[label])

ax2.set_xlabel(r"$\theta_y$ [rad]", fontsize=18)
ax2.set_ylabel(r"$\rho_y$ [mm]", fontsize=18)
ax2.set_title(r"$\rho_y = y\sin\theta_y + z\cos\theta_y$", fontsize=20)
ax2.set_xlim(0, np.pi)
ax2.grid(True)
ax2.legend(fontsize=16)
ax2.tick_params(axis='both', labelsize=16) # Make tick labels bigger

# Adjust spacing
# plt.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.1, wspace=0.25)
# plt.show()
plt.savefig("hough_transform_real_example.pdf")





##########################################################
##########################################################
##########################################################
##########################################################

# -----------------------
# Input points (mm)
# -----------------------
# Plane z-positions (mm)
plane_z = [0, 20, 40, 60, 80]

# Plane size (mm)
Lx = 1024 * 0.02924   # ≈ 29.93 mm
Ly = 512  * 0.02688   # ≈ 13.76 mm

# -----------------------
# Figure
# -----------------------
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# -----------------------
# Draw sensor planes (centered on z-axis)
# -----------------------
for z in plane_z:
    x = [-Lx/2,  Lx/2,  Lx/2, -Lx/2]
    y = [-Ly/2, -Ly/2,  Ly/2,  Ly/2]
    zc = [z, z, z, z]

    verts = [list(zip(x, y, zc))]
    plane = Poly3DCollection(
        verts,
        facecolors='green',
        edgecolors='g',
        linewidths=0.5,
        alpha=0.20
    )
    ax.add_collection3d(plane)

# -----------------------
# Plot hit points
# -----------------------
xs, ys, zs = zip(*points.values())

###########
###########
###########
xs = np.array(xs)
ys = np.array(ys)
zs = np.array(zs)

xmin, xmax = xs.min(), xs.max()
ymin, ymax = ys.min(), ys.max()
zmin, zmax = zs.min(), zs.max()

# Include plane extents
xmin = min(xmin, -Lx/2)
xmax = max(xmax,  Lx/2)
ymin = min(ymin, -Ly/2)
ymax = max(ymax,  Ly/2)
zmin = min(zmin, min(plane_z))
zmax = max(zmax, max(plane_z))

# Compute cubic bounding box
cx = 0.5 * (xmin + xmax)
cy = 0.5 * (ymin + ymax)
cz = 0.5 * (zmin + zmax)

max_range = max(
    xmax - xmin,
    ymax - ymin,
    zmax - zmin
) / 2

ax.set_xlim(cx - max_range, cx + max_range)
ax.set_ylim(cy - max_range, cy + max_range)
ax.set_zlim(cz - max_range, cz + max_range)
# Force equal axis scaling
ax.set_box_aspect([1, 1, 1])
###########
###########
###########


# ax.scatter(xs, ys, zs, color='red', s=40, label='Hits')
ax.scatter(xs[0], ys[0], zs[0], color=colors["ALPIDE_0"], s=40, label='ALPIDE_0')
ax.scatter(xs[1], ys[1], zs[1], color=colors["ALPIDE_1"], s=40, label='ALPIDE_1')
ax.scatter(xs[2], ys[2], zs[2], color=colors["ALPIDE_2"], s=40, label='ALPIDE_2')
ax.scatter(xs[3], ys[3], zs[3], color=colors["ALPIDE_3"], s=40, label='ALPIDE_3')
ax.scatter(xs[4], ys[4], zs[4], color=colors["ALPIDE_4"], s=40, label='ALPIDE_4')
ax.plot(xs, ys, zs, color='red', linestyle='--', linewidth=1)

# -----------------------
# Labels & formatting
# -----------------------
ax.set_xlabel('X [mm]')
ax.set_ylabel('Y [mm]')
ax.set_zlabel('Z [mm]')

ax.legend()

# ax.set_xlim(-20, 20)
# ax.set_ylim(-10, 10)
# ax.set_zlim(-5, 85)

# Equal aspect ratio
ax.set_box_aspect([40, 20, 90])

plt.tight_layout()
plt.show()

