import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sphere parameters
radius = 4

# Create a meshgrid for the sphere
theta, phi = np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100)
theta, phi = np.meshgrid(theta, phi)
x = radius * np.sin(phi) * np.cos(theta)
y = radius * np.sin(phi) * np.sin(theta)
z = radius * np.cos(phi)

# Plot the sphere
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, color='green', alpha=0.3)

# Draw a line on the sphere - for simplicity, let's draw the equator
theta_line = np.linspace(0, 2 * np.pi, 100)

for phi_line in np.linspace(0, np.pi/(4), 10):
    x_line = radius * np.cos(theta_line) * np.cos(phi_line)
    y_line = radius * np.sin(phi_line) * np.ones_like(theta_line)
    z_line = radius * np.sin(theta_line) * np.cos(phi_line)
    ax.plot(x_line, y_line, z_line, color='red', linewidth=1.5)
    for i in range(0, len(x_line), 100):
        ax.quiver(0, 0, 0, x_line[i], y_line[i], z_line[i], arrow_length_ratio=0.05)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
len_axes = 6


# Set limits on the axes
ax.set_xlim([-radius, radius])
ax.set_ylim([-radius, radius])
ax.set_zlim([-radius, radius])


# X-axis
ax.quiver(0, 0, 0, len_axes, 0, 0, color='r', arrow_length_ratio=0.05)
# Y-axis
ax.quiver(0, 0, 0, 0, len_axes, 0, color='g', arrow_length_ratio=0.05)
# Z-axis
ax.quiver(0, 0, 0, 0, 0, len_axes, color='b', arrow_length_ratio=0.05)

# Adjust the view: 90 degrees around the x-axis to make the y-axis point down
ax.view_init(90, -90)

plt.show()
