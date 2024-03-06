import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Given matrix (assuming this is the view matrix and the vectors are columns of the rotation matrix)
mat = np.array([
    [-0.970508, 0, 0.24107, 1.02022],
    [-0.0506011, 0.977723, -0.203711, 1.55078],
    [0.2357, 0.209902, 0.948887, 0.50755],
    [0, 0, 0, 1]
])

# Extracting right (R), up (U), and forward (F) vectors from the matrix
R = mat[:3, 0]
U = mat[:3, 1]
F = -mat[:3, 2] # Inverting forward because of OpenGL's coordinate system

# Plotting
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# Origin
origin = np.array([0, 0, 0])

# Plot vectors
ax.quiver(*origin, *R, color='r', length=1, label='Right')
ax.quiver(*origin, *U, color='g', length=1, label='Up')
ax.quiver(*origin, *F, color='b', length=1, label='Forward')

# Setting plot characteristics
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.view_init(elev=20., azim=30)

plt.show()