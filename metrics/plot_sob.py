# Plotting
from matplotlib import pyplot as plt
import numpy as np


#test = "."
#test = "./AO_th_0.9"
test = "./AO_th_0.5"
removed_nerf_pts = np.load(f"{test}/removed_nerf_pts.npy")
removed_3dgs_pts = np.load(f"{test}/removed_3dgs_pts.npy")

parts = np.arange(0, len(removed_3dgs_pts))  # Part numbers, just for plotting

# Calculate a moving average for smoothing the curves
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Determine the window size for smoothing, e.g., averaging over 10 points
window_size = 25

# Calculate moving averages for both datasets
smoothed_3dgs = moving_average(removed_3dgs_pts, window_size)
smoothed_nerf = moving_average(removed_nerf_pts, window_size)

# Adjusted parts for the smoothed data due to 'valid' mode convolution
smoothed_parts = parts[:len(smoothed_3dgs)]

# Plot configuration
plt.figure(figsize=(10, 6))
plt.plot(parts, removed_3dgs_pts, '-o', label='3DGS Removed Points', color='blue', alpha=0.2, markersize=3)
plt.plot(parts, removed_nerf_pts, '-o', label='NeRF Removed Points', color='red', alpha=0.2, markersize=3)
plt.plot(smoothed_parts, smoothed_3dgs, label='3DGS Moving Average', color='navy', linewidth=3)
plt.plot(smoothed_parts, smoothed_nerf, label='NeRF Moving Average', color='darkred', linewidth=3)

plt.xlabel('Part Number')
plt.ylabel('Number of Points Removed')
plt.title('Comparison of Outliers Removed by 3DGS and NeRF')
plt.legend()
plt.grid(True)

# Set sensible limits for the y-axis to focus on the bulk of the data, exclude extremes
upper_limit = np.percentile(np.concatenate((removed_3dgs_pts, removed_nerf_pts)), 99)  # 99th percentile
plt.ylim([0, upper_limit])

# Show the plot
plt.show()