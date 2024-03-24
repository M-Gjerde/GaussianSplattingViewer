# Plotting
from matplotlib import pyplot as plt
import numpy as np


#test = "."
test = "./AO_th_0.75"
test = "./AO_th_none"
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
smoothed_parts = parts[:int(len(smoothed_3dgs))]

# Plot configuration
plt.figure(figsize=(10, 6))
plt.plot(parts[:len(smoothed_3dgs)], removed_3dgs_pts[:len(smoothed_3dgs)], '-o', label='3DGS Removed Points', color='blue', alpha=0.2, markersize=3)
plt.plot(parts[:len(smoothed_3dgs)], removed_nerf_pts[:len(smoothed_3dgs)], '-o', label='NeRF Removed Points', color='red', alpha=0.2, markersize=3)
plt.plot(smoothed_parts, smoothed_3dgs, label='3DGS Moving Average n=25', color='navy', linewidth=3)
plt.plot(smoothed_parts, smoothed_nerf, label='NeRF Moving Average n=25', color='darkred', linewidth=3)

# Adjust subplot parameters to reduce whitespace
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.01)

# To reduce whitespace further, especially around the saved figure, you can use tight_layout
plt.tight_layout()


#plt.xlabel('Scene', fontsize='large')
#plt.ylabel('Number of Points Removed', fontsize='large')
#plt.title('Comparison of Outliers Removed by 3DGS and NeRF', fontsize='large')
plt.grid(False)
plt.legend(prop={'size': 22}, loc='upper left')  # Adjust the size as needed
plt.legend(prop={'size': 18})  # Adjust the size as needed

# Set sensible limits for the y-axis to focus on the bulk of the data, exclude extremes
upper_limit = np.percentile(np.concatenate((removed_3dgs_pts, removed_nerf_pts)), 99)  # 99th percentile
plt.ylim([0, upper_limit])

# Show the plot
plt.show()