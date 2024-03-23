# Plotting
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, median_absolute_error

# Load the data from the provided file
blur_scores = np.load('./blur_calculation/blur_scores.npy')

# Creating an index array for the x-values
index_array = np.arange(len(blur_scores))

# Calculate a moving average for smoothing the curves
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Determine the window size for smoothing, e.g., averaging over 10 points
window_size = 25  # Adjust the window size as needed

# Calculate moving averages for each column in blur_scores
smoothed_pts = np.array([moving_average(blur_scores[:, i], window_size) for i in range(blur_scores.shape[1])]).T

# Adjusted index array for the smoothed data due to 'valid' mode convolution
smoothed_index_array = index_array[:len(smoothed_pts)]

# Plotting the original data
plt.figure(figsize=(10, 6))
#plt.plot(index_array, blur_scores[:, 0], label='Input views', color='red', linewidth=2, linestyle='-', marker='o', alpha=0.7)
#plt.plot(index_array, blur_scores[:, 1], label='NeRF-supervised', color='green', linewidth=2, linestyle='--', marker='x', alpha=0.7)
#plt.plot(index_array, blur_scores[:, 2], label='3DGS-supervised', color='blue', linewidth=2, linestyle='-.', marker='^', alpha=0.7)

# Plotting the smoothed data
plt.plot(smoothed_index_array, smoothed_pts[:, 0], label='Input views (Smoothed)', color='darkred', linewidth=2)
plt.plot(smoothed_index_array, smoothed_pts[:, 1], label='NeRF-supervised (Smoothed)', color='darkgreen', linewidth=2)
plt.plot(smoothed_index_array, smoothed_pts[:, 2], label='3DGS-supervised (Smoothed)', color='darkblue', linewidth=2)

plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Smooth Line Plot for Each Column')
plt.legend()
plt.show()


# Calculate MAE
mae_nerf = mean_absolute_error(smoothed_pts[:, 0], smoothed_pts[:, 1]) * 100
mae_3dgs = mean_absolute_error(smoothed_pts[:, 0], smoothed_pts[:, 2]) * 100

# Calculate MAD
mad_nerf = np.median(np.abs(smoothed_pts[:, 0] - np.median(smoothed_pts[:, 1]))) * 100
mad_3dgs = np.median(np.abs(smoothed_pts[:, 0] - np.median(smoothed_pts[:, 2]))) * 100

print(f"MAE for NeRF-supervised: {mae_nerf}")
print(f"MAE for 3DGS-supervised: {mae_3dgs}")
print(f"MAD for NeRF-supervised: {mad_nerf}")
print(f"MAD for 3DGS-supervised: {mad_3dgs}")