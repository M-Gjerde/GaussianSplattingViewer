import numpy as np
from matplotlib import pyplot as plt

# Function to calculate the median of every chunk of size 'chunk_size'
def chunked_median(data, chunk_size):
    # Split data into chunks of 'chunk_size', ignoring the remainder
    chunks = [data[i:i + chunk_size] for i in range(0, len(data) - chunk_size + 1, chunk_size)]
    # Calculate the median of each chunk
    medians = [np.median(chunk) for chunk in chunks]
    return np.array(medians)

# Load the data
blur_scores = np.load('test/blur_scores.npy')

# Set chunk size
chunk_size = 5

# Calculate medians for each column
median_pts = np.array([chunked_median(blur_scores[:, i], chunk_size) for i in range(blur_scores.shape[1])]).T

# Create an index array for the median values
median_index_array = np.arange(len(median_pts)) * chunk_size

# Plotting the medians
plt.figure(figsize=(10, 6))
plt.plot(median_index_array, median_pts[:, 0], label='Input views', color='red', linewidth=2, linestyle='-', marker='o')
plt.plot(median_index_array, median_pts[:, 1], label='NeRF-supervised', color='green', linewidth=2, linestyle='--', marker='x')
plt.plot(median_index_array, median_pts[:, 2], label='3DGS-supervised', color='blue', linewidth=2, linestyle='-.', marker='^')

plt.xlabel('Index (Median of Each 50 Points)')
plt.ylabel('Median Value')
plt.title('Plot of Median Values for Each 50-Point Chunk')
plt.legend()
plt.show()