import numpy as np
import open3d as o3d
import cv2  # For image operations
from tqdm import tqdm

from scipy.spatial import cKDTree

#from tqdm import tqdm


def median_filter_float32(image, kernel_size):
    """
    Apply a median filter to a float32 image.

    Parameters:
    - image: numpy.ndarray, the float32 image to filter.
    - kernel_size: int, the size of the median filter kernel; must be an odd number.

    Returns:
    - filtered_image: numpy.ndarray, the filtered image.
    """
    # Ensure kernel_size is odd to have a central pixel
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd number.")

    # Pad the image to handle edges
    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='edge')

    # Prepare an empty image to store the filtered result
    filtered_image = np.zeros_like(image)

    # Get image dimensions
    rows, cols = image.shape

    # Within the median_filter_float32 function, replace the for loop with:
    for i in tqdm(range(rows), desc="Applying Median Filter"):
        for j in range(cols):
            # Extract the window for the current pixel
            window = padded_image[i:i + kernel_size, j:j + kernel_size]
            # Compute the median and assign it to the filtered image
            filtered_image[i, j] = np.median(window)

    return filtered_image


# Reworking the depth_sharpening function to handle indexing appropriately
def depth_sharpening(disparity_map, threshold=3):
    # Calculate the Sobel edge filter response in both horizontal and vertical directions
    sobelx = cv2.Sobel(disparity_map, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(disparity_map, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Identify flying pixels: pixels where the Sobel magnitude is greater than the threshold
    flying_pixels = sobel_mag > threshold

    # Create a mask for non-flying pixels
    non_flying_mask = ~flying_pixels

    # Find the nearest non-flying pixel for each flying pixel using distance transform
    _, labels = cv2.distanceTransformWithLabels(non_flying_mask.astype(np.uint8), cv2.DIST_L2, 5)

    # Correct the disparities of flying pixels by assigning the disparity of the nearest non-flying pixel
    # Use the labels to index into disparity_map
    height, width = disparity_map.shape
    corrected_disparity = disparity_map.copy()
    for y in tqdm(range(height)):
        for x in range(width):
            if flying_pixels[y, x]:
                # Find label of the nearest non-flying pixel
                label = labels[y, x]
                # Find the position of the nearest non-flying pixel with the same label
                ny, nx = np.where(labels == label)
                if ny.size > 0 and nx.size > 0:
                    # Use the first found position (closest non-flying pixel)
                    corrected_disparity[y, x] = disparity_map[ny[0], nx[0]]

    return corrected_disparity


def improved_depth_sharpening(disparity_map, threshold=6):
    # Calculate the Sobel edge filter response
    sobelx = cv2.Sobel(disparity_map, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(disparity_map, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Identify flying pixels: pixels where the Sobel magnitude is greater than the threshold
    flying_pixels = sobel_mag > threshold

    # Find indices of non-flying pixels
    non_flying_indices = np.nonzero(~flying_pixels)
    non_flying_values = disparity_map[non_flying_indices]

    # For each flying pixel, find the nearest non-flying pixel and its disparity value
    flying_pixel_indices = np.transpose(np.nonzero(flying_pixels))
    tree = cKDTree(np.column_stack(non_flying_indices))
    _, nearest_indices = tree.query(flying_pixel_indices)
    nearest_disparities = non_flying_values[nearest_indices]

    # Correct the disparities of flying pixels
    corrected_disparity_map = disparity_map.copy()
    corrected_disparity_map[np.nonzero(flying_pixels)] = nearest_disparities

    return corrected_disparity_map


def optimized_depth_sharpening(disparity_map, threshold=6):
    # Calculate the Sobel edge filter response
    sobelx = cv2.Sobel(disparity_map, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(disparity_map, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Identify flying pixels: pixels where the Sobel magnitude is greater than the threshold
    flying_pixels = sobel_mag > threshold

    # Create a mask for non-flying pixels and perform distance transform
    non_flying_mask = ~flying_pixels
    dist_transform, labels = cv2.distanceTransformWithLabels(non_flying_mask.astype(np.uint8), cv2.DIST_L1, 5)

    # Labels are 1-indexed, so we offset by -1 to make them 0-indexed for array access
    labels -= 1

    # Get all unique labels of non-flying pixels
    unique_labels = np.unique(labels * non_flying_mask)

    # Map each label to its disparity using the disparity map
    label_to_disparity = disparity_map.ravel()[unique_labels]

    # Replace the flying pixels' disparities with the mapped disparities of the nearest non-flying pixels
    corrected_disparity_map = disparity_map.copy()
    corrected_disparity_map[flying_pixels] = label_to_disparity[labels[flying_pixels]]

    return corrected_disparity_map
def disparity_to_depth(disparity_image, focal_length, baseline):
    # Convert disparity image to depth map
    disparity_image[disparity_image == 0] = 0.1  # Avoid division by zero
    depth_map = (focal_length * baseline) / disparity_image
    return depth_map

def load_image(file_path, color=False):
    """
    Load an image from file. Can load in grayscale or color.
    """
    if color:
        # Load color image
        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        # Load disparity image (assuming 16-bit PNG)
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        image =  image / 65535  # Adjust scale depending on your disparity computation method
        image *= 1160
    return image

def create_colored_point_cloud(depth_map, color_image, intrinsic_matrix):
    """
    Generate and view colored point cloud from depth map and color image.
    """
    # Create Open3D depth image from depth map
    depth_image = o3d.geometry.Image(depth_map.astype(np.float32))
    # Create Open3D color image from color image
    color_image_o3d = o3d.geometry.Image(color_image.astype(np.uint8))
    # Create Open3D intrinsic object
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=depth_map.shape[1], height=depth_map.shape[0],
                             fx=intrinsic_matrix[0, 0], fy=intrinsic_matrix[1, 1],
                             cx=intrinsic_matrix[0, 2], cy=intrinsic_matrix[1, 2])
    # Generate RGBD image from depth and color images
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_o3d, depth_image,
                                                                    depth_scale=1.0, depth_trunc=1000.0,
                                                                    convert_rgb_to_intensity=False)
    # Generate point cloud
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    # Transform point cloud to the original camera location
    point_cloud.transform([[1, 0, 0, 0],
                           [0, -1, 0, 0],  # Flip the point cloud for a correct view
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])
    # Visualize the point cloud
    # Create a Visualizer object
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Set the background color to dark grey
    vis.get_render_option().background_color = np.array([0.2, 0.2, 0.2])

    # Add the point cloud to the Visualizer
    vis.add_geometry(point_cloud)

    # Run the Visualizer
    vis.run()

    # Destroy the Visualizer window
    vis.destroy_window()

if __name__ == "__main__":
    disparity_file_path = "C:\\Users\\mgjer\\PycharmProjects\\GaussianSplattingViewer\\out_baseline_05\\scene_0000\\depth\\37.png"
    disparity_file_path_nerf = "C:\\Users\\mgjer\\Downloads\\stereo_dataset_v1_part1\\0000\\Q\\baseline_0.50\\disparity\\IMG_20220818_174000.png"
    left_file_path = "C:\\Users\\mgjer\\PycharmProjects\\GaussianSplattingViewer\\out_baseline_05\\scene_0000\\left\\37.png"
    left_file_path_nerf = "C:\\Users\\mgjer\\Downloads\\stereo_dataset_v1_part1\\0000\\Q\\baseline_0.50\\left\\IMG_20220818_174000.jpg"

    disparity_image = load_image(disparity_file_path)
    disparity_image_nerf = load_image(disparity_file_path_nerf)


    corrected_disparity_map = optimized_depth_sharpening(disparity_image)
    corrected_disparity_map = improved_depth_sharpening(disparity_image)
    corrected_disparity_image_nerf = improved_depth_sharpening(disparity_image_nerf)


    fx = 3439.3083700227126 / 4
    fy = 3445.0110843463276 / 4
    cx = 2320 / 4
    cy = 1044 / 4
    baseline =  0.5
    intrinsic_matrix = np.array([[fx, 0, cx ],  # fx, 0, cx
                                 [0, fy, cy ],  # 0, fy, cy
                                 [0, 0, 1]])  # Intrinsic matrix of the camera

    color_image = load_image(left_file_path, color=True)
    depth_map = disparity_to_depth(disparity_image, fx, baseline)
    create_colored_point_cloud(depth_map, color_image, intrinsic_matrix)

    color_image = load_image(left_file_path, color=True)
    depth_map = disparity_to_depth(corrected_disparity_map, fx, baseline)
    create_colored_point_cloud(depth_map, color_image, intrinsic_matrix)

    color_image = load_image(left_file_path, color=True)
    depth_map = disparity_to_depth(disparity_image_nerf, fx, baseline)
    create_colored_point_cloud(depth_map, color_image, intrinsic_matrix)

    color_image = load_image(left_file_path, color=True)
    depth_map = disparity_to_depth(corrected_disparity_image_nerf, fx, baseline)
    create_colored_point_cloud(depth_map, color_image, intrinsic_matrix)

    cv2.imshow("disparity_image", (disparity_image).astype(np.uint8))
    cv2.imshow("corrected_disparity_map", (corrected_disparity_map).astype(np.uint8))

    cv2.imshow("disparity_image_nerf", (disparity_image_nerf).astype(np.uint8))
    cv2.imshow("corrected_disparity_map_nerf", (corrected_disparity_image_nerf).astype(np.uint8))

    cv2.waitKey(0)
    cv2.destroyAllWindows()