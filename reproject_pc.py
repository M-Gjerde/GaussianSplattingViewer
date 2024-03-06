import numpy as np
import open3d as o3d
import cv2  # For image operations
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
        image =  image / 65535.0  # Adjust scale depending on your disparity computation method
        #image = median_filter_float32(image, 5) * 255
        image *= 255
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
    o3d.visualization.draw_geometries([point_cloud])

if __name__ == "__main__":
    disparity_file_path = "C:\\Users\\mgjer\\PycharmProjects\\GaussianSplattingViewer\\out\\scene_0000\\depth\\10.png"
    color_file_path = "C:\\Users\\mgjer\\PycharmProjects\\GaussianSplattingViewer\\out\\scene_0000\\right\\10.png"
    focal_length = 3437.474032508210  # Example focal length in pixels
    baseline =  0.193001 * 5  # Example baseline in meters
    intrinsic_matrix = np.array([[3437.474032508210 / 3, 0, 1160 / 2],  # fx, 0, cx
                                 [0, 3443.9159462085058 / 3, 552 / 2],  # 0, fy, cy
                                 [0, 0, 1]])  # Intrinsic matrix of the camera

    disparity_image = load_image(disparity_file_path)
    color_image = load_image(color_file_path, color=True)
    depth_map = disparity_to_depth(disparity_image, focal_length, baseline)
    create_colored_point_cloud(depth_map, color_image, intrinsic_matrix)

