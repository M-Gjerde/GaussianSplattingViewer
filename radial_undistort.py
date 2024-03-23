import cv2
import numpy as np


def undistort_radially(image, k1):
    """Undistorts the image radially based on the provided k1 parameter."""
    h, w = image.shape[:2]
    fx = fy = w // 2
    cx, cy = w // 2, h // 2

    camera_matrix = np.array([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]], dtype=np.float32)

    dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float32)

    undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)
    return undistorted_image


def on_trackbar_change(val):
    """Callback function for trackbar event."""
    k1 = (val - 500) / 500  # Convert trackbar value to k1 range
    undistorted_image = undistort_radially(image, k1)
    cv2.imshow("Undistorted Image", undistorted_image)


# Load the image
image_path = 'C:\\Users\\mgjer\\PycharmProjects\\GaussianSplattingViewer\\out_baseline_05\\scene_0000\\left\\0.png'
image = cv2.imread(image_path)

# Create a window and a trackbar
cv2.namedWindow("Undistorted Image")
cv2.createTrackbar("K1", "Undistorted Image", 500, 1000, on_trackbar_change)

# Display the original image before any undistortion
cv2.imshow("Undistorted Image", image)

# Wait until a key is pressed and destroy all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
