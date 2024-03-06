import cv2
import numpy as np


# Function to normalize and display a 16-bit disparity PNG image
def normalize_and_display_image(input_image_path, output_image_path):
    # Read the 16-bit PNG image
    img = cv2.imread(input_image_path, cv2.IMREAD_UNCHANGED)

    # Normalize the image to the range 0-255
    normalized_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Write the normalized image to the output path
    cv2.imwrite(output_image_path, normalized_img)

    # Display the normalized image
    cv2.imshow('Normalized Disparity Image', normalized_img)
    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()


normalize_and_display_image("output/scene_0017/depth/9.png", "depth_normalized.png")