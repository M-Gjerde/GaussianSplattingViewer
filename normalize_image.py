
import cv2
import numpy as np

def apply_colormap_on_disparity(disparity_image, color_map='jet'):
    # Normalize the disparity image to range 0-255
    disparity_normalized = cv2.normalize(disparity_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply the specified colormap
    if color_map == 'jet':
        color_mapped_image = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    elif color_map == 'heat':
        color_mapped_image = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_HOT)
    else:
        raise ValueError("Unsupported color_map. Use 'jet' or 'heat'.")

    return color_mapped_image

# Example usage:
disparity_path_gs = "C:\\Users\\mgjer\\PycharmProjects\\GaussianSplattingViewer\\out\\scene_0\\depth\\0.png"
disparity_path_nerf = "C:\\Users\\mgjer\\Downloads\\stereo_dataset_v1_part1\\0000\\Q\\baseline_0.50\\disparity\\IMG_20220818_174109.png"

# Load your 16-bit disparity image (replace 'path_to_your_disparity_image' with the actual path)
disparity_image_gs = cv2.imread(disparity_path_gs, cv2.IMREAD_UNCHANGED)
disparity_image_nerf = cv2.imread(disparity_path_nerf, cv2.IMREAD_UNCHANGED)

# Apply colormap (either 'jet' or 'heat')
color_mapped_disparity_gs = apply_colormap_on_disparity(disparity_image_gs, 'jet')
color_mapped_disparity_nerf = apply_colormap_on_disparity(disparity_image_nerf, 'jet')

# Display the result
# Create named windows with the ability to resize
cv2.namedWindow('Colormap on GS-Disparity', cv2.WINDOW_NORMAL)
cv2.namedWindow('Colormap on NeRF-Disparity', cv2.WINDOW_NORMAL)

cv2.resizeWindow('Colormap on GS-Disparity', 1160, 522)
cv2.resizeWindow('Colormap on NeRF-Disparity', 1160, 522)

cv2.imshow('Colormap on GS-Disparity', color_mapped_disparity_gs)
cv2.imshow('Colormap on NeRF-Disparity', color_mapped_disparity_nerf)

cv2.imwrite('Colormap on GS-Disparity.png', color_mapped_disparity_gs)
cv2.imwrite('Colormap on NeRF-Disparity.png', color_mapped_disparity_nerf)
cv2.waitKey(0)
cv2.destroyAllWindows()
