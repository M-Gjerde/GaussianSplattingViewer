import cv2
import numpy as np

outputFolder = "out"
scene_folder = "scene_0000"
scale = 56.495

image_width = 1160

np.random.seed(2024)  # You can choose any seed value

def show():
    for pose_index in range(0, 100):
        imagePath = f"{outputFolder}/{scene_folder}/depth/{pose_index}.png"
        img = np.array(cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)).astype(np.float32)
        img = img / 65535
        img_8bit = (img * 255).astype(np.uint8)

        cv2.imshow("output", img_8bit)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

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
        image =  image / 65535.0  # Normalize float image to 0-1
        image *= 1160 # Scale to pixel disparities
    return image


def draw_random_circles(image, num_circles=5, radius=5):
    """Draws a specified number of circles at random locations on an image."""
    spots = []
    for _ in range(num_circles):
        # Ensure circles are drawn within the image bounds and account for the radius
        x = np.random.randint(radius, image.shape[1] - radius)
        y = np.random.randint(radius, image.shape[0] - radius)
        spots.append((x, y))
        cv2.circle(image, (x, y), radius, (0, 255, 0), 1)  # Green circle
    return spots

def draw_adjusted_circles(left_spots, right_image, disparity_image, radius=5):
    """Draws circles on the right image adjusted by the disparity from the left image spots."""
    for x, y in left_spots:
        # Assuming disparity_image is single-channel and normalized to the same scale as the pixel coordinates
        disparity = disparity_image[y, x]
        adjusted_x = int(x - disparity)  # Adjust the x-coordinate by the disparity
        # Ensure the adjusted coordinate is within the image bounds
        adjusted_x = min(max(adjusted_x, radius), right_image.shape[1] - radius)
        cv2.circle(right_image, (adjusted_x, y), radius, (0, 0, 255), 1)  # Red circle for distinction

if __name__ == "__main__":
    image_no = 0
    scene_folder = "scene_0000"
    disparity_file_path = f"C:\\Users\\mgjer\\PycharmProjects\\GaussianSplattingViewer\\out\\{scene_folder}\\depth\\{image_no}.png"
    left_file_path = f"C:\\Users\\mgjer\\PycharmProjects\\GaussianSplattingViewer\\out\\{scene_folder}\\left\\{image_no}.png"
    right_file_path = f"C:\\Users\\mgjer\\PycharmProjects\\GaussianSplattingViewer\\out\\{scene_folder}\\right\\{image_no}.png"

    disparity_image = load_image(disparity_file_path)
    left_image = load_image(left_file_path, color=True)
    right_image = load_image(right_file_path, color=True)

# Draw random circles on the left image
    random_spots = draw_random_circles(left_image, num_circles=10)

    # Draw corresponding adjusted circles on the right image
    draw_adjusted_circles(random_spots, right_image, disparity_image)

    # Display the images
    cv2.imshow("Left Image with Circles", left_image)
    cv2.imshow("Right Image with Adjusted Circles", right_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



cv2.destroyAllWindows()
