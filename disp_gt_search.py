import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


def bilinear_interpolate(image, x, y):
    """
    Perform bilinear interpolation for a given x, y location using the surrounding pixels.

    :param image: numpy array of the image or depth map
    :param x: float, the x-coordinate in the image
    :param y: float, the y-coordinate in the image
    :return: interpolated value at (x, y)
    """
    x1, y1 = int(x), int(y)
    x2, y2 = x1 + 1, y1 + 1

    # Bound x2, y2 to be within the image dimensions
    x2 = min(x2, image.shape[1] - 1)
    y2 = min(y2, image.shape[0] - 1)

    # Compute the differences
    dx = x - x1
    dy = y - y1

    # Interpolate
    value = (image[y1, x1] * (1 - dx) * (1 - dy) +
             image[y1, x2] * dx * (1 - dy) +
             image[y2, x1] * (1 - dx) * dy +
             image[y2, x2] * dx * dy)

    return value
def calculate_scale(scene_folder, images_numbers = (0, 5), save_to_file=False, debug_plot=False):
    scenes = os.listdir(scene_folder)
    for scene in scenes:
        print(f"Scene: {scene}")
        median_scales = []
        for image_name in images_numbers:
            img1 = cv.imread(f'{scene_folder}/{scene}/left/{image_name}.png', cv.IMREAD_GRAYSCALE)  # queryImage
            img2 = cv.imread(f'{scene_folder}/{scene}/right/{image_name}.png', cv.IMREAD_GRAYSCALE)  # trainImage
            depth = cv.imread(f'{scene_folder}/{scene}/depth/{image_name}.png', cv.IMREAD_UNCHANGED)  # trainImage
            # Initiate SIFT detector
            sift = cv.SIFT_create()
            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)
            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des2, k=2)
            # Need to draw only good matches, so create a mask
            matchesMask = [[0, 0] for i in range(len(matches))]
            # ratio test as per Lowe's paper
            matches_count = 0
            good_matches = []
            dist_factor = 0.0
            while matches_count <= 25:
                matches_count = 0
                for i, (m, n) in enumerate(matches):
                    if m.distance < dist_factor * n.distance:
                        matchesMask[i] = [1, 0]
                        matches_count += 1
                        good_matches.append(m)
                dist_factor += 0.01
                # print(f"Num matches: {matches_count}")

            # Calculate disparities for good matches
            disparities = []
            key_points = []
            for match in good_matches:
                # KeyPoint for img1
                x1 = kp1[match.queryIdx].pt[0]
                y1 = kp1[match.queryIdx].pt[1]
                # KeyPoint for img2
                x2 = kp2[match.trainIdx].pt[0]
                y2 = kp2[match.trainIdx].pt[1]
                # Disparity in x-direction
                disparity = x1 - x2
                disparities.append(disparity)
                key_points.append(((x1, y1), (x2, y2)))

            key_points = np.array(key_points)
            disparities = np.array(disparities)

            scale_factors = []
            for d, pt in zip(disparities, key_points):
                x1, y1, x2, y2 = pt.flatten()
                scaled_disparity = bilinear_interpolate(depth, x1, y1)
                scale_factor = scaled_disparity / d
                scale_factors.append(scale_factor)
                res = scaled_disparity / scale_factor


            # Calculate average
            average_scale = np.mean(scale_factors)
            # Calculate median
            median_scale = np.median(scale_factors)
            # Calculate standard deviation
            std_dev_scale = np.std(scale_factors)

            depth_gt = depth.copy().astype(np.float64)

            depth_gt /= median_scale
            depth_gt = depth_gt.astype(np.uint16)


            if debug_plot:
                draw_params = dict(matchColor=(0, 255, 0),
                                   singlePointColor=(255, 0, 0),
                                   matchesMask=matchesMask,
                                   flags=cv.DrawMatchesFlags_DEFAULT)

                img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
                plt.imshow(img3, ), plt.show()

            # print(f"Mean (Average) Scale: {average_scale}")
            median_scales.append(median_scale)
            # print(f"Standard Deviation of Scale: {std_dev_scale}")

        median_scale = np.median(median_scales)
        print(f"Median Scale: {median_scale}")
        print(f"Mean Scale: {np.average(median_scales)}")
        for image_name in images_numbers:
            if save_to_file:

                depth = cv.imread(f'{scene_folder}/{scene}/depth/{image_name}.png', cv.IMREAD_UNCHANGED)  # trainImage
                depth_gt = depth.copy().astype(np.float64)

                depth_gt /= median_scale
                depth_gt = (depth_gt * 64).astype(np.uint16)

                if not os.path.exists(f"{scene_folder}/{scene}/depth_gt"):
                    os.mkdir(f"{scene_folder}/{scene}/depth_gt")
                cv.imwrite(f"{scene_folder}/{scene}/depth_gt/{image_name}.png", depth_gt)


if __name__ == "__main__":
    scene_folder = "C:/Users/mgjer/Downloads/data_part1_naive"
    save_to_file = True
    debug_plot = False
    images_per_scene = np.arange(0, 100, 5)
    calculate_scale(scene_folder,images_per_scene, save_to_file, debug_plot)
