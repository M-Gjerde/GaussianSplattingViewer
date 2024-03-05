import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os


def calculate_scale(scene_folder, save_to_file, debug_plot = False):
    scenes = os.listdir(scene_folder)
    for scene in scenes:
        img1 = cv.imread(f'{scene_folder}/{scene}/left/5.png',cv.IMREAD_GRAYSCALE)          # queryImage
        img2 = cv.imread(f'{scene_folder}/{scene}/right/5.png',cv.IMREAD_GRAYSCALE) # trainImage
        depth = cv.imread(f'{scene_folder}/{scene}/depth/5.png',cv.IMREAD_UNCHANGED) # trainImage
        # Initiate SIFT detector
        sift = cv.SIFT_create()
        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1,None)
        kp2, des2 = sift.detectAndCompute(img2,None)
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary
        flann = cv.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)
        # Need to draw only good matches, so create a mask
        matchesMask = [[0,0] for i in range(len(matches))]
        # ratio test as per Lowe's paper
        matches_count = 0
        good_matches = []
        dist_factor = 0.0
        while matches_count <= 100:
            matches_count = 0
            for i,(m,n) in enumerate(matches):
                if m.distance < dist_factor*n.distance:
                    matchesMask[i]=[1,0]
                    matches_count += 1
                    good_matches.append(m)
            dist_factor += 0.01
            #print(f"Num matches: {matches_count}")

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
            scaled_disparity = depth[int(y1), int(x1)]
            scale_factor = scaled_disparity / d
            scale_factors.append(scale_factor)


        # Calculate average
        average_scale = np.mean(scale_factors)
        # Calculate median
        median_scale = np.median(scale_factors)
        # Calculate standard deviation
        std_dev_scale = np.std(scale_factors)

        depth_gt = depth.copy().astype(np.float64)

        depth_gt /= median_scale
        depth_gt = depth_gt.astype(np.uint16)

        if save_to_file:
            if not os.path.exists(f"{scene_folder}/{scene}/depth_gt"):
                os.mkdir(f"{scene_folder}/{scene}/depth_gt")
            cv.imwrite(f"{scene_folder}/{scene}/depth_gt/5.png", depth_gt)


        if debug_plot:
            draw_params = dict(matchColor=(0, 255, 0),
                               singlePointColor=(255, 0, 0),
                               matchesMask=matchesMask,
                               flags=cv.DrawMatchesFlags_DEFAULT)

            img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
            plt.imshow(img3,),plt.show()


        print(f"Scene: {scene}")
        #print(f"Mean (Average) Scale: {average_scale}")
        print(f"Median Scale: {median_scale}")
        #print(f"Standard Deviation of Scale: {std_dev_scale}")
        print()

if __name__ == "__main__":
    scene_folder = "C:/Users/mgjer/Downloads/data_part1_naive"
    save_to_file = False
    debug_plot = False
    calculate_scale(scene_folder, save_to_file, debug_plot)