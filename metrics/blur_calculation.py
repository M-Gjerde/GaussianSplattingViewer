from pathlib import Path
import cv2
import random
import skimage
from math import log10, sqrt
import cv2
import numpy as np
from pathlib import Path
import os
import subprocess
import datetime
# Base paths for the gs_model and colmap_poses directories
import time

from matplotlib import pyplot as plt

gs_model_base_path = r"C:\Users\mgjer\PycharmProjects\GaussianSplattingViewer\out_baseline_05\scene_"
download_dir = "C:\\Users\\mgjer\\Downloads"

original_paths = []
nerf_paths = []
# Loop through all parts
for part in range(1, 15):
    part_dir_name = f"raw_data_v1_part{part}"
    part_dir_path = os.path.join(download_dir, part_dir_name)

    part_dir_name_nerf = f"stereo_dataset_v1_part{part}"
    part_dir_path_nerf = os.path.join(download_dir, part_dir_name_nerf)

    # Check if the directory exists
    if os.path.exists(part_dir_path) and os.path.isdir(part_dir_path):
        # Loop through all subdirectories in the part directory
        for subfolder in os.listdir(part_dir_path):
            subfolder_path = os.path.join(part_dir_path, subfolder)
            subfolder_path_nerf = os.path.join(part_dir_path_nerf, subfolder)
            # Check if it's a directory
            if os.path.isdir(subfolder_path):
                original_paths.append(subfolder_path)
            if os.path.isdir(subfolder_path_nerf):
                nerf_paths.append(subfolder_path_nerf)
    # Loop through all parts
for part in range(1, 15):
    part_dir_name = f"stereo_dataset_v1_part{part}"
    part_dir_path = os.path.join(download_dir, part_dir_name)

    # Check if the directory exists
    if os.path.exists(part_dir_path) and os.path.isdir(part_dir_path):
        # Loop through all subdirectories in the part directory
        for subfolder in os.listdir(part_dir_path):
            subfolder_path = os.path.join(part_dir_path, subfolder)
            # Check if it's a directory
            if os.path.isdir(subfolder_path):
                nerf_paths.append(subfolder_path)

# Total number of scenes to process
total_scenes = 40
skip_parts = [7, 19, 30, 37, 21]

np.random.seed(42)
blur_scores = []
# Iterate through all scene directoriess
for i in range(0, 40):
    # Determine the part number (increment every 20 scenes)
    part_number = i // 20 + 1  # Integer division, starts with part 1 for scenes 0-19
    if i in skip_parts:
        continue
    try:
        # Generate folder names
        scene_folder = f"{gs_model_base_path}{i:04d}/right"
        file_names_original = os.listdir(os.path.join(original_paths[i], "images"))
        file_names_nerf = os.listdir(os.path.join(nerf_paths[i], "Q","baseline_0.50", "right"))

        random_numbers = np.random.choice(range(100), 25, replace=False)


        for x in random_numbers:
            original_file_name = os.path.join(os.path.join(original_paths[i], "images"), file_names_original[x])
            nerf_file_name = os.path.join(os.path.join(nerf_paths[i], "Q","baseline_0.50", "right"), file_names_nerf[x])

            img_name = str(x) + ".png"
            full_rendered_path_name = os.path.join(scene_folder, img_name)

            rendered_3dgs = cv2.imread(str(full_rendered_path_name))  # Ensure to convert Path object to string

            # Create a mask where the pixels that are not black (i.e., not 0) are set to 255
            black_pixels_mask = cv2.inRange(rendered_3dgs, 0, 0)

            # Count the non-black (non-zero) pixels in the mask
            non_black_count = cv2.countNonZero(black_pixels_mask)

            # The number of black pixels is the total number of pixels minus the non-black ones
            total_pixels = rendered_3dgs.size
            black_count = total_pixels - non_black_count

            percentage = (black_count/total_pixels) * 100
            if percentage < 99.5:
                continue

            values = []
            try:
                # Select an image file sequentially from the shuffled list
                kSize = 3
                rendered_3dgs_gray = cv2.cvtColor(rendered_3dgs, cv2.COLOR_BGR2GRAY)
                blur_ref_3dgs = skimage.measure.blur_effect(rendered_3dgs_gray, h_size=kSize)

                rendered_nerf = cv2.imread(str(nerf_file_name))  # Ensure to convert Path object to string
                rendered_nerf_gray = cv2.cvtColor(rendered_nerf, cv2.COLOR_BGR2GRAY)
                blur_ref_nerf = skimage.measure.blur_effect(rendered_nerf_gray, h_size=kSize)

                original = cv2.imread(original_file_name)
                original_resized = cv2.resize(original, (rendered_3dgs.shape[1], rendered_3dgs.shape[0]))
                original_resized_gray = cv2.cvtColor(original_resized, cv2.COLOR_BGR2GRAY)
                blur_ref_original = skimage.measure.blur_effect(original_resized_gray, h_size=kSize)

                #print(f"Blurred 3DGS: {blur_ref_3dgs}, Nerf: {blur_ref_nerf} FileNames: {Path(full_rendered_path_name).name}, Nerf: {Path(nerf_file_name).name}")
                print(f"Scene: {i:04d}, Number of black pixels: {percentage:0.2f},  FileNames: {i:04d}/left/{Path(full_rendered_path_name).name} Most blurred {blur_ref_3dgs - blur_ref_nerf}, 3dgs_less blurry?: {blur_ref_3dgs < blur_ref_nerf}")
                # writer.writerow(["Reference:", str(sum(values) / len(values))])

                blur_scores.append([blur_ref_original, blur_ref_nerf, blur_ref_3dgs])
            except Exception as e:
                print(e)
                # writer.writerow(["Reference:", 0])

            # file.close()


            #print(f"Winner is: {winner} PSNR value nerf: {value_nerf:.3f} dB, 3dgs: {value_3dgs:.3f} dB. FileNames: {Path(full_rendered_path_name).name}, Nerf: {Path(nerf_file_name).name}")

    except Exception as e:
        print(e)

blur_scores= np.array(blur_scores)

folder = "blur_calculation"
if not os.path.exists(folder):
    os.mkdir(folder)

np.save(f"{folder}/blur_scores.npy", blur_scores)
