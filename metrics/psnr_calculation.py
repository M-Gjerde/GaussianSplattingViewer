from math import log10, sqrt
import cv2
import numpy as np

from pathlib import Path

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if (mse == 0):  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


import os
import subprocess
import datetime
# Base paths for the gs_model and colmap_poses directories
import time

gs_model_base_path = r"C:\Users\mgjer\PycharmProjects\GaussianSplattingViewer\out_baseline_05\scene_"
download_dir = "C:\\Users\\mgjer\\Downloads"

original_paths = []
nerf_paths = []
# Loop through all parts
for part in range(1, 15):
    part_dir_name = f"raw_data_v1_part{part}"
    part_dir_path = os.path.join(download_dir, part_dir_name)

    # Check if the directory exists
    if os.path.exists(part_dir_path) and os.path.isdir(part_dir_path):
        # Loop through all subdirectories in the part directory
        for subfolder in os.listdir(part_dir_path):
            subfolder_path = os.path.join(part_dir_path, subfolder)
            # Check if it's a directory
            if os.path.isdir(subfolder_path):
                original_paths.append(subfolder_path)

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
total_scenes = 270

winners = []
# Iterate through all scene directories
for i in range(0, 1):
    # Determine the part number (increment every 20 scenes)
    part_number = i // 20 + 1  # Integer division, starts with part 1 for scenes 0-19

    try:
        # Generate folder names
        scene_folder = f"{gs_model_base_path}{i:04d}/left"
        file_names_original = os.listdir(os.path.join(original_paths[i], "images"))
        file_names_nerf = os.listdir(os.path.join(nerf_paths[i], "Q", "center"))

        for x in range(0, 100):
            original_file_name = os.path.join(os.path.join(original_paths[i], "images"), file_names_original[x])
            nerf_file_name = os.path.join(os.path.join(nerf_paths[i], "Q", "center"), file_names_nerf[x])

            img_name = str(x) + ".png"
            full_rendered_path_name = os.path.join(scene_folder, img_name)

            rendered_3dgs = cv2.imread(full_rendered_path_name)
            rendered_nerf = cv2.imread(nerf_file_name)

            original = cv2.imread(original_file_name)
            original_resized = cv2.resize(original, (rendered_3dgs.shape[1], rendered_3dgs.shape[0]))

            value_3dgs = PSNR(original_resized, rendered_3dgs)
            value_nerf = PSNR(original_resized, rendered_nerf)
            winners.append(value_3dgs > value_nerf)

            if value_nerf > value_3dgs:
                winner = "nerf"
            else:
                winner = "3dgs"
            print(f"Winner is: {winner} PSNR value nerf: {value_nerf:.3f} dB, 3dgs: {value_3dgs:.3f} dB. FileNames: {Path(full_rendered_path_name).name}, Nerf: {Path(nerf_file_name).name}")

    except Exception as e:
        print(e)
