import os
import subprocess
import datetime
# Base paths for the gs_model and colmap_poses directories
import time

gs_model_base_path = r"C:\Users\mgjer\PycharmProjects\GaussianSplattingViewer\out_baseline_05\scene_"
raw_data_base_path = r"C:\Users\mgjer\Downloads"

# Total number of scenes to process
total_scenes = 40

# Iterate through all scene directories
for i in range(0, total_scenes):
    # Determine the part number (increment every 20 scenes)
    part_number = i // 20 + 1  # Integer division, starts with part 1 for scenes 0-19

    # Generate folder names
    scene_folder = f"{gs_model_base_path}{i:04d}"
    depth_folder = os.path.join(scene_folder, 'depth')
    try:
        if len(os.listdir(depth_folder)) >= 100:
            # print("Found 100 files in folder", scene_folder)
            pass
        else:
            print("Missing files in folder", scene_folder, "Number of files: ", len(os.listdir(depth_folder)))
    except Exception as e:
        print(e)