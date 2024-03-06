import os
import subprocess
import datetime
# Base paths for the gs_model and colmap_poses directories
import time
gs_model_base_path = r"C:\Users\mgjer\PycharmProjects\gaussian-splatting\output\scene_"
raw_data_base_path = r"C:\Users\mgjer\Downloads"

# Total number of scenes to process
total_scenes = 270

# Iterate through all scene directories
for i in range(45, total_scenes):
    # Determine the part number (increment every 20 scenes)
    part_number = i // 20 + 1  # Integer division, starts with part 1 for scenes 0-19

    try:
        # Generate folder names
        scene_folder = f"{gs_model_base_path}{i:04d}"
        colmap_poses_folder = os.path.join(raw_data_base_path, f"raw_data_v1_part{part_number}", f"{i:04d}", "poses",
                                           "colmap_text")

        # Construct command
        command = f"python main.py --gs_model {scene_folder} --colmap_poses {colmap_poses_folder}"

        # Execute command
        print(f"Executing: {command}")
        subprocess.run(command, check=True, shell=True)
    except Exception as e:
        print(e)
        continue

