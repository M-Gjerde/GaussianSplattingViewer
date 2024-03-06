import os
import shutil
import zipfile

# Define the root directory where the zip files are located
root_dir = "C:\\Users\\mgjer\\Downloads\\"
zip_file_pattern = "raw_data_v1_part"

# Find all zip files matching the pattern and unzip them
for part in range(14, 15):
    zip_file_name = f"{zip_file_pattern}{part}.zip"
    zip_file_path = os.path.join(root_dir, zip_file_name)
    if os.path.exists(zip_file_path):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            extract_path = os.path.join(root_dir, zip_file_name.replace('.zip', ''))
            zip_ref.extractall(extract_path)

        # After extraction, iterate through the subfolders and perform the required operations
        for subfolder in os.listdir(extract_path):
            subfolder_path = os.path.join(extract_path, subfolder)
            if os.path.isdir(subfolder_path):
                poses_path = os.path.join(subfolder_path, 'poses')
                if os.path.exists(poses_path):
                    colmap_sparse_path = os.path.join(poses_path, 'colmap_sparse')
                    if os.path.exists(colmap_sparse_path):
                        # Copy the colmap_sparse folder one level up and rename it to sparse
                        destination_path = os.path.join(subfolder_path, 'sparse')
                        shutil.copytree(colmap_sparse_path, destination_path)

# Indicate completion
"Process completed. The 'colmap_sparse' folders have been copied and renamed as required."
