import os
from zipfile import ZipFile

# Assuming the folder path to "output" and it contains folders named "scene_xxxx"
folder_path = "./out_baseline_05"
output_folders = sorted([f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))])
split = 5
# Split the list of folders into 10 nearly equal parts
split_size = len(output_folders) // split
splits = [output_folders[i:i + split_size] for i in range(0, len(output_folders), split_size)]

# Ensure the list is split into exactly 5 parts, adjusting if necessary
while len(splits) > split:
    splits[-2].extend(splits[-1])
    splits.pop()

# Zip the splits into separate zip files
zip_file_paths = []
for i, split in enumerate(splits, start=1):
    zip_file_path = f"out_baseline_05_compressed/data_part{i}.zip"
    zip_file_paths.append(zip_file_path)
    with ZipFile(zip_file_path, 'w') as zipf:
        for folder in split:
            folder_path_full = os.path.join(folder_path, folder)
            for root, dirs, files in os.walk(folder_path_full):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(folder_path_full, "..")))

