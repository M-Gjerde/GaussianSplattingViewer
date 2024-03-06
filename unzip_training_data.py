import zipfile
import os

# Assuming the folder path to "out_flat_shading" where the zipped parts are located
compressed_folder_path = "out_flat_shading"
# Path where to extract the files
extract_path = "test"

compressed_files = [f for f in os.listdir(compressed_folder_path) if(f.endswith('.zip') and f.startswith("data_part"))]

# Create the output directory if it doesn't exist
os.makedirs(extract_path, exist_ok=True)

# Extract each zip file into the specified output directory
for zip_file in compressed_files:
    with zipfile.ZipFile(os.path.join(compressed_folder_path, zip_file), 'r') as zip_ref:
        zip_ref.extractall(extract_path)

