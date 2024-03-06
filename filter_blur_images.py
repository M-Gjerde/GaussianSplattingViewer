import os
import random
from pathlib import Path

import cv2
import pandas as pd
from io import StringIO

import skimage
from sklearn.preprocessing import MinMaxScaler


def normalize_blur_scores(df):
    # Extracting the Blur Score column for normalization
    blur_scores = df['Blur score 0 - Low and 1- High'].values.reshape(-1, 1)
    # Renaming the columns

    # Initializing the MinMaxScaler
    scaler = MinMaxScaler()

    # Normalizing the Blur Scores
    df['Normalized Blur Score'] = scaler.fit_transform(blur_scores)

    df.rename(columns={
        'Rendered ID': 'id',
        'Blur score 0 - Low and 1- High': 'score',
        'Normalized Blur Score': 'norm_score'
    }, inplace=True)
    return df

def blur_reference_score(original_data_path):
    # Check blur score with training images
    for folder in os.listdir(original_data_path):
        num_blur_checks = 5
        path = Path(original_data_path)
        images_path = path / folder / "images"
        # List all image files (assuming .jpg and .png files for this example)
        image_files = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
        random.shuffle(image_files)
        # Open the CSV file in write mode ('w') or append mode ('a') as needed
        # CSV file to store the results
        values = []
        for x in range(num_blur_checks):
            # Select an image file sequentially from the shuffled list
            if len(image_files) > 0:  # Check if there are still images left to select
                selected_image = image_files.pop(0)  # Remove the first image from the list to avoid repetition
                refImage = cv2.imread(str(selected_image))  # Ensure to convert Path object to string
                refImage_gray = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY)
                refImage_gray_resized = cv2.resize(refImage_gray, (1160, 552))
                blur_ref = skimage.measure.blur_effect(refImage_gray_resized, h_size=11)
                values.append(blur_ref)
                print("Reference blur score: ", blur_ref)
            else:
                print("No more unique images available.")
                break  # or handle according to your needs
        print("Average blur score: ", sum(values) / len(values))

def calculate_blur_scores(dataset_path):
    # L/R/D folders
    pass
    # Calcualte reference score
    # Resize to training data size 1152 x 522


if __name__ == "__main__":
    blur_reference_score(r"C:\Users\mgjer\Downloads\stereo_dataset_v1_part1")

    calculate_blur_scores("data")

    df = pd.read_csv(r"C:\Users\mgjer\Downloads\data_part1_naive\scene_0001\blur_score.csv")
    df = normalize_blur_scores(df)
    print(df)
