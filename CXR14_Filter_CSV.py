# Import required libraries
import os
import pandas as pd
import shutil

# Define file paths for data organization
base_path = "./Raw_CXR14"
output_base_path = "./CXR14"
output_nodule_path = os.path.join(output_base_path, "nodule")
output_normal_path = os.path.join(output_base_path, "normal")
csv_file_path = os.path.join(base_path, "Data_Entry_2017_v2020.csv")

# Create output directories for organized dataset structure
os.makedirs(output_nodule_path, exist_ok=True)
os.makedirs(output_normal_path, exist_ok=True)

# Load and process the dataset metadata provided in the CSV file
df = pd.read_csv(csv_file_path)

# Filter images based on specific medical conditions:
# 'Nodule' for positive cases
# 'No Finding' for normal cases
nodule_images = df[df['Finding Labels'].str.contains('Nodule', na=False)]
normal_images = df[df['Finding Labels'] == 'No Finding']

def copy_images(image_list, output_path):
    total_images = len(image_list)
    for i, img_name in enumerate(image_list):
        img_path = None
        # Search through all possible source folders
        for folder in range(1, 13):  # Raw CXR14 folders are named images_001 to images_012
            potential_path = os.path.join(base_path, f"images_{folder:03}", img_name)
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path:
            shutil.copy(img_path, os.path.join(output_path, img_name))
        
        # Progress tracking
        if i % 100 == 0:
            print(f"Copied {i}/{total_images} images to {output_path}")

# Process and organize the dataset
print("Copying nodule images")
copy_images(nodule_images['Image Index'], output_nodule_path)
print("Nodule images copied")

print("Copying normal images")
copy_images(normal_images['Image Index'], output_normal_path)
print("Normal images copied")

print("Filtering complete")