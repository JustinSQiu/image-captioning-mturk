import os
import shutil
from PIL import Image
import pandas as pd

input_base = 'mmid/output_images'  # Original images folder
output_base_filtered = 'mmid_images_new'  # New folder for filtered images

os.makedirs(output_base_filtered, exist_ok=True)

# Function to check if an image is valid
def is_valid_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify that it's a valid image
        return True
    except Exception:
        return False

# Iterate over the original images folder structure
for language_folder in os.listdir(input_base):
    language_input_folder = os.path.join(input_base, language_folder)
    
    if os.path.isdir(language_input_folder):
        language_output_folder = os.path.join(output_base_filtered, language_folder)
        os.makedirs(language_output_folder, exist_ok=True)

        # Iterate through all files in each language folder
        for image_file in os.listdir(language_input_folder):
            image_path = os.path.join(language_input_folder, image_file)
            
            # Check file size and if the image is valid
            if os.path.getsize(image_path) >= 50 * 1024 and is_valid_image(image_path):  # 100KB
                # If the image passes, copy it to the new folder
                destination_file = os.path.join(language_output_folder, image_file)
                shutil.copy(image_path, destination_file)
                print(f'Copied {image_file} to {destination_file}')
                valid_image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
                if not os.path.splitext(destination_file)[1] or os.path.splitext(destination_file)[1] not in valid_image_extensions:
                    new_name = f'{os.path.splitext(destination_file)[0]}.jpg'  # Add .jpg extension
                    os.rename(destination_file, new_name)
                    print(f'Renamed {destination_file} to {new_name}')
            else:
                print(f'Filtered out {image_file} (Size or invalid image)')

print("Image filtering and copying complete.")
