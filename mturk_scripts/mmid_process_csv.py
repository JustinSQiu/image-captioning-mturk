import os
import pandas as pd
import random

base_folder = 'mmid_images'  # Base folder containing language subfolders
output_folder = 'mturk_upload_files'
base_url = 'https://raw.githubusercontent.com/JustinSQiu/image-captioning-mturk/master/mmid_images/'

os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

# Iterate over each language subfolder
for language_folder in os.listdir(base_folder):
    language_folder_path = os.path.join(base_folder, language_folder)
    
    if os.path.isdir(language_folder_path):
        image_list = []

        # Iterate through all files in the language folder
        for image_file in os.listdir(language_folder_path):
            image_path = os.path.join(language_folder_path, image_file)
            
            # Check if it's a file (not a subfolder)
            if os.path.isfile(image_path):
                # Prepend the URL to the image file name
                image_url = f'{base_url}{language_folder}/{image_file}'
                image_list.append(image_url)

        # Create a DataFrame and save it as a full CSV
        df = pd.DataFrame(image_list, columns=['image_url'])
        full_csv_path = os.path.join(output_folder, f'{language_folder}_image_urls.csv')
        df.to_csv(full_csv_path, index=False)
        print(f'CSV created for {language_folder}: {full_csv_path}')

        # Create a sampled CSV with up to 500 random URLs
        sample_size = min(500, len(image_list))
        sampled_image_list = random.sample(image_list, sample_size)
        sampled_df = pd.DataFrame(sampled_image_list, columns=['image_url'])
        sampled_csv_path = os.path.join(output_folder, f'{language_folder}_image_urls_batch_1.csv')
        sampled_df.to_csv(sampled_csv_path, index=False)
        print(f'Sampled CSV created for {language_folder}: {sampled_csv_path}')
