import os
import pandas as pd

base_folder = 'mmid_images'  # Base folder containing language subfolders
output_folder = 'output'
base_url = 'https://raw.githubusercontent.com/JustinSQiu/image-captioning-mturk/master/mmid_images/'

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
                image_list.append([image_url])

        # Create a DataFrame and save it as a CSV
        df = pd.DataFrame(image_list, columns=['image_url'])
        csv_file_path = os.path.join(output_folder, f'{language_folder}_image_urls.csv')
        df.to_csv(csv_file_path, index=False)

        print(f'CSV created for {language_folder}: {csv_file_path}')
