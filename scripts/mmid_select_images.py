import os
import shutil
import pandas as pd
import json
import requests
from concurrent.futures import ThreadPoolExecutor

csv_path = 'mmid/output/filtered_translations_new.csv'
mmid_folder = 'mmid'
output_base = 'mmid/output_images_new'

os.makedirs(output_base, exist_ok=True)

df = pd.read_csv(csv_path)

grouped = df.groupby('language')

def download_image(args):
    image_link, translated_word, original_index, key, language_output_folder = args
    try:
        response = requests.get(image_link, timeout=10)
        if response.status_code == 200:
            image_extension = os.path.splitext(image_link)[-1]
            new_file_name = f'{translated_word}-{original_index}-{key}{image_extension}'
            destination_file = os.path.join(language_output_folder, new_file_name)
            if os.path.exists(destination_file):
                print(f"File {new_file_name} already exists. Skipping download.")
            else:
                with open(destination_file, 'wb') as img_file:
                    img_file.write(response.content)
                file_size = os.path.getsize(destination_file)
                print(f'Downloaded {image_link} to {destination_file} (Size: {file_size} bytes)')
        else:
            print(f'Failed to download {image_link} (Status code: {response.status_code})')
    except Exception as e:
        print(f'Error downloading {image_link}: {e}')

for language, group in grouped:
    language_output_folder = os.path.join(output_base, f'{language}_images')
    os.makedirs(language_output_folder, exist_ok=True)

    download_tasks = []

    for _, row in group.iterrows():
        original_index = row['original_index']
        translated_word = row['translated_english_word'].replace(' ', '_')

        source_folder = os.path.join(mmid_folder, f'mini-{language.lower()}-package', str(original_index))
        metadata_file = os.path.join(source_folder, 'metadata.json')

        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            for key, data in metadata.items():
                image_link = data.get('image_link')
                if image_link:
                    download_tasks.append((image_link, translated_word, original_index, key, language_output_folder))
        else:
            print(f'Metadata file not found: {metadata_file}')

    with ThreadPoolExecutor(max_workers=50) as executor:
        executor.map(download_image, download_tasks)
