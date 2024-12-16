'''
Script to get vocaroo images into audio folder from mturk output csv
'''


import os
import pandas as pd
import requests

csv_files = [
    'mturk_output/Batch_411295_batch_results.csv',
    'mturk_output/Batch_411298_batch_results.csv',
    'mturk_output/Batch_411301_batch_results.csv',
    'mturk_output/Batch_411302_batch_results.csv',
    'mturk_output/Batch_411304_batch_results.csv',
    'mturk_output/Batch_411306_batch_results.csv',
    'mturk_output/Batch_411316_batch_results.csv',
    'mturk_output/Batch_411358_batch_results.csv',
    'mturk_output/Batch_411359_batch_results.csv',
    'mturk_output/Batch_411361_batch_results.csv',
    'mturk_output/Batch_411363_batch_results.csv',
    'mturk_output/Batch_411364_batch_results.csv',
    'mturk_output/Batch_411533_batch_results.csv',
]
output_folder = 'audio'

for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    vocaroo_links = df['Answer.vocaroo_link']

    for link in vocaroo_links:
        if pd.notna(link):
            vocaroo_id = os.path.basename(link)
            file_name = vocaroo_id + '.mp3'
            file_path = os.path.join(output_folder, file_name)
            mp3_url = f'https://media.vocaroo.com/mp3/{vocaroo_id}'
            if not os.path.exists(file_path):
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Referer': link,
                    }
                    response = requests.get(mp3_url, headers=headers, stream=True)
                    response.raise_for_status()
                    with open(file_path, 'wb') as audio_file:
                        for chunk in response.iter_content(chunk_size=8192):
                            audio_file.write(chunk)
                    print(f'Downloaded: {file_name}')
                except requests.RequestException as e:
                    print(f'Failed to download {mp3_url}: {e}')
            else:
                print(f'File already exists: {file_name}')
