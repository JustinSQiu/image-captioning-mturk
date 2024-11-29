import pandas as pd
import os
import subprocess
from math import ceil

# Paths
filtered_csv_path = 'output/filtered_file_list.csv'
filtered_images_folder = 'filtered_images'

# Read file names from the CSV
df = pd.read_csv(filtered_csv_path)
file_names = df['File Name'].tolist()

# Split the files into 10 chunks
chunk_size = ceil(len(file_names) / 10)
chunks = [file_names[i:i + chunk_size] for i in range(0, len(file_names), chunk_size)]

# Process each chunk
for chunk_index, chunk in enumerate(chunks, start=1):
    print(f"Processing chunk {chunk_index}/{len(chunks)}...")
    for file_name in chunk:
        image_path = os.path.join(filtered_images_folder, file_name)
        if os.path.exists(image_path):
            try:
                subprocess.run(["git", "add", image_path], check=True)
                print(f"Added {image_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error adding {image_path}: {e}")
        else:
            print(f"File not found: {image_path}")
    
    # Commit the changes for the current chunk
    commit_message = f"Add filtered images - chunk {chunk_index}"
    try:
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        print(f"Chunk {chunk_index} committed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error committing chunk {chunk_index}: {e}")

# Increase Git's buffer size to handle large files
subprocess.run(["git", "config", "--global", "http.postBuffer", "524288000"], check=True)

# Push the changes
try:
    subprocess.run(["git", "push"], check=True)
    print("All changes pushed successfully.")
except subprocess.CalledProcessError as e:
    print(f"Error pushing changes: {e}")
