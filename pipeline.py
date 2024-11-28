import os
import warnings
import pandas as pd
import whisper

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Load Whisper model
model = whisper.load_model('base')

# Define the path for the audio files and metadata CSVs
audio_folder = '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/audio'
batch_csv_paths = [
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411295_batch_results.csv',
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411298_batch_results.csv'
]

# Step 1: Load and concatenate the metadata from all the CSV files
metadata_df = pd.concat([pd.read_csv(csv_path) for csv_path in batch_csv_paths], ignore_index=True)

# Step 2: Initialize an empty list to hold the processed data
data = []

# Step 3: Load the existing output CSV (if exists) to check for already processed rows
output_csv = 'transcriptions_output.csv'
if os.path.exists(output_csv):
    existing_df = pd.read_csv(output_csv)
else:
    existing_df = pd.DataFrame(columns=["id"])

# Step 4: Loop through each mp3 file in the audio folder
for audio_file in os.listdir(audio_folder):
    if audio_file.endswith('.mp3'):
        # Extract Vocaroo ID from the audio filename
        vocaroo_id = audio_file.split('.')[0]  # Assuming filename is the Vocaroo link ID
        
        # Skip processing if this vocaroo ID already exists in the output CSV
        if vocaroo_id in existing_df['id'].values:
            continue
        
        # Construct the full path to the audio file
        audio_path = os.path.join(audio_folder, audio_file)
        
        # Transcribe the audio using Whisper
        result = model.transcribe(audio_path)
        
        # Extract the transcription text
        transcription = result['text']
        
        # Find the metadata row based on the Vocaroo ID
        metadata_row = metadata_df[metadata_df['Answer.vocaroo_link'].str.contains(vocaroo_id)]
        
        if not metadata_row.empty:
            # Extract relevant metadata fields
            culturally_distinct = metadata_row['Answer.culturally_distinct.yes'].values[0]
            cultural_distinction_explanation = metadata_row['Answer.cultural_distinction_explanation'].values[0]
            vocaroo_link = metadata_row['Answer.vocaroo_link'].values[0]
            image_link = metadata_row['Input.image_url'].values[0]  # Assuming image_url is in 'Input.image_url'
            
            # Extract language from the "Title" column (assumes the title includes the language name)
            title = metadata_row['Title'].values[0]
            language = title.split('(')[-1].split(')')[0] if '(' in title else 'Unknown'
            
            # Add the data for this row to the list
            data.append({
                'id': vocaroo_id,
                'language': language,
                'culturally_distinct': culturally_distinct,
                'cultural_distinction_explanation': cultural_distinction_explanation,
                'vocaroo_link': vocaroo_link,
                'image_link': image_link,
                'transcription': transcription
            })

# Step 5: Convert the list of dictionaries into a Pandas DataFrame
df = pd.DataFrame(data)

# Step 6: Rearrange the columns to match the desired order
df = df[['id', 'language', 'culturally_distinct', 'cultural_distinction_explanation', 'vocaroo_link', 'image_link', 'transcription']]

# Step 7: Append new data to the existing CSV (if any)
if os.path.exists(output_csv):
    df.to_csv(output_csv, mode='a', header=False, index=False)
else:
    df.to_csv(output_csv, index=False)

# Step 8: Display the DataFrame (optional)
print(df)
