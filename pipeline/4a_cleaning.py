import os
import csv
import pandas as pd
from helpers import get_gpt_response

# Define file paths
input_file = 'processed_output/output_transcription_translated_final.csv'
output_file = 'processed_output/output_translation_cleaned_english_final.csv'

# Load the input CSV, preserving the original structure
df_in = pd.read_csv(input_file)

# Build a set of already processed row IDs based on the output file (if it exists)
processed_ids = set()
if os.path.exists(output_file) and os.stat(output_file).st_size > 0:
    try:
        df_out = pd.read_csv(output_file)
        # Use the "id" column from the CSV as the unique identifier
        processed_ids = set(df_out['id'].astype(str).tolist())
    except Exception as e:
        print(f"Error reading output file, starting fresh: {e}")

# Open the output file in append mode so that we write as we go
with open(output_file, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if os.stat(output_file).st_size == 0:
        header = df_in.columns.tolist() + ['cleaned_translation']
        writer.writerow(header)

    # Process each row one by one
    for index, row in df_in.iterrows():
        row_id = str(row['id'])
        # Skip processing if this row was already handled
        if row_id in processed_ids:
            print(f"Row {row_id} already processed; skipping.")
            continue

        transcription = row['translation']
        language = row['language']

        # Build GPT messages for cleaning
        messages = [
            {
                'role': 'system',
                'content': (
                    "Remove any hallucinations from the following transcription. Do not add any new information or change the style or content of the original transcription. Your output should strictly be a subset of the original transcription, with hallucinations removed. Only remove hallucinations from the end of the transcription. "
                )
            },
            {
                'role': 'user',
                'content': f"Transcription: {transcription}"
            },
            {
                'role': 'assistant',
                'content': f"{transcription}"
            },
            {
                'role': 'user',
                'content': f"Transcription: {transcription}"
            }
        ]
        try:
            cleaned_transcription = get_gpt_response(messages)
            # Remove newline characters for a cleaner CSV output
            cleaned_transcription = cleaned_transcription.replace("\n", " ").replace("\r", " ")
            print(f"Row {row_id} cleaned successfully.")
        except Exception as e:
            print(f"Error cleaning row {row_id}: {e}")
            cleaned_transcription = ""

        # Write the original row plus the new cleaned_transcription column to the output CSV
        new_row = row.tolist() + [cleaned_transcription]
        writer.writerow(new_row)
        f.flush()  # Flush after every row to ensure progress is saved

print("Cleaning process complete.")
