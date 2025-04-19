import os
import csv
import pandas as pd
from helpers import get_gpt_response

# Define file paths
input_file = 'processed_output/output_transcription_cvqa_whisper_only_language_specified.csv'
output_file = 'processed_output/output_translated.csv'

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
    # If the file is empty, write the header row.
    # Assuming the input CSV has the following columns:
    # id, language, culturally_distinct, cultural_distinction_explanation, vocaroo_link, image_link, transcription, selected_other_languages
    if os.stat(output_file).st_size == 0:
        header = df_in.columns.tolist() + ['translation']
        writer.writerow(header)

    # Process each row one by one
    for index, row in df_in.iterrows():
        row_id = str(row['id'])
        # Skip processing if this row was already handled
        if row_id in processed_ids:
            print(f"Row {row_id} already processed; skipping.")
            continue

        transcription = row['transcription']
        language = row['language']

        # For rows with language "English", no translation is needed.
        if language.strip().lower() == "english":
            translation = transcription
            print(f"Row {row_id} is already English; copying transcription to the translation column.")
        else:
            # Build GPT messages for translation
            messages = [
                {
                    'role': 'system',
                    'content': (
                        "Translate the following transcription into English. "
                        "Preserve the original meaning, tone, and details. "
                        "Ensure that your output is entirely in English."
                    )
                },
                {
                    'role': 'user',
                    'content': f"Transcription: {transcription}"
                }
            ]
            try:
                translation = get_gpt_response(messages)
                # Remove newline characters for a cleaner CSV output
                translation = translation.replace("\n", " ").replace("\r", " ")
                print(f"Row {row_id} translated successfully.")
            except Exception as e:
                print(f"Error translating row {row_id}: {e}")
                translation = ""

        # Write the original row plus the new translation column to the output CSV
        new_row = row.tolist() + [translation]
        writer.writerow(new_row)
        f.flush()  # Flush after every row to ensure progress is saved

print("Translation process complete.")
