# import os
# import csv
# import pandas as pd
# from helpers import get_gpt_response

# # Define file paths
# input_file = 'processed_output/output_summarized_by_image_english_final.csv'
# output_file = 'processed_output/output_summarized_by_english_backtranslated_final.csv'

# # Load the input CSV, preserving the original structure
# df_in = pd.read_csv(input_file)

# # Build a set of already processed row IDs based on the output file (if it exists)
# processed_ids = set()
# if os.path.exists(output_file) and os.stat(output_file).st_size > 0:
#     try:
#         df_out = pd.read_csv(output_file)
#         # Use the "id" column from the CSV as the unique identifier
#         processed_ids = set(df_out['id'].astype(str).tolist())
#     except Exception as e:
#         print(f"Error reading output file, starting fresh: {e}")

# # Open the output file in append mode so that we write as we go
# with open(output_file, mode='a', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     # If the file is empty, write the header row.
#     # Assuming the input CSV has the following columns:
#     # id, language, culturally_distinct, cultural_distinction_explanation, vocaroo_link, image_link, transcription, selected_other_languages
#     if os.stat(output_file).st_size == 0:
#         header = df_in.columns.tolist() + ['backtranslation']
#         writer.writerow(header)

#     # Process each row one by one
#     for index, row in df_in.iterrows():
#         row_id = str(row['id'])
#         # Skip processing if this row was already handled
#         if row_id in processed_ids:
#             print(f"Row {row_id} already processed; skipping.")
#             continue

#         transcription = row['summary']
#         for language in ['Chinese', 'Korean', 'Russian', 'Hindi', 'Tamil', 'Japanese', 'Vietnamese', 'Bengali', 'Spanish', 'Telugu', 'Norwegian', 'German', 'Thai', 'Urdu', 'French']:

#             # Build GPT messages for translation
#             messages = [
#                 {
#                     'role': 'system',
#                     'content': (
#                         f"Translate the following transcription into {language}. "
#                         "Preserve the original meaning, tone, and specific details. "
#                         f"Ensure that your output is entirely in {language}. "
#                         "Do not include any other text or formatting besides the translation itself. For example, do not include 'Translation:' or 'The transcription describes' or any other meta text. "
#                     )
#                 },
#                 {
#                     'role': 'user',
#                     'content': f"Transcription: {transcription}"
#                 }
#             ]
#             try:
#                 translation = get_gpt_response(messages)
#                 # Remove newline characters for a cleaner CSV output
#                 translation = translation.replace("\n", " ").replace("\r", " ")
#                 print(f"Row {row_id} translated successfully.")
#             except Exception as e:
#                 print(f"Error translating row {row_id}: {e}")
#                 translation = ""

#             # Write the original row plus the new translation column to the output CSV
#             new_row = row.tolist() + [translation]
#             writer.writerow(new_row)
#             f.flush()  # Flush after every row to ensure progress is saved

# print("Translation process complete.")

import os
import csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from helpers import get_gpt_response

input_file  = 'processed_output/output_summarized_by_image_english_final.csv'
output_file = 'processed_output/output_summarized_by_english_backtranslated_final.csv'

df_in = pd.read_csv(input_file)

processed_ids = set()
if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
    try:
        processed_ids = set(pd.read_csv(output_file)['image_link'].astype(str))
    except Exception as e:
        print(f"Error reading output file, starting fresh: {e}")

languages = [
    'Chinese', 'Korean', 'Russian', 'Hindi', 'Tamil', 'Japanese',
    'Vietnamese', 'Bengali', 'Spanish', 'Telugu', 'Norwegian',
    'German', 'Thai', 'Urdu', 'French'
]

write_lock = Lock()          # one lock guards file writes
fout = open(output_file, 'a', newline='', encoding='utf‑8')
writer = csv.writer(fout)

if os.path.getsize(output_file) == 0:
    header = df_in.columns.tolist() + ['backtranslation']
    writer.writerow(header)
    fout.flush()

def translate_and_write(row_series, language):
    """Translate one row into one language and append to CSV (thread‑safe)."""
    row_id       = str(row_series['image_link'])
    transcription = row_series['summary']
    messages = [
        {
            'role': 'system',
            'content': (
                f"Translate the following transcription into {language}. "
                "Preserve the original meaning, tone, and specific details. "
                f"Ensure your output is **entirely in {language}**—no extra headings or meta text."
            )
        },
        {'role': 'user', 'content': f"Transcription: {transcription}"}
    ]
    try:
        translation = get_gpt_response(messages).replace('\n', ' ').replace('\r', ' ')
        print(f"[{row_id}] → {language}: done")
    except Exception as e:
        print(f"[{row_id}] → {language}: ERROR → {e}")
        translation = ""
    with write_lock:
        writer.writerow(row_series.tolist() + [translation])
        fout.flush()

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = []
    for _, row in df_in.iterrows():
        if str(row['image_link']) in processed_ids:
            print(f"[{row['image_link']}] already processed; skipping.")
            continue

        for lang in languages:
            # `.copy()` so each thread owns its own data and is pickle‑safe
            futures.append(pool.submit(translate_and_write, row.copy(), lang))

    for future in futures:
        future.result()

fout.close()
print("🎉  All translations complete.")
