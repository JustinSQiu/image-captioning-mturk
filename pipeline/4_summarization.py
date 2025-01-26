import csv
import os

import pandas as pd
from helpers import get_gpt_response

input_file = 'processed_output/output_transcription_cleaned.csv'
output_file = 'processed_output/output_summarization_multiple.csv'

df = pd.read_csv(input_file)
df.dropna(subset=['transcription'], inplace=True)
grouped = df.groupby(['image_link', 'language'])

if os.path.exists(output_file) and os.stat(output_file).st_size > 0:
    completed_df = pd.read_csv(output_file)
    completed_keys = set(
        zip(completed_df['image_link'], completed_df['language'])
    )
else:
    completed_keys = set()

with open(output_file, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if os.stat(output_file).st_size == 0:
        writer.writerow(['image_link', 'language', 'ids', 'transcriptions', 'distinct', 'explanations', 'summary'])

    for (image_link, language), group in grouped:
        if (image_link, language) in completed_keys:
            print('.', end='')
            continue

        ids = group['id'].tolist()
        transcriptions = group['transcription'].tolist()
        distinct = group['culturally_distinct'].tolist()
        explanations = group['cultural_distinction_explanation'].tolist()

        if len(transcriptions) == 1:
            continue


        if len(transcriptions) > 1:
            prompt = f"Enhance these transcriptions into a high-quality image caption in the same language as the transcriptions and preserving all the details in the original captions. Your output must be in the same language as the original transcription:\n\n{''.join(transcriptions)}"
        else:
            prompt = f"Enhance this transcription into a high-quality image caption in the same language as the transcription. Your output must be in the same language as the original transcription.:\n\n{transcriptions[0]}"

        try:
            summary = get_gpt_response(prompt)
            summary = summary.replace("\n", " ").replace("\r", " ")
        except Exception as e:
            print(f"Error processing group {image_link}, {language}: {e}")
            summary = "ERROR_PROCESSING"

        writer.writerow([image_link, language, ids, transcriptions, distinct, explanations, summary])
        print(f"Captioned group: {image_link}, {language}")
