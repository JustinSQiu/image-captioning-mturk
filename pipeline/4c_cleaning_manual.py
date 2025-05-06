import csv
import os

import pandas as pd

input_file = 'processed_output/output_transcription_translated_manually_cleaned_final.csv'
output_file = 'processed_output/output_transcription_translated_manually_cleaned_ready_for_summarization_final.csv'

df = pd.read_csv(input_file)
df = df[df['Goodness'] != 'INVALID']

df.to_csv(output_file, index=False)
