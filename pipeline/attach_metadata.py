import pandas as pd
import glob

input_csv_path = '/home1/j/jsq/dev/image-captioning-mturk/processed_output/output_transcription_translated_manually_cleaned_final.csv'
mturk_folder = '/home1/j/jsq/dev/image-captioning-mturk/mturk_output/'
batch_csv_paths = glob.glob(f'{mturk_folder}/*_batch_results.csv')
output_csv_path = '/home1/j/jsq/dev/image-captioning-mturk/processed_output/output_transcription_translated_manually_cleaned_metadata_attached_final.csv'

input_df = pd.read_csv(input_csv_path)
metadata_df = pd.concat([pd.read_csv(csv_path) for csv_path in batch_csv_paths], ignore_index=True)
metadata_df['id'] = metadata_df['Answer.vocaroo_link'].str.split('/').str[-1]

metadata_df_clean = metadata_df.drop_duplicates(subset=['id'], keep=False)
merged_df = pd.merge(input_df, metadata_df_clean[['id', 'WorkerId']], on='id', how='inner')

output_columns = input_df.columns.tolist() + ['WorkerId']
final_df = merged_df[output_columns]
final_df.to_csv(output_csv_path, index=False)
