import pandas as pd
import glob

input_csv_path = '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/processed_output/output_transcription_2.csv'
mturk_folder = '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/'
batch_csv_paths = glob.glob(f'{mturk_folder}/*_batch_results.csv')
output_csv_path = '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/processed_output/output_transcription_full_2.csv'

input_df = pd.read_csv(input_csv_path)
print(len(input_df)) # 2720
metadata_df = pd.concat([pd.read_csv(csv_path) for csv_path in batch_csv_paths], ignore_index=True)
print(len(metadata_df)) # 2873
metadata_df['id'] = metadata_df['Answer.vocaroo_link'].str.split('/').str[-1]
# Check for duplicate IDs in metadata_df
duplicate_mask = metadata_df.duplicated(subset=['id'], keep=False)
print(f"Duplicate IDs in metadata: \n{metadata_df[duplicate_mask]['id']}")

merged_df = pd.merge(input_df, metadata_df[['id', 'WorkerId']], on='id', how='left')
print(len(merged_df)) # 2753
output_columns = [
    'id', 'language', 'culturally_distinct', 'cultural_distinction_explanation',
    'vocaroo_link', 'image_link', 'transcription', 'selected_other_languages', 'WorkerId'
]
final_df = merged_df[output_columns]
# final_df.to_csv(output_csv_path, index=False)
