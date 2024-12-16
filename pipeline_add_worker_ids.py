import pandas as pd

input_csv_path = '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/processed_output/output_transcription.csv'
batch_csv_paths = [
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411295_batch_results.csv',
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411298_batch_results.csv',
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411301_batch_results.csv',
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411302_batch_results.csv',
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411304_batch_results.csv',
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411306_batch_results.csv',
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411307_batch_results.csv',
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411316_batch_results.csv',
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411358_batch_results.csv',
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411359_batch_results.csv',
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411361_batch_results.csv',
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411363_batch_results.csv',
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411364_batch_results.csv',
    '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/Batch_411533_batch_results.csv',
]
output_csv_path = '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/processed_output/output_transcription_new.csv'

input_df = pd.read_csv(input_csv_path)
metadata_df = pd.concat([pd.read_csv(csv_path) for csv_path in batch_csv_paths], ignore_index=True)
metadata_df['vocaroo_id'] = metadata_df['Answer.vocaroo_link'].str.split('/').str[-1]
input_df['vocaroo_id'] = input_df['vocaroo_link'].str.split('/').str[-1]
merged_df = pd.merge(input_df, metadata_df[['vocaroo_id', 'WorkerId']], on='vocaroo_id', how='left')
output_columns = [
    'id', 'language', 'culturally_distinct', 'cultural_distinction_explanation',
    'vocaroo_link', 'image_link', 'transcription', 'selected_other_languages', 'WorkerId'
]
final_df = merged_df[output_columns]
final_df.to_csv(output_csv_path, index=False)
