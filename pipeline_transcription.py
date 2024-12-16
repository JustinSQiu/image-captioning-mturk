import os
import warnings

import pandas as pd
import whisper

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

model = whisper.load_model('base')

audio_folder = '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/audio'
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

metadata_df = pd.concat([pd.read_csv(csv_path) for csv_path in batch_csv_paths], ignore_index=True)
output_csv = 'processed_output/output_transcription.csv'
if os.path.exists(output_csv):
    existing_df = pd.read_csv(output_csv)
else:
    existing_df = pd.DataFrame(columns=["id"])

data = []
existing_ids = set(existing_df['id'].values)

for audio_file in os.listdir(audio_folder):
    if audio_file.endswith('.mp3'):
        vocaroo_id = audio_file.split('.')[0]
        if vocaroo_id in existing_ids:
            continue
        audio_path = os.path.join(audio_folder, audio_file)
        result = model.transcribe(audio_path)
        transcription = result['text']

        metadata_row = metadata_df[metadata_df['Answer.vocaroo_link'].str.contains(vocaroo_id)]
        if not metadata_row.empty:
            culturally_distinct = metadata_row['Answer.culturally_distinct.yes'].values[0]
            cultural_distinction_explanation = metadata_row['Answer.cultural_distinction_explanation'].values[0]
            vocaroo_link = metadata_row['Answer.vocaroo_link'].values[0]
            image_link = metadata_row['Input.image_url'].values[0]
            title = metadata_row['Title'].values[0]
            language = title.split('(')[-1].split(')')[0] if '(' in title else 'Unknown'
            selected_other_languages = False
            if language == 'Other Languages':
                selected_other_languages = True
                language = metadata_row['Answer.preferred_language'].values[0].capitalize()
            data.append(
                {
                    'id': vocaroo_id,
                    'language': language,
                    'culturally_distinct': culturally_distinct,
                    'cultural_distinction_explanation': cultural_distinction_explanation,
                    'vocaroo_link': vocaroo_link,
                    'image_link': image_link,
                    'transcription': transcription,
                    'selected_other_languages': selected_other_languages
                }
            )
            print(f'Processed {vocaroo_id}')

new_df = pd.DataFrame(data)
combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=['id'])
combined_df.to_csv(output_csv, index=False)
print(combined_df)
