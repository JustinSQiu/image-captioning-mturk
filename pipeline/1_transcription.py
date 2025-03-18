import glob
import os
import warnings

import pandas as pd
import torch
import whisper
from transformers import pipeline

warnings.filterwarnings('ignore', message='FP16 is not supported on CPU')

base_model = whisper.load_model('base')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

hindi_transcribe = pipeline(task='automatic-speech-recognition', model='vasista22/whisper-hindi-small', chunk_length_s=30, device=device)
hindi_transcribe.model.config.forced_decoder_ids = hindi_transcribe.tokenizer.get_decoder_prompt_ids(language='hi', task='transcribe')

telugu_transcribe = pipeline(task='automatic-speech-recognition', model='vasista22/whisper-telugu-base', chunk_length_s=30, device=device)
telugu_transcribe.model.config.forced_decoder_ids = telugu_transcribe.tokenizer.get_decoder_prompt_ids(language='te', task='transcribe')


def transcribe(audio_path, language):
    if language == 'Hindi':
        return hindi_transcribe(audio_path)['text']
    elif language == 'Telugu':
        return telugu_transcribe(audio_path)['text']
    else:
        return base_model.transcribe(audio_path)['text']


if __name__ == '__main__':
    audio_folder = '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/audio'
    mturk_folder = '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/'
    output_csv = 'processed_output/output_transcription_2.csv'
    batch_csv_paths = glob.glob(f'{mturk_folder}/*_batch_results.csv')

    metadata_df = pd.concat([pd.read_csv(csv_path) for csv_path in batch_csv_paths], ignore_index=True)
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
    else:
        existing_df = pd.DataFrame(columns=['id'])

    data = []

    for audio_file in os.listdir(audio_folder):
        if audio_file.endswith('.mp3'):
            vocaroo_id = audio_file.split('.')[0]
            metadata_row = metadata_df[metadata_df['Answer.vocaroo_link'].str.contains(vocaroo_id)]
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

            # skip if already exists or language is not Hindi or Telugu
            if vocaroo_id not in existing_df['id'].values:
                print(f'MISSING {vocaroo_id}')
            else:
                print(f'SKIPPING {vocaroo_id}')
                continue
            # if not (language == 'Hindi' or language == 'Telugu'):
            #     continue

            audio_path = os.path.join(audio_folder, audio_file)
            transcription = transcribe(audio_path, language)
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
    combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=['id'], keep='last')
    combined_df.to_csv('processed_output/output_transcription_cvqa.csv', index=False)
    print(combined_df)
