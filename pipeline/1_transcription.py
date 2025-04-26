print('Loading libraries...', flush=True)

import os
os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

import glob
import warnings

import pandas as pd
import torch
import whisper
from transformers import pipeline

from huggingface_hub import login
from keys import hf_token

login(token=hf_token)

print('Loading models...', flush=True)

warnings.filterwarnings('ignore', message='FP16 is not supported on CPU')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}', flush=True)

# base_model = whisper.load_model('base')
# turbo_model = whisper.load_model('turbo')

# hindi_transcribe = pipeline(task='automatic-speech-recognition', model='vasista22/whisper-hindi-small', chunk_length_s=30, device=device)
# hindi_transcribe.model.config.forced_decoder_ids = hindi_transcribe.tokenizer.get_decoder_prompt_ids(language='hi', task='transcribe')

# telugu_transcribe = pipeline(task='automatic-speech-recognition', model='vasista22/whisper-telugu-small', chunk_length_s=30, device=device)
# telugu_transcribe.model.config.forced_decoder_ids = telugu_transcribe.tokenizer.get_decoder_prompt_ids(language='te', task='transcribe')

tamil_transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-tamil-small", chunk_length_s=30, device=device)
tamil_transcribe.model.config.forced_decoder_ids = tamil_transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")

# viet_transcribe = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-large", device=device)

# nepali_transcribe = pipeline("automatic-speech-recognition", model="kiranpantha/whisper-large-v3-nepali", device=device) # doesn't work

# bengali_transcribe = pipeline("automatic-speech-recognition", model="KhushiDS/whisper-large-v3-Bengali", device=device)

# telugu_transcribe = pipeline("automatic-speech-recognition", model="KhushiDS/whisper-large-v3-Telugu", device=device)

# amharic_transcribe = pipeline("automatic-speech-recognition", model="drmeeseeks/whisper-large-v2-amet", device=device) # doesn't work

# thai_transcribe = thai_transcribe = pipeline("automatic-speech-recognition", model="biodatlab/whisper-th-large-combined", device=device)

# kinyarwanda_transcribe = pipeline("automatic-speech-recognition", model="mbazaNLP/Whisper-Small-Kinyarwanda", device=device)


def transcribe(audio_path, language):
    '''
    We need to do Tamil, Viet, Nepali, Bengali, Amharic, Thai, and Kinyarwanda because Whisper does a poor job on them.
    '''
    # if language == 'Hindi':
    #     return hindi_transcribe(audio_path)['text']
    # if language == 'Telugu':
    #     return telugu_transcribe(audio_path)['text']
    if language == 'Tamil':
        return tamil_transcribe(audio_path)['text']
    if language == 'Vietnamese':
        return viet_transcribe(audio_path, return_timestamps=True)['text']
    # if language == 'Nepali':
    #     return nepali_transcribe(audio_path)['text']
    if language == 'Bengali':
        return bengali_transcribe(audio_path, return_timestamps=True)['text']
    if language == 'Telugu':
        return telugu_transcribe(audio_path, return_timestamps=True)['text']
    if language == 'Thai':
        return thai_transcribe(audio_path, return_timestamps=True)['text']
    # if language == 'Kinyarwanda':
    #     return kinyarwanda_transcribe(audio_path)['text']
    return turbo_model.transcribe(audio_path, language=language.lower())['text']
    # return base_model.transcribe(audio_path)['text']


if __name__ == '__main__':
    # audio_folder = '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/audio'
    audio_folder = '/nlp/data/jsq/audio'
    # mturk_folder = '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/'
    mturk_folder = '/nlp/data/jsq/mturk_output/'
    previous_csv = '/home1/j/jsq/dev/image-captioning-mturk/processed_output/output_transcription_cvqa_whisper_with_finetunes.csv'
    output_csv = '/home1/j/jsq/dev/image-captioning-mturk/processed_output/output_transcription_cvqa_whisper_with_finetunes.csv'
    # batch_csv_paths = glob.glob(f'{mturk_folder}/*_batch_results.csv')
    batch_csv_paths = glob.glob(f'{mturk_folder}/*_batch_results.csv')

    metadata_df = pd.concat([pd.read_csv(csv_path) for csv_path in batch_csv_paths], ignore_index=True)
    if os.path.exists(previous_csv):
        existing_df = pd.read_csv(previous_csv)
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
            language = title.split('(')[-1].split(')')[0] if '(' in title else title.split(' - ')[-1]
            selected_other_languages = False
            if language == 'Other Languages':
                selected_other_languages = True
                language = metadata_row['Answer.preferred_language'].values[0].capitalize()

            # if vocaroo_id in existing_df['id'].values:
            #     continue

            audio_path = os.path.join(audio_folder, audio_file)
            if language != 'Tamil':
                transcription = existing_df[existing_df['id'] == vocaroo_id]['transcription'].values
                print(f'Found {vocaroo_id} in existing_df, language is {language}', flush=True)
            else:
                try:
                    transcription = transcribe(audio_path, language)
                    print(f'Finished {vocaroo_id}; transcription: {transcription}', flush=True)
                except Exception as e:
                    print(f'Error in {vocaroo_id}: {e}')
                    continue
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
            # print(f'Finished {vocaroo_id}; transcription: {transcription}', flush=True)

    new_df = pd.DataFrame(data)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=['id'], keep='last')
    combined_df.to_csv(output_csv, index=False)
    print(combined_df)
