print('Loading libraries...', flush=True)

import os
os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

import glob
import warnings

import pandas as pd
import torch
import whisper
from transformers import pipeline

<<<<<<< Updated upstream
from huggingface_hub import login
from keys import hf_token

login(token=hf_token)

print('Loading models...', flush=True)

warnings.filterwarnings('ignore', message='FP16 is not supported on CPU')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}', flush=True)

base_model = whisper.load_model('base')
turbo_model = whisper.load_model('turbo')
=======
# warnings.filterwarnings('ignore', message='FP16 is not supported on CPU')

# base_model = whisper.load_model('base')
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
>>>>>>> Stashed changes

# hindi_transcribe = pipeline(task='automatic-speech-recognition', model='vasista22/whisper-hindi-small', chunk_length_s=30, device=device)
# hindi_transcribe.model.config.forced_decoder_ids = hindi_transcribe.tokenizer.get_decoder_prompt_ids(language='hi', task='transcribe')

<<<<<<< Updated upstream
telugu_transcribe = pipeline(task='automatic-speech-recognition', model='vasista22/whisper-telugu-small', chunk_length_s=30, device=device)
telugu_transcribe.model.config.forced_decoder_ids = telugu_transcribe.tokenizer.get_decoder_prompt_ids(language='te', task='transcribe')
=======
# telugu_transcribe = pipeline(task='automatic-speech-recognition', model='vasista22/whisper-telugu-base', chunk_length_s=30, device=device)
# telugu_transcribe.model.config.forced_decoder_ids = telugu_transcribe.tokenizer.get_decoder_prompt_ids(language='te', task='transcribe')
>>>>>>> Stashed changes

tamil_transcribe = pipeline(task="automatic-speech-recognition", model="vasista22/whisper-tamil-small", chunk_length_s=30, device=device)
tamil_transcribe.model.config.forced_decoder_ids = tamil_transcribe.tokenizer.get_decoder_prompt_ids(language="ta", task="transcribe")

viet_transcribe = pipeline("automatic-speech-recognition", model="vinai/PhoWhisper-small", device=device)

bengali_transcribe = pipeline("automatic-speech-recognition", model="anuragshas/whisper-small-bn", device=device)

nepali_transcribe = pipeline("automatic-speech-recognition", model="DrishtiSharma/whisper-large-v2-hindi-to-nepali-transfer-learning-200-steps", device=device)

kinyarwanda_transcribe = pipeline("automatic-speech-recognition", model="mbazaNLP/Whisper-Small-Kinyarwanda", device=device)


# def transcribe(audio_path, language):
#     if language == 'Hindi':
#         return hindi_transcribe(audio_path)['text']
#     elif language == 'Telugu':
#         return telugu_transcribe(audio_path)['text']
#     else:
#         return base_model.transcribe(audio_path)['text']


if __name__ == '__main__':
    audio_folder = '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/audio'
    mturk_folder = '/Users/Justin Qiu/Desktop/senior_thesis/image-captioning-mturk/mturk_output/'
    previous_csv = 'processed_output/output_transcription_2.csv'
    output_csv = 'processed_output/output_transcription_cvqa_empty.csv'
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

<<<<<<< Updated upstream
            # if vocaroo_id not in existing_df['id'].values:
            #     pass
            #     # print(f'Processing {vocaroo_id}', flush=True)
            # else:
            #     # print(f'Skipping {vocaroo_id}', flush=True)
            #     continue

            audio_path = os.path.join(audio_folder, audio_file)
            # transcription = existing_df[existing_df['id'] == vocaroo_id]['transcription'].values
            try:
                transcription = transcribe(audio_path, language)
            except Exception as e:
                print(f'Error in {vocaroo_id}: {e}')
                continue
=======
            # skip if already exists or language is not Hindi or Telugu
            if vocaroo_id not in existing_df['id'].values:
                # print(f'MISSING {vocaroo_id}')
                pass
            else:
                # print(f'SKIPPING {vocaroo_id}')
                pass
                continue
            # if not (language == 'Hindi' or language == 'Telugu'):
            #     continue

            audio_path = os.path.join(audio_folder, audio_file)
            # transcription = transcribe(audio_path, language)
            transcription = ''
>>>>>>> Stashed changes
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
            print(f'Finished {vocaroo_id}; transcription: {transcription}', flush=True)

    new_df = pd.DataFrame(data)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=['id'], keep='last')
    combined_df.to_csv(output_csv, index=False)
    print(combined_df)
