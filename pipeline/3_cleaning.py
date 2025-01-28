import pandas as pd
from langdetect import detect, lang_detect_exception

def check_lang(val):
    try:
        lang = detect(str(val))
    except lang_detect_exception.LangDetectException:
        lang = 'UNKNOWN'
    return lang

input_file = 'processed_output/output_transcription_full_2.csv'
output_file = 'processed_output/output_transcription_errors_2.csv'
cleaned_file = 'processed_output/output_transcription_cleaned_2.csv'

df = pd.read_csv(input_file)
error_rows = []
invalid_ids = set()

for index, row in df.iterrows():
    transcription = row['transcription']
    lang = check_lang(transcription)
    if row['language'] == 'English':
        if len(str(transcription)) < 300:
            error_rows.append({
                'id': row['id'], 
                'language': row['language'], 
                'culturally_distinct': row['culturally_distinct'], 
                'cultural_distinction_explanation': row['cultural_distinction_explanation'], 
                'vocaroo_link': row['vocaroo_link'], 
                'image_link': row['image_link'], 
                'transcription': transcription, 
                'response': 'Length', 
                'workerid': row['WorkerId']
            })
            invalid_ids.add(row['id'])
        elif lang != 'en':
            error_rows.append({
                'id': row['id'], 
                'language': row['language'], 
                'culturally_distinct': row['culturally_distinct'], 
                'cultural_distinction_explanation': row['cultural_distinction_explanation'], 
                'vocaroo_link': row['vocaroo_link'], 
                'image_link': row['image_link'], 
                'transcription': transcription, 
                'response': 'Language', 
                'workerid': row['WorkerId']
            })
            invalid_ids.add(row['id'])

    elif row['language'] == 'Chinese':
        if len(str(transcription)) < 50:
            error_rows.append({
                'id': row['id'], 
                'language': row['language'], 
                'culturally_distinct': row['culturally_distinct'], 
                'cultural_distinction_explanation': row['cultural_distinction_explanation'], 
                'vocaroo_link': row['vocaroo_link'], 
                'image_link': row['image_link'], 
                'transcription': transcription, 
                'response': 'Length', 
                'workerid': row['WorkerId']
            })
            invalid_ids.add(row['id'])
        elif not lang.startswith('zh') and lang != 'ko':
            error_rows.append({
                'id': row['id'], 
                'language': row['language'], 
                'culturally_distinct': row['culturally_distinct'], 
                'cultural_distinction_explanation': row['cultural_distinction_explanation'], 
                'vocaroo_link': row['vocaroo_link'], 
                'image_link': row['image_link'], 
                'transcription': transcription, 
                'response': 'Language', 
                'workerid': row['WorkerId']
            })
            invalid_ids.add(row['id'])

    elif row['language'] == 'French':
        error_rows.append({
            'id': row['id'], 
            'language': row['language'], 
            'culturally_distinct': row['culturally_distinct'], 
            'cultural_distinction_explanation': row['cultural_distinction_explanation'], 
            'vocaroo_link': row['vocaroo_link'], 
            'image_link': row['image_link'], 
            'transcription': transcription, 
            'response': 'Length', 
            'workerid': row['WorkerId']
        })
        invalid_ids.add(row['id'])

error_df = pd.DataFrame(error_rows)
error_df.to_csv(output_file, index=False)

cleaned_df = df[~df['id'].isin(invalid_ids)]
cleaned_df.to_csv(cleaned_file, index=False)