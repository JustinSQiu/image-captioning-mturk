import os
import re

import pandas as pd

folder_path = 'mmid-tools/dictionaries'

data = []

language_map = {
    'zh': 'Chinese',
    'es': 'Spanish',
    'hi': 'Hindi',
    'fr': 'French',
    'ko': 'Korean',
    'ja': 'Japanese',
    'de': 'German',
    'ar': 'Arabic',
    'ru': 'Russian',
}

culturally_distinct_words = [
    'wedding',
    'marriage',
    'marry',
    'funeral',
    'clothing',
    'celebrat',
    'food'
]

for file_name in os.listdir(folder_path):
    if 'dict' not in file_name:
        continue
    lang_code = file_name.split('.')[-1]
    language = language_map.get(lang_code, 'Unknown')
    if language == 'Unknown':
        continue
    file_path = os.path.join(folder_path, file_name)

    with open(file_path, 'r', encoding='utf-8') as file:
        for index, line in enumerate(file):
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                original_word = parts[0]
                translations = parts[1:]
                for translation in translations:
                    if any(substr in translation for substr in culturally_distinct_words):
                        data.append(
                            {'translated_english_word': translation, 'original_word': original_word, 'original_index': index + 1, 'language': language}
                        )

df = pd.DataFrame(data)
output_path = 'mmid/output/filtered_translations.csv'
df.to_csv(output_path, index=False, encoding='utf-8')
