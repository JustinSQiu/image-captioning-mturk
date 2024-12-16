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
    'wedding', 'marriage', 'married', 'marry', 'bride', 'groom', 'honeymoon', 'vows', 'ceremony',
    'festival', 'parade', 'feast', 'carnival', 'new year', 'dance',
    'art', 'pottery', 'weaving',
    'food', 'cuisine', 'meal', 'feast',
    'clothing', 'dress', 'robe', 'gown', 'kimono', 'saree', 'kurta', 'hanbok', 'dirndl', 'lederhosen', 'beret', 'headdress', 'veil', 'poncho',
    'culture', 'cultures', 'cultural', 'tradition', 'traditions', 'traditional', 'sport', 'sports'
]


def match_whole_word(word_list, word):
    """
    This function checks if any word or phrase from the list matches any part
    of the input word or phrase as a whole word, case-insensitively.
    
    Parameters:
    - word_list: List of words or phrases to match
    - word: The word or phrase to search for in the list
    
    Returns:
    - True if any word or phrase from the list matches any part of the input word or phrase as a whole word, False otherwise
    """
    pattern = r'\b(?:' + '|'.join(map(re.escape, word_list)) + r')\b'
    return bool(re.search(pattern, word, re.IGNORECASE))


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
                translation = parts[1]
                if match_whole_word(culturally_distinct_words, translation):
                    data.append(
                        {'translated_english_word': translation, 'original_word': original_word, 'original_index': index, 'language': language}
                    )

df = pd.DataFrame(data)
output_path = 'mmid/output/filtered_translations_new.csv'
df.to_csv(output_path, index=False, encoding='utf-8')
