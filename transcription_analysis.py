import pandas as pd
import matplotlib.pyplot as plt
from langdetect import detect, lang_detect_exception

def check_lang(val):
    try:
        lang = detect(str(val))
    except lang_detect_exception.LangDetectException:
        lang = 'UNKNOWN'
    return lang


pd.options.display.max_rows = 100
pd.options.display.max_colwidth = 300
pd.options.display.max_columns = 100

df = pd.read_csv('/home1/j/jsq/dev/image-captioning-mturk/processed_output/output_transcription_cvqa.csv')

df = df[df['language'] == 'Hindi']
print(df['transcription'])
print(len(df))

# # given string, calculate percent of it that's alphabet
# def percent_alpha(s):
#     s = str(s)
#     return sum([c.isalpha() for c in s]) / len(s)

# # print the rows of the dataframe that have a transcription that is less than 50% alphabet
# for index, row in df.iterrows():
#     lang = check_lang(row['transcription'])
#     if lang != 'en':
#         print(lang)
#         print(row['transcription'])
#     elif percent_alpha(row['transcription']) < 0.6:
#         print(row['transcription'])