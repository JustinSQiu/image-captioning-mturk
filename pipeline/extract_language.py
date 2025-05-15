import pandas as pd
from langdetect import detect, DetectorFactory
from tqdm import tqdm

# reproducible
DetectorFactory.seed = 0

df = pd.read_csv("processed_output/output_summarized_by_english_backtranslated_with_lang_final.csv")

print(df['detected_language'].value_counts())

# def detect_lang(text):
#     text = str(text).strip()
#     if not text:
#         return None
#     try:
#         return detect(text)
#     except:
#         return None

# # Wrap the Series in tqdm to show progress
# langs = []
# for txt in tqdm(df["backtranslation"], desc="Detecting languages"):
#     langs.append(detect_lang(txt))
# df["detected_language"] = langs

# df.to_csv("processed_output/output_summarized_by_english_backtranslated_with_lang_final.csv", index=False)
