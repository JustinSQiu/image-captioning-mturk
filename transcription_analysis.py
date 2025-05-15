# import pandas as pd
# import matplotlib.pyplot as plt

# pd.options.display.max_rows = 200
# pd.options.display.max_colwidth = 300
# pd.options.display.max_columns = 100

# df = pd.read_csv('/home1/j/jsq/dev/image-captioning-mturk/processed_output/output_transcription_translated_manually_cleaned_final.csv')

# language_counts = df["language"].value_counts()
# print("Annotations per language:")
# print(language_counts)
# print(f"Total annotations: {len(df)}")

# grouped = df.groupby("image_link")
# print(f"Number of unique images: {grouped.ngroups}")

# multi_lang_groups = grouped.filter(lambda group: group["language"].nunique() > 1)
# # print(f"Number of images with at least two language annotations: {multi_lang_groups['image_link'].nunique()}")
# # print(f"Number of annotations in images with at least two language annotations: {len(multi_lang_groups)}")

# # three_lang_groups = grouped.filter(lambda group: group["language"].nunique() > 2)
# # print(f"Number of images with at least three language annotations: {three_lang_groups['image_link'].nunique()}")
# # print(f"Number of annotations in images with at least three language annotations: {len(three_lang_groups)}")

# # four_lang_groups = grouped.filter(lambda group: group["language"].nunique() > 3)
# # print(f"Number of images with at least four language annotations: {four_lang_groups['image_link'].nunique()}")
# # print(f"Number of annotations in images with at least four language annotations: {len(four_lang_groups)}")

# # sort by number of transcriptions that each image link got
# # counts = df.groupby("image_link").size().sort_values(ascending=False)
# # print(counts)

# stats_df = df.groupby("image_link").agg(
#     unique_languages    = ('language', 'nunique')
# ).sort_values(by='unique_languages', ascending=False)
# print(stats_df.head(200))

#!/usr/bin/env python
"""
Language-level statistics for
  justinsunqiu/multilingual_transcriptions_translated_raw

Outputs a pandas DataFrame with:
  language | annotation_count | median_word_count | median_char_count
You can copy-and-paste the two median columns into your LaTeX table
(and decide later whether to keep words or characters for each lang).
"""

from datasets import load_dataset, get_dataset_split_names
import pandas as pd
import numpy as np
from collections import defaultdict

DATASET_NAME = "justinsunqiu/multilingual_transcriptions_translated_raw"
TRANS_COL     = "transcription"      # change here if you want “translation”
LANG_COL      = "language"

# 1. Load *all* splits (train/validation/test) so nothing is missed
splits = get_dataset_split_names(DATASET_NAME)
ds_list = [load_dataset(DATASET_NAME, split=split) for split in splits]
ds = None
for d in ds_list:
    ds = d if ds is None else ds.concatenate(d)

# 2. Accumulate per-language statistics
records = defaultdict(list)

for ex in ds:
    lang = ex[LANG_COL]
    text = ex[TRANS_COL]

    # Skip empty or null transcriptions
    if not text:
        continue

    char_len  = len(text)
    word_len  = len(text.split())   # simple whitespace split
    records[lang].append((char_len, word_len))

# 3. Build result table
result_rows = []
for lang, lens in records.items():
    char_lens, word_lens = zip(*lens)
    result_rows.append(
        {
            "language":           lang,
            "annotation_count":   len(lens),
            "median_word_count":  int(np.median(word_lens)),
            "median_char_count":  int(np.median(char_lens)),
        }
    )

df = pd.DataFrame(result_rows).sort_values("annotation_count", ascending=False)

# 4. Display + save (CSV is handy for copy-and-paste)
print(df.to_string(index=False))
df.to_csv("language_stats.csv", index=False)
print("\nWrote language-level stats to language_stats.csv")
