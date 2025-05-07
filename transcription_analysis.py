import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.max_rows = 200
pd.options.display.max_colwidth = 300
pd.options.display.max_columns = 100

df = pd.read_csv('/home1/j/jsq/dev/image-captioning-mturk/processed_output/output_transcription_translated_manually_cleaned_final.csv')

language_counts = df["language"].value_counts()
print("Annotations per language:")
print(language_counts)
print(f"Total annotations: {len(df)}")

grouped = df.groupby("image_link")
print(f"Number of unique images: {grouped.ngroups}")

multi_lang_groups = grouped.filter(lambda group: group["language"].nunique() > 1)
# print(f"Number of images with at least two language annotations: {multi_lang_groups['image_link'].nunique()}")
# print(f"Number of annotations in images with at least two language annotations: {len(multi_lang_groups)}")

# three_lang_groups = grouped.filter(lambda group: group["language"].nunique() > 2)
# print(f"Number of images with at least three language annotations: {three_lang_groups['image_link'].nunique()}")
# print(f"Number of annotations in images with at least three language annotations: {len(three_lang_groups)}")

# four_lang_groups = grouped.filter(lambda group: group["language"].nunique() > 3)
# print(f"Number of images with at least four language annotations: {four_lang_groups['image_link'].nunique()}")
# print(f"Number of annotations in images with at least four language annotations: {len(four_lang_groups)}")

# sort by number of transcriptions that each image link got
# counts = df.groupby("image_link").size().sort_values(ascending=False)
# print(counts)

stats_df = df.groupby("image_link").agg(
    unique_languages    = ('language', 'nunique')
).sort_values(by='unique_languages', ascending=False)
print(stats_df.head(200))
