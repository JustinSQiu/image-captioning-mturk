import pandas as pd

# Read the two input files
df1 = pd.read_csv("processed_output/inference_english_translated_llama_final_fixed.csv")      # columns: url, caption
df2 = pd.read_csv("processed_output/inference_llama_11b.csv")      # columns: url, caption
# replace newlines with something more readable in the csv
df1["caption"] = df1["caption"].str.replace("\n", " ", regex=False)
df2["caption"] = df2["caption"].str.replace("\n", " ", regex=False)

# Merge on the url column; rows that appear in both files are kept
merged = df1.merge(
    df2,
    on="url",
    how="inner",          # use "outer" if you want to keep unmatched rows
    suffixes=("_finetuned", "_base")   # gives caption1 and caption2
)
print(merged)

# Rename url -> url1 and arrange columns
merged = merged.rename(columns={"url": "url1"})[["url1", "caption_finetuned", "caption_base"]]

# Save
merged.to_csv("processed_output/inference_merged.csv", index=False)

print("✅  Wrote merged.csv")
