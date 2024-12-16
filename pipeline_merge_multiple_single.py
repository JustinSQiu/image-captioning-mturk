import pandas as pd

# Load the data
df = pd.read_csv('processed_output/output_summarization.csv')
multi_df = pd.read_csv('processed_output/output_summarization_multiple.csv')

# Filter out rows from df that are in multi_df, then combine with multi_df
result = (
    df.merge(multi_df[['image_link', 'language']], on=['image_link', 'language'], how='left', indicator=True)
      .query('_merge == "left_only"')  # Keep rows only in df
      .drop(columns=['_merge'])        # Drop the merge indicator column
)
result = pd.concat([result, multi_df], ignore_index=True)

# Save the result to a new CSV file
result.to_csv('processed_output/output_summarization_merged.csv', index=False)
