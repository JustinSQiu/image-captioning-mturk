# Senior Thesis

Using multilingual dense captions to improve performance and cultural awareness of multimodal models

## Process for MMID Dataset

1. Run mmid_process_data.py to get all translations into a csv file. Modify culturally_distinct_words to include more/less
2. Run mmid_select_images.py to use the metadata files to download all relevant images from the internet. This is because the MMID dataset does not download at full quality.
3. Run mmid_filter_images.py to filter out problematic images
4. Upload the images now in mmid_images to Github
5. Run mmid_process_csv.py to generate the csvs to upload to Mturk