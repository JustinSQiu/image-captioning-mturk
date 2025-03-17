import csv

# List of CSV files to check
csv_files = [
    'mturk_output/Batch_413468_batch_results (1).csv',
    'mturk_output/Batch_413478_batch_results (1).csv',
    'mturk_output/Batch_413643_batch_results.csv',
    'mturk_output/Batch_413644_batch_results (1).csv',
    'mturk_output/Batch_413647_batch_results (1).csv',
    'mturk_output/Batch_413648_batch_results.csv',
    'mturk_output/Batch_413649_batch_results (1).csv',
    'mturk_output/Batch_413654_batch_results.csv',
    'mturk_output/Batch_413655_batch_results.csv',
    'mturk_output/Batch_413658_batch_results (1).csv',
    'mturk_output/Batch_413660_batch_results (1).csv',
]

output_filename = "mturk_upload_files/CVQA_processed_1.csv"
all_urls = []

# Loop through each file in the list
for file_path in csv_files:
    with open(file_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            url = row.get("Input.image_url")
            if url and "cvqa/Chinese_images" not in url:
                all_urls.append(url)

print(len(all_urls))

# Optional: Remove duplicates while preserving order
all_urls = list(dict.fromkeys(all_urls))

print(len(all_urls))

# Write the URLs to the output file with a header
with open(output_filename, "w", encoding="utf-8") as outfile:
    outfile.write("image_url\n")
    for url in all_urls:
        outfile.write(f"{url}\n")

print(f"Extracted {len(all_urls)} image URLs to {output_filename}")
