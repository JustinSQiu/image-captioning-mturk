import os
import csv
import pandas as pd
from helpers import get_gpt_response

# Define file paths
input_file = 'processed_output/output_transcription_filtered_final.csv'
output_file = 'processed_output/output_transcription_cleaned_multilingual_final.csv'

# Load the input CSV, preserving the original structure
df_in = pd.read_csv(input_file)

# Build a set of already processed row IDs based on the output file (if it exists)
processed_ids = set()
if os.path.exists(output_file) and os.stat(output_file).st_size > 0:
    try:
        df_out = pd.read_csv(output_file)
        # Use the "id" column from the CSV as the unique identifier
        processed_ids = set(df_out['id'].astype(str).tolist())
    except Exception as e:
        print(f"Error reading output file, starting fresh: {e}")

# Open the output file in append mode so that we write as we go
with open(output_file, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if os.stat(output_file).st_size == 0:
        header = df_in.columns.tolist() + ['cleaned_trancription']
        writer.writerow(header)

    # Process each row one by one
    for index, row in df_in.iterrows():
        row_id = str(row['id'])
        # Skip processing if this row was already handled
        if row_id in processed_ids:
            print(f"Row {row_id} already processed; skipping.")
            continue

        transcription = row['transcription']
        language = row['language']

        # Build GPT messages for cleaning
        messages = [
            {
                'role': 'system',
                'content': (
                    "Remove any hallucinations from the following transcription. Do not add any new information or change the style or content of the original transcription. Your output should strictly be a subset of the original transcription, with hallucinations removed. Only remove hallucinations from the end of the transcription. Keep the transcription in the same language as the original."
                )
            },
            {
                'role': 'user',
                'content': f"Transcription: This image shows an open durian fruit with its outer husk split apart, revealing an empty interior where the flesh was previously. We see two durian husks and a wooden table. We see the two durian husks are placed on the table, and the sharp, spiky edges of the husk are clearly visible. The durian husks have jagged edges and a thorny texture. The husk also appears fresh, indicating that the fruit was recently eaten. The wood table surface is visible. A dark, blurred area in the top part of the image, possibly a hand or a chair, is also visible. We see that the wooden table has a brown tone, giving a warm, natural feel to the image, and the inside of the husk is creamy white. The image suggests that the fruit was eaten in a casual dining setting, probably like a food court or a home. We also see that the durian has recently been consumed, so it leaves behind a little bit of a taste. We continue to discuss the ability of the going Bahntu Wentu-Schio or Steamed Spirits to be repetitive and fast."
            },
            {
                'role': 'assistant',
                'content': f"Transcription: This image shows an open durian fruit with its outer husk split apart, revealing an empty interior where the flesh was previously. We see two durian husks and a wooden table. We see the two durian husks are placed on the table, and the sharp, spiky edges of the husk are clearly visible. The durian husks have jagged edges and a thorny texture. The husk also appears fresh, indicating that the fruit was recently eaten. The wood table surface is visible. A dark, blurred area in the top part of the image, possibly a hand or a chair, is also visible. We see that the wooden table has a brown tone, giving a warm, natural feel to the image, and the inside of the husk is creamy white. The image suggests that the fruit was eaten in a casual dining setting, probably like a food court or a home. We also see that the durian has recently been consumed, so it leaves behind a little bit of a taste."
            },
            {
                'role': 'user',
                'content': f"Transcription: This picture is showing a plate. It looks like it's some sort of board game. And I've never seen this board game before, so I don't know what's the rule or how's the plate. But it seems like it's played by four persons. There's some small text on it, but I cannot read what that is. There's also some arrow on it. It may be showing the direction of the or the order of the plane. And since there's a four color and the board is divided four color and four section, there's a green, yellow, blue and red on it. And it seems like this game is designed for four persons playing together. And the board was sitting on top of the wood tables. And looks... Yeah. Looks like it's something like... And there's a four square on each corner of the board game. On the board of the board games. And looks like there's some red star and there's a other red square on the board. And in the center of the board, it seems like there's a foldable things that you can throw the board to then carry on. Yeah. And the place that you can do that actually kindness to. And you may see that one of the hardest thing is the była black этого , and it's kind of hard to do it on the card. Because sometimes it weakest one of it's kind of hard because when you start to play, not always you only have one really easy, right? And you probably hear this type of white ball sound."
            },
            {
                'role': 'assistant',
                'content': f"Transcription: This picture is showing a plate. It looks like it's some sort of board game. And I've never seen this board game before, so I don't know what's the rule or how's the plate. But it seems like it's played by four persons. There's some small text on it, but I cannot read what that is. There's also some arrow on it. It may be showing the direction of the or the order of the plane. And since there's a four color and the board is divided four color and four section, there's a green, yellow, blue and red on it. And it seems like this game is designed for four persons playing together. And the board was sitting on top of the wood tables. And looks... Yeah. Looks like it's something like... And there's a four square on each corner of the board game. On the board of the board games. And looks like there's some red star and there's a other red square on the board. And in the center of the board, it seems like there's a foldable things that you can throw the board to then carry on. Yeah."
            },
            {
                'role': 'user',
                'content': f"Transcription: {transcription}"
            }
        ]
        try:
            cleaned_transcription = get_gpt_response(messages)
            # Remove newline characters for a cleaner CSV output
            cleaned_transcription = cleaned_transcription.replace("\n", " ").replace("\r", " ")
            print(f"Row {row_id} cleaned successfully.")
        except Exception as e:
            print(f"Error cleaning row {row_id}: {e}")
            cleaned_transcription = ""

        # Write the original row plus the new cleaned_transcription column to the output CSV
        new_row = row.tolist() + [cleaned_transcription]
        writer.writerow(new_row)
        f.flush()  # Flush after every row to ensure progress is saved

print("Cleaning process complete.")
