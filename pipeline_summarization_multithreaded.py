import csv
import os
import pandas as pd
import time
import threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from keys import openai_api_key

input_file = 'output_transcriptions.csv'
output_file = 'output_summarization.csv'
client = OpenAI(api_key=openai_api_key)

file_lock = threading.Lock()

def get_gpt_response(prompt):
    """Fetch GPT response for a given prompt."""
    try:
        response = client.chat.completions.create(
            messages=[{'role': 'user', 'content': prompt}],
            model='gpt-4',
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error with OpenAI request: {e}")
        return "ERROR_PROCESSING"

def process_group(group, completed_keys):
    """Process a single group (image_link, language) and get the summary."""
    image_link, language = group.iloc[0]['image_link'], group.iloc[0]['language']

    if (image_link, language) in completed_keys:
        print(f"Skipping already completed group: {image_link}, {language}")
        return None

    ids = group['id'].tolist()
    transcriptions = group['transcription'].tolist()
    distinct = group['culturally_distinct'].tolist()
    explanations = group['cultural_distinction_explanation'].tolist()

    if len(transcriptions) > 1:
        prompt = f"Summarize these transcriptions into a high-quality image caption in the same language as the transcriptions:\n\n{''.join(transcriptions)}"
    else:
        prompt = f"Enhance this transcription into a high-quality image caption in the same language as the transcription:\n\n{transcriptions[0]}"
    summary = get_gpt_response(prompt)
    summary = summary.replace("\n", " ").replace("\r", " ")

    return (image_link, language, ids, transcriptions, distinct, explanations, summary)

def write_to_csv(data, writer):
    """Thread-safe writing to the CSV file."""
    with file_lock:
        writer.writerow(data)
        print(f"Captioned group: {data[0]}, {data[1]}")

def main():
    df = pd.read_csv(input_file)
    grouped = df.groupby(['image_link', 'language'])
    if os.path.exists(output_file) and os.stat(output_file).st_size > 0:
        completed_df = pd.read_csv(output_file)
        completed_keys = set(zip(completed_df['image_link'], completed_df['language']))
    else:
        completed_keys = set()

    with open(output_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if os.stat(output_file).st_size == 0:
            writer.writerow(['image_link', 'language', 'ids', 'transcriptions', 'distinct', 'explanations', 'summary'])

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for _, group in grouped:
                futures.append(executor.submit(process_group, group, completed_keys))
            for future in as_completed(futures):
                result = future.result()
                if result:
                    write_to_csv(result, writer)
                # time.sleep(1)

if __name__ == "__main__":
    main()
