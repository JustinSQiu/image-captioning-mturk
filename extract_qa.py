import os
import re
import csv
import json
import pandas as pd
from helpers import get_gpt_response

INPUT_FILE  = 'processed_output/output_translated.csv'
OUTPUT_FILE = 'processed_output/output_facts_and_questions.csv'

# 1) load and group
df = pd.read_csv(INPUT_FILE)
df.dropna(subset=['transcription'], inplace=True)
groups = df.groupby('image_link')

# 2) resume progress
if os.path.exists(OUTPUT_FILE) and os.stat(OUTPUT_FILE).st_size > 0:
    done = set(pd.read_csv(OUTPUT_FILE)['image_link'])
else:
    done = set()

with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout)
    if os.stat(OUTPUT_FILE).st_size == 0:
        writer.writerow([
            'image_link',
            'native_transcription',
            'non_native_transcription',
            'facts_missing_in_non_native',
            'questions_per_fact'
        ])

    for image_link, group in groups:
        if image_link in done:
            continue

        # infer source language from the URL path
        m = re.search(r'/([^/]+)_images/', image_link)
        source_lang = m.group(1) if m else ''

        native_rows     = group[group['language'].str.lower() == source_lang.lower()]
        non_native_rows = group[group['language'].str.lower() != source_lang.lower()]

        if native_rows.empty or non_native_rows.empty:
            print(f"Skip {image_link}: need both native & non-native")
            continue

        native_text     = native_rows.iloc[0]['translation']
        non_native_text = non_native_rows.iloc[0]['translation']

        # --- STEP 1: Extract missing facts ---
        fact_prompt = [
            {
                'role': 'system',
                'content': (
                    "You have two English descriptions of the same image:\n"
                    "- NATIVE (detailed, from someone whose language matches the image source)\n"
                    "- NON-NATIVE (a translation)\n\n"
                    "List ONLY the individual factual statements that appear in NATIVE but are MISSING from NON-NATIVE.\n"
                    "Each fact must be as specific as possible (e.g. “the sari has gold embroidery”, not “distinct clothes”).\n"
                    "Return a JSON list of strings under the key `facts_missing_in_non_native`."
                )
            },
            {
                'role': 'user',
                'content': json.dumps({
                    'native_transcription':     native_text,
                    'non_native_transcription': non_native_text
                }, ensure_ascii=False)
            }
        ]

        try:
            fact_resp = get_gpt_response(fact_prompt).strip()
            fact_data = json.loads(fact_resp)
            facts = fact_data.get('facts_missing_in_non_native', [])
        except Exception as e:
            print(f"Error extracting facts for {image_link}: {e}")
            continue

        # --- STEP 2: For each fact, generate one question ---
        # we pass back the native transcription again for context
        questions_prompt = [
            {
                'role': 'system',
                'content': (
                    "Given an English DESCRIPTION of an image and a list of SPECIFIC facts, "
                    "write exactly one question per fact such that the question can be answered "
                    "only if you know that fact from the description.\n\n"
                    "Return a JSON object mapping each fact to its question, under the key `questions_per_fact`."
                )
            },
            {
                'role': 'user',
                'content': json.dumps({
                    'native_transcription': native_text,
                    'facts': facts
                }, ensure_ascii=False)
            }
        ]

        try:
            q_resp = get_gpt_response(questions_prompt).strip()
            q_data = json.loads(q_resp)
            questions_per_fact = q_data.get('questions_per_fact', {})
        except Exception as e:
            print(f"Error generating questions for {image_link}: {e}")
            continue

        # 3) write out
        writer.writerow([
            image_link,
            native_text,
            non_native_text,
            json.dumps(facts, ensure_ascii=False),
            json.dumps(questions_per_fact, ensure_ascii=False),
        ])
        fout.flush()
        print(f"Done {image_link}")
