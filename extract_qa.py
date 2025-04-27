import os
import re
import csv
import json
import pandas as pd
from helpers import get_gpt_response

INPUT_FILE  = 'processed_output/output_summarized_by_type.csv'
OUTPUT_FILE = 'processed_output/output_facts_and_questions.csv'

df = pd.read_csv(INPUT_FILE)
groups = df.groupby('image_link')

if os.path.exists(OUTPUT_FILE) and os.stat(OUTPUT_FILE).st_size > 0:
    done = set(pd.read_csv(OUTPUT_FILE)['image_link'])
else:
    done = set()

with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout)
    if os.stat(OUTPUT_FILE).st_size == 0:
        writer.writerow(['image_link', 'question', 'answer'])

    for image_link, group in groups:
        if image_link in done:
            continue

        m = re.search(r'.*/([^/]+)_images/', image_link)
        source_lang = m.group(1) if m else ''

        native_row     = group[group['annotation_type'] == 'native']
        non_native_row = group[group['annotation_type'] == 'nonnative']
        if native_row.empty or non_native_row.empty:
            continue

        native_text     = native_row['summary'].iloc[0]
        non_native_text = non_native_row['summary'].iloc[0]

        # 1: Generate cultural facts
        fact_prompt = [
            {'role': 'system', 'content': (
                "You have two English descriptions of the same image:\n"
                "- NATIVE (detailed, cultural)\n"
                "- NON-NATIVE (possibly lacking culture)\n\n"
                "List the culturally distinct statements in NATIVE that are MISSING in NON-NATIVE."
                " Return JSON: {'cultural_facts': [ ... ]}."
            )},
            {'role': 'user', 'content': json.dumps({
                'native_transcription': native_text,
                'non_native_transcription': non_native_text
            }, ensure_ascii=False)}
        ]
        try:
            raw = get_gpt_response(fact_prompt).strip()
            raw = re.sub(r"^```(\w+)?|```$", "", raw).strip()
            facts = json.loads(raw).get('cultural_facts', [])
        except Exception:
            continue

        # 2: Generate question-answer pairs
        qa_prompt = [
            {'role': 'system', 'content': (
                "Given an English description of an image and a list of specific facts,"
                " write one concise question-answer pair per fact for a VQA dataset. Remember that the dataset is intended for visual question answering, and there will be an image associated with each question. However, since you don't have the image, you should only use the text to generate the question and answer; just keep in mind that the question is being framed with the image."
                " Questions should be clear (e.g. 'What does the structure resemble?')"
                " and answers brief and clear (e.g. 'a minaret', 'holi')."
                " Return JSON: {'qa_pairs': [{ 'fact': ..., 'question': ..., 'answer': ... }, ...]}."
            )},
            {'role': 'user', 'content': json.dumps({
                'native_transcription': native_text,
                'facts': facts
            }, ensure_ascii=False)}
        ]
        try:
            qa_raw = get_gpt_response(qa_prompt).strip()
            qa_raw = re.sub(r"^```(\w+)?|```$", "", qa_raw).strip()
            qa_data = json.loads(qa_raw)
            qa_pairs = qa_data.get('qa_pairs', [])
        except Exception:
            continue

        # 3: Filter out culturally distinct facts
        filter_prompt = [
            {'role': 'system', 'content': (
                "Given the native transcription of the image and a list of question-answer pairs with their facts,"
                " remove any pair whose question is not culturally distinct or cannot be answered from the native transcription. You can be very selective."
                " Return JSON list of facts to KEEP under 'filtered_facts'."
            )},
            {'role': 'user', 'content': json.dumps({
                'native_transcription': native_text,
                'qa_pairs': qa_pairs
            }, ensure_ascii=False)}
        ]
        try:
            f_raw = get_gpt_response(filter_prompt).strip()
            f_raw = re.sub(r"^```(\w+)?|```$", "", f_raw).strip()
            f_data = json.loads(f_raw)
            keep = set(f_data.get('filtered_facts', []))
        except Exception:
            keep = set([pair['fact'] for pair in qa_pairs])

        for pair in qa_pairs:
            fact = pair.get('fact')
            if fact not in keep:
                continue
            question = pair.get('question')
            answer = pair.get('answer')
            writer.writerow([image_link, question, answer])
        fout.flush()
