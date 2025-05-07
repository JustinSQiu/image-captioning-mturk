import os
import re
import csv
import json
import pandas as pd
from helpers import get_gpt_response

INPUT_FILE  = 'processed_output/output_summarized_by_type_final.csv'
OUTPUT_FILE = 'processed_output/output_vqa_final.csv'

df = pd.read_csv(INPUT_FILE)
groups = df.groupby('image_link')

if os.path.exists(OUTPUT_FILE) and os.stat(OUTPUT_FILE).st_size > 0:
    done = set(pd.read_csv(OUTPUT_FILE)['image_link'])
else:
    done = set()

def build_prompt(native_text: str, non_native_text: str):
    """
    Construct a one‑shot prompt that shows the model what we expect.
    """
    example_native = (
        "A bustling Holi festival scene in northern India, where revelers "
        "throw bright powders while a temple with distinctive shikhara towers "
        "looms in the background."
    )
    example_non_native = (
        "People are celebrating with colorful powder in front of a building."
    )

    example_qa = {
        "qa_pairs": [
            {
                "question": "What Hindu festival is being celebrated with colored powders?",
                "answer": "Holi"
            },
            {
                "question": "What religious structure with shikhara towers is visible behind the crowd?",
                "answer": "a temple"
            }
        ]
    }

    return [
        {
            "role": "system",
            "content": (
                "You receive two English descriptions of the **same image**:\n"
                "• **NATIVE** – detailed, culturally nuanced\n"
                "• **NON‑NATIVE** – simpler, may miss cultural cues\n\n"
                "Produce high‑quality visual‑question‑answer (VQA) data **directly**:\n"
                "Return JSON of the form:\n"
                "{'qa_pairs': [{'question': ..., 'answer': ...}, ...]}\n\n"
                "Guidelines:\n"
                "• Questions must target cultural or visually specific details **present in NATIVE but absent from NON‑NATIVE**.\n"
                "• Keep questions concise and unambiguous.\n"
                "• Answers should be short noun phrases (≤ 5 words) with correct spelling and capitalization.\n"
                "• Do **not** invent details—stay faithful to the NATIVE caption.\n"
                "• 1 – 3 QAs per image is typical; skip if none qualify."
                "• Do **not** include QA that doesn't include cultural detail. For example, asking 'what color is the sky?' and answering 'blue' or asking 'where is this animal living' and answering 'it's natural habitat' both do not qualify as culturally distinct. Be very selective!\n\n"
            )
        },
        # ---------- ONE‑SHOT EXAMPLE ----------
        {
            "role": "user",
            "content": json.dumps(
                {
                    "native": example_native,
                    "non_native": example_non_native
                },
                ensure_ascii=False,
            )
        },
        {
            "role": "assistant",
            "content": json.dumps(example_qa, ensure_ascii=False)
        },
        # ---------- ACTUAL TASK ----------
        {
            "role": "user",
            "content": json.dumps(
                {
                    "native": native_text,
                    "non_native": non_native_text
                },
                ensure_ascii=False,
            )
        },
    ]

with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as fout:
    writer = csv.writer(fout)
    if os.stat(OUTPUT_FILE).st_size == 0:
        writer.writerow(['image_link', 'question', 'answer'])

    for image_link, group in groups:
        if image_link in done:
            continue

        # Expect exactly one row of each type
        native_row     = group[group['annotation_type'] == 'native']
        non_native_row = group[group['annotation_type'] == 'nonnative']
        if native_row.empty or non_native_row.empty:
            continue

        native_text     = native_row['summary'].iloc[0]
        non_native_text = non_native_row['summary'].iloc[0]

        # Build prompt and query the model
        prompt = build_prompt(native_text, non_native_text)
        try:
            raw = get_gpt_response(prompt).strip()
            raw = re.sub(r"^```(\w+)?|```$", "", raw).strip()  # remove optional code fences
            data = json.loads(raw)
            qa_pairs = data.get('qa_pairs', [])
        except Exception as err:
            # If parsing fails, skip this image to keep pipeline robust
            print(f"Skipping {image_link}: {err}")
            continue

        # Write one CSV row per QA pair
        for pair in qa_pairs:
            question = pair.get('question', '').strip()
            answer   = pair.get('answer', '').strip()
            if question and answer:
                writer.writerow([image_link, question, answer])
        fout.flush()
