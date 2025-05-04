"""
caption_cvqa.py
Generate one English caption per unique image in the afaji/cvqa dataset
and append results to processed_output/output_cvqa_captions_synthetic.csv
"""

import os, csv, base64
from io import BytesIO
from datasets import load_dataset
from PIL import Image
from helpers import get_gpt_response

OUTPUT_FILE = "processed_output/output_cvqa_captions_synthetic.csv"
MODEL_NAME  = "ft:gpt-4o-2024-08-06:iarpa-hiatus-pausit-team:summaries-rlhf:BSgLWUY0"

processed_ids = set()
if os.path.exists(OUTPUT_FILE) and os.stat(OUTPUT_FILE).st_size > 0:
    try:
        with open(OUTPUT_FILE, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)                       # skip header
            for row in reader:
                processed_ids.add(row[0])      # first column is image_id
    except Exception as e:
        print(f"Could not read existing output → starting fresh: {e}")

ds1 = load_dataset("afaji/cvqa", split="test", streaming=False)

ds2 = load_dataset("mair-lab/CulturalVQA", split="test", streaming=False)
ds2 = ds2.rename_column("u_id", "ID")

def pil_to_data_url(img: Image.Image) -> str:
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, mode="a", newline="", encoding="utf-8") as fout:
    writer = csv.writer(fout)
    if os.stat(OUTPUT_FILE).st_size == 0:
        writer.writerow(["image_id", "caption"])

    # for row in ds and ds2:
    for ds in [ds1, ds2]:
        for row in ds:
            full_id   = row["ID"]
            image_id  = full_id.split("_", 1)[0]

            if image_id in processed_ids:
                continue

            try:
                data_url = pil_to_data_url(row["image"])
                messages = [
                    {
                        "role": "user",
                        "content": [
                            { "type": "text",
                            "text": f"Provide a detailed caption of this image, with a strong focus on cultural elements but also an emphasis on all details." + f"The image is from the culture/language of {row['Subset']}; if useful, incorporate that in your caption." if row["Subset"] else "" },
                            { "type": "image_url",
                            "image_url": { "url": data_url } }
                        ]
                    }
                ]

                caption = get_gpt_response(messages=messages, model=MODEL_NAME, temperature=1.0, max_tokens=2048, top_p=1.00)
                caption = caption.strip().replace("\n", " ").replace("\r", " ")
                print(f"✓ {image_id}")

            except Exception as e:
                print(f"✗ {image_id}  —  {e}")
                caption = ""

            writer.writerow([image_id, caption])
            fout.flush()
