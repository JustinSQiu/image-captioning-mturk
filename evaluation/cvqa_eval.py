import os
import pandas as pd
from datasets import load_dataset
from PIL import Image
import requests
from unsloth import FastVisionModel
from transformers import TextStreamer
import torch

# Set HF cache location
os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"
hf_home = os.environ["HF_HOME"]

# Load dataset
cvqa_test = load_dataset("afaji/cvqa", split="test")

# def to_pillow(examples):
#     urls = examples['image']
#     images = []
#     for url in urls:
#         try:
#             image = Image.open(requests.get(url, stream=True).raw)
#             images.append(image)
#         except Exception as e:
#             print(f"Error loading image from {url}: {e}")
#             raise e
    
#     examples['image'] = images
#     return examples

# Preload all images and add to dataset
# cvqa_test = cvqa_test.map(to_pillow, batched=True, num_proc=1)

# Load model
model_name = "finetune/models/cvqa_only"
model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=True,
    local_files_only=False,
)
FastVisionModel.for_inference(model)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Run inference
rows = []
for sample in cvqa_test:
    sample_id = sample["ID"]
    img = sample["image"]
    if img is None:
        print(f"⚠️  Skipping {sample_id} (image load failed)")
        continue

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "You are a helpful assistant that answers questions about images. You will be given a question and a list of four multiple choice answers. You need to select the correct one. Your output should only consist of one singular number between 0 and 3 indicating the index of the correct answer. Do not include any other text.\nQuestion: " + sample["Question"] + "\nOptions: " + str(sample["Options"])}
        ]}
    ]
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        img,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)

    output = model.generate(**inputs, max_new_tokens=4)
    text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    print(f"✓ {sample_id} → {text}", flush=True)

    try:
        pred = int(text)
        if pred not in {0, 1, 2, 3}:
            raise ValueError
    except:
        print(f"⚠️  Got unexpected output for {sample_id}: “{text}” — setting to 0")
        pred = 0

    rows.append({"ID": sample_id, "Prediction": pred})

# Save results
df = pd.DataFrame(rows)
df.to_csv("submission.csv", index=False)
print("➡️  Saved submission.csv with", len(df), "rows")
