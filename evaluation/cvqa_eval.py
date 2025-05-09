import os
import pandas as pd
from datasets import load_dataset
from PIL import Image
import requests
from unsloth import FastVisionModel
from transformers import TextStreamer
import torch

from finetune.models import get_trained_model
from finetune.data import get_cvqa_dataset

os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"
hf_home = os.environ["HF_HOME"]

cvqa_dataset = get_cvqa_dataset(eval_only=True)

model, tokenizer = get_trained_model('english_translated_llama_vqa_final')
FastVisionModel.for_inference(model)

device = "cuda" if torch.cuda.is_available() else "cpu"

rows = []
for sample in cvqa_dataset:
    sample_id = sample["ID"]
    img = sample["image"]
    sample["Translated Options"] = list(map(lambda x: x.strip(), sample["Translated Options"]))

    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": sample["Translated Question"] + "\nOptions: " + str(sample["Translated Options"]) + "\nYour output should only contain exactly one of the four options, without any other text. Your answer must exactly match one of the four options."},
        ]}
    ]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
    inputs = tokenizer(
        img,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device)
    input_ids = inputs["input_ids"]
    prompt_len = input_ids.shape[-1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,
        temperature=0.0,
        min_p=0.1,
    )
    new_token_ids = output_ids[0, prompt_len:]
    answer = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

    print(f"ID: {sample_id}")
    print(f"Question: {sample['Translated Question']}")
    print(f"Options: {sample['Translated Options']}")
    print(f"Prediction: {answer}")

    try:
        pred = sample["Translated Options"].index(answer)
    except:
        print(f"⚠️  Got unexpected output for {sample_id}: “{answer}” — setting to -1")
        pred = -1
    
    print("=" * 50)
    rows.append({"ID": sample_id, "Prediction": pred})

# Save results
df = pd.DataFrame(rows)
df.to_csv("submission.csv", index=False)
print("➡️  Saved submission.csv with", len(df), "rows")
