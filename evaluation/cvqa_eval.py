import os
import argparse
import pandas as pd
from PIL import Image
import random
import torch

from unsloth import FastVisionModel
from transformers import TextStreamer

from finetune.models import get_trained_model, get_llama_11b_model
from finetune.data import get_cvqa_dataset

# -----------------------------------------------------------------------------
# Command-line flags
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="Run CVQA inference with configurable model and question format.")
parser.add_argument(
    "--model",
    default="llama_11b",
    help="Which model to run. 'llama_11b' loads the base 11B checkpoint,"
         " while 'trained' loads the fine-tuned English-translated model.")
parser.add_argument(
    "--use_translations",
    action="store_true",
    help="If set, use the translated questions and options; otherwise use the "
         "original language fields.")
parser.add_argument(
    "--max_attempts",
    type=int,
    default=5,
    help="Maximum attempts to generate a valid answer before falling back to a "
         "random guess.")
args = parser.parse_args()

# -----------------------------------------------------------------------------
# Environment and model loading
# -----------------------------------------------------------------------------
os.environ.setdefault("HF_HOME", "/nlp/data/huggingface_cache")
print("🤖  HuggingFace cache dir:", os.environ["HF_HOME"])

device = "cuda" if torch.cuda.is_available() else "cpu"
print("🚀  Using device:", device)

if args.model == "llama":
    model_tag = "llama_11b"
    model, tokenizer = get_llama_11b_model()
else:
    model_tag = args.model
    model, tokenizer = get_trained_model(model_tag)

FastVisionModel.for_inference(model)
print(f"📦  Loaded model '{model_tag}'.")

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
cvqa_dataset = get_cvqa_dataset(eval_only=True)
print(f"📚  Dataset size: {len(list(cvqa_dataset))} samples.")

q_field   = "Translated Question" if args.use_translations else "Question"
opt_field = "Translated Options"  if args.use_translations else "Options"
print(f"📝  Using question field '{q_field}' and option field '{opt_field}'.")

# -----------------------------------------------------------------------------
# Inference loop
# -----------------------------------------------------------------------------
rows = []
for sample in cvqa_dataset:
    sample_id = sample["ID"]
    img       = sample["image"]
    question  = sample[q_field]
    options   = [opt.strip() for opt in sample[opt_field]]
    norm_opts = [opt.lower().replace(" ", "").replace(".", "") for opt in options]

    # Build the chat-prompt
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": (f"{question}\nOptions: {options}\n"
                                           "Your output should ONLY contain exactly one of the four "
                                           "options, without any other text. Do NOT include 'transcription:' or anything other than exactly the text in the option.")}
        ]
    }]

    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(img, input_text,
                       add_special_tokens=False,
                       return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[-1]

    answer = None
    for attempt in range(1, args.max_attempts + 1):
        output_ids = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=attempt > 1,
            temperature=0.0 if attempt == 1 else 0.7,
            use_cache=True,
            min_p=0.1,
        )
        new_token_ids = output_ids[0, prompt_len:]
        raw_answer = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        norm_answer = raw_answer.lower().replace(" ", "").replace(".", "")
        if ':' in norm_answer:
            norm_answer = norm_answer.split(':', 1)[-1].strip()
        if norm_answer in norm_opts:
            answer = raw_answer  # keep original formatting
            pred   = norm_opts.index(norm_answer)
            break
        else:
            print(f"❌ Attempt {attempt}: Invalid answer '{raw_answer}' (norm: '{norm_answer}').")

    # Fallback
    if answer is None:
        pred   = random.randint(0, 3)
        answer = options[pred]
        print(f"⚠️  After {args.max_attempts} tries, no valid answer – picking random ({answer}).")

    # Logging
    print(f"ID: {sample_id}")
    print(f"Question: {question}")
    print(f"Options: {options}")
    print(f"Prediction: {answer}")
    print(f"Expected: {options[sample['Label']]}")
    print(f"Correctness: {pred == sample['Label']}")
    print("=" * 50)

    rows.append({"ID": sample_id, "Prediction": pred})

# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------
os.makedirs("processed_output", exist_ok=True)
mode_suffix = "translated" if args.use_translations else "orig"
output_path = f"processed_output/dca_{model_tag}_{mode_suffix}_submission.csv"

pd.DataFrame(rows).to_csv(output_path, index=False)
print(f"➡️  Saved {output_path} with {len(rows)} rows")
