#!/usr/bin/env python
# run_inference.py
import argparse, os, torch
from unsloth import FastVisionModel
from PIL import Image

from keys import hf_token
from finetune.data import (
    get_cvqa_dataset,
    get_multilingual_transcriptions_dataset,
    get_english_translated_transcriptions_dataset,
)

def load_eval_split(name: str):
    """Return just the *eval* part of the requested dataset."""
    if name == "cvqa":
        return get_cvqa_dataset()[1]                        # (train_ds, eval_ds)
    if name == "multilingual_transcriptions":
        return get_multilingual_transcriptions_dataset()[1]
    if name == "english_translated_transcriptions":
        return get_english_translated_transcriptions_dataset()[1]
    raise ValueError("Unknown dataset!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="outputs/full",
                        help="Local folder *or* hub repo (e.g. justinsunqiu/outputs/full)")
    parser.add_argument("--dataset", choices=[
        "cvqa", "multilingual_transcriptions", "english_translated_transcriptions"
    ], default="cvqa")
    parser.add_argument("--num_samples", type=int, default=2,
                        help="How many examples to run")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_dir,
        load_in_4bit=False,          # flip to True if you saved a 4‑bit checkpoint
        local_files_only=False,
    )
    model.to(device).eval()
    image_processor = model.get_vision_tower().image_processor  # convenience accessor

    eval_ds = load_eval_split(args.dataset)
    sample_ids = list(range(min(args.num_samples, len(eval_ds))))

    for i in sample_ids:
        example = eval_ds[i]
        img = example["image"] if isinstance(example["image"], Image.Image) \
              else Image.open(example["image"])

        prompt = (
            example.get("question")
            or example.get("caption")
            or example.get("transcription")
            or example.get("prompt")
        )
        if prompt is None:
            raise RuntimeError(f"Don’t know which field to use as prompt! Keys: {list(example.keys())}")

        pixel_values = image_processor(images=img, return_tensors="pt").pixel_values.to(device)
        input_ids    = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                max_new_tokens=64,
            )

        answer = tokenizer.decode(generated[0], skip_special_tokens=True)

        print(f"\n=== Sample {i} ===")
        print("Prompt:", prompt)
        print("Answer:", answer)
        print("-" * 60)

if __name__ == "__main__":
    main()
