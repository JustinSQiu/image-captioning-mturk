#!/usr/bin/env python3
"""
Caption any number of images from their URLs and save the results to a CSV file.

Example
-------
python caption_from_urls.py \
    --model_dir justinsunqiu/english_translated_llama_final \
    --device cuda:0 \
    --output_csv my_captions.csv
"""

import argparse
import csv
import io
from pathlib import Path
from typing import List

import torch
import requests
from PIL import Image
from transformers import TextStreamer
from unsloth import FastVisionModel
from datasets import load_dataset

# local helpers
from finetune.models import get_trained_model, get_llama_11b_model


def load_image_from_url(url: str, timeout: float = 10.0) -> Image.Image:
    """Download `url` and return a PIL.Image (RGB)."""
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def get_images_links() -> List[str]:
    """Load image links from the test split of the multilingual dataset."""
    dataset = load_dataset(
        "justinsunqiu/multilingual_transcriptions_translated_english_final",
        split="test",
    )
    return dataset["image_link"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="justinsunqiu/english_translated_llama_final",
        help="Local path or HF repo ID for the *(merged)* weights",
    )
    parser.add_argument(
        "--image_urls",
        nargs="+",
        help="One or more image URLs to caption. If omitted, uses the dataset links.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Where to run inference",
    )
    parser.add_argument(
        "--output_csv",
        default="processed_output/captions.csv",
        help="Path to save the <url, caption> rows (CSV)",
    )
    args = parser.parse_args()

    # --- Load model ------------------------------------------------------------------
    if args.model_dir == "llama-11b":
        model, tokenizer = get_llama_11b_model()
    else:
        model, tokenizer = get_trained_model(args.model_dir)
    FastVisionModel.for_inference(model)

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    # --- Determine image set ---------------------------------------------------------
    image_urls = args.image_urls if args.image_urls else get_images_links()
    if not image_urls:
        raise ValueError("No image URLs provided and dataset links unavailable.")

    # --- Caption each image ----------------------------------------------------------
    rows = []  # will store {"url": url, "caption": caption}

    instruction = "Write a detailed caption for this image."

    for url in image_urls:
        print(f"\n🔗  Captioning {url}...")
        try:
            image = load_image_from_url(url)
        except Exception as e:
            print(f"[WARN] Could not load {url}: {e}")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        # Tokenizer accepts PIL image directly (UnsLoTH vision wrapper handles it)
        inputs = tokenizer(
            image,
            chat_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(args.device)

        # Generate caption (also stream to console)
        output_ids = model.generate(
            **inputs,
            streamer=text_streamer,
            max_new_tokens=2048,
            temperature=1.2,
            min_p=0.1,
            use_cache=True,
        )

        # Decode only the new text (after the prompt)
        prompt_len = inputs["input_ids"].shape[-1]
        generated_caption = tokenizer.decode(
            output_ids[0][prompt_len:], skip_special_tokens=True
        ).strip()

        print(f"\n🖼️  Caption for {url}: {generated_caption}")

        rows.append({"url": url, "caption": generated_caption})

    # --- Save captions to CSV --------------------------------------------------------
    output_path = Path(args.output_csv).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "caption"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅  Saved {len(rows)} captions to {output_path.resolve()}")


if __name__ == "__main__":
    main()
