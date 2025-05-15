###############################################################################
# vision_generate_xflickrco.py  (ADAPTED)                                     #
###############################################################################
"""Generate captions with a *local* FastVision/Llama model for the xFlickrCO
benchmark.

This is a direct port of the original `multilingual_transcriptions` script: we
just swap in the xFlickrCO dataset and add a `--rows_per_lang` limit.

Usage
-----
$ python vision_generate_xflickrco.py --model llama --rows_per_lang 100

The CSV will be written incrementally to:
    evaluation/output/xflickrco_<model>.csv
with the header:
    id,language,caption
"""

import argparse
import base64
import csv
import io
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from PIL import Image
from transformers import TextStreamer
from unsloth import FastVisionModel

from finetune.models import get_llama_11b_model, get_trained_model

# Languages present on the HF card
XFCO_LANGS = ["de", "en", "es", "id", "ja", "ru", "tr", "zh"]

###############################################################################
# Helper functions                                                            #
###############################################################################

def ensure_pil(img: Any) -> Image.Image:
    """Return a ‑RGB‑ PIL.Image regardless of how *img* is stored.

    The xFlickrCO dataset may provide either

    * a ready‑decoded PIL.Image, or
    * a mapping `{"bytes": <base64‑or‑bytes>}`.

    We normalise both cases here so that the rest of the code can always pass
    a PIL image to `tokenizer(...)`.
    """
    if isinstance(img, Image.Image):
        return img.convert("RGB")

    if isinstance(img, dict) and "bytes" in img:
        data = img["bytes"]
    else:
        data = img  # bytes or base64‑string

    if isinstance(data, str):
        data = base64.b64decode(data)

    if isinstance(data, bytes):
        return Image.open(io.BytesIO(data)).convert("RGB")

    raise TypeError(f"Cannot convert type {type(img)} to PIL.Image")

def load_lang_split(lang_code: str, rows: int):
    """Return an *iterator* over `rows` examples of the xFlickrCO test split.
    If ``rows <= 0`` we stream the full split.
    Each example is mapped through ``to_pillow`` so FastVisionModel can accept
    the ``image`` column.
    """
    ds = load_dataset("floschne/xflickrco", split=lang_code)
    # rename 
    # ds = ds.map(to_pillow, batched=True, num_proc=4)
    if rows > 0:
        ds = ds.select(range(min(rows, len(ds))))
    return ds


def build_prompt(lang_code: str) -> str:
    lang_code = lang_code.lower()
    """Return a language‑appropriate captioning prompt (8 langs of xFlickrCO)."""
    prompts = {
        "en": "Write a detailed caption for this image.",
        "de": "Schreibe eine detaillierte Bildunterschrift für dieses Bild.",
        "es": "Escribe un título detallado para esta imagen.",
        "id": "Tulis keterangan mendetail untuk gambar ini.",
        "ja": "この画像の詳細なキャプションを書いてください。",
        "ru": "Напишите подробную подпись к этому изображению.",
        "tr": "Bu resim için ayrıntılı bir başlık yazın.",
        "zh": "为这张图片写一个详细的标题。",
    }
    return prompts.get(lang_code.lower(), prompts["en"])

###############################################################################
# Main                                                                        #
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Local V‑LM captioning for xFlickrCO")
    parser.add_argument("--model", default="llama",
                        help="Either 'llama' (11B check‑point) or the name of a fine‑tuned model")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rows_per_lang", type=int, default=100,
                        help="Process at most N examples per language (<=0 for all)")
    args = parser.parse_args()

    out_csv = Path("evaluation/output") / f"xflickrco_{args.model}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load model
    if args.model == "llama":
        model, tokenizer = get_llama_11b_model()
    else:
        model, tokenizer = get_trained_model(args.model)
    FastVisionModel.for_inference(model)
    model.to(args.device).eval()
    streamer = TextStreamer(tokenizer, skip_prompt=True)

    # 2. Prepare CSV writer
    first_write = not out_csv.exists()
    fp = out_csv.open("a", newline="", encoding="utf-8")
    writer = csv.writer(fp)
    if first_write:
        writer.writerow(["id", "language", "caption"])  # no image_url needed for evaluator

    # 3. Iterate languages
    for lang in XFCO_LANGS:
        print(f"\n=== {lang} ({args.rows_per_lang if args.rows_per_lang>0 else 'all'}) ===")
        ds = load_lang_split(lang, args.rows_per_lang)

        for idx, sample in enumerate(ds):
            prompt = build_prompt(lang)
            print(f"[{idx+1}/{len(ds)}] {sample['id']} {lang} → {prompt}")
            messages = [
                {"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]}
            ]
            chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

            inputs = tokenizer(
                ensure_pil(sample["image"]),
                chat_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(args.device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    streamer       = streamer,
                    max_new_tokens = 2048,
                    temperature    = 1.2,
                    min_p          = 0.1,
                    use_cache      = True,
                )
            caption = tokenizer.decode(
                output_ids[0, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()

            writer.writerow([sample["id"], lang, caption])
            fp.flush()
            print(f"[{idx+1}/{len(ds)}] ✓")

    fp.close()
    print(f"\nFinished – CSV saved to {out_csv.resolve()}")


if __name__ == "__main__":
    main()
