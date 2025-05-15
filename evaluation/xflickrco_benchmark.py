# =============================================================
# generate_captions_xflickrco.py
# =============================================================
"""Generate multilingual captions for the xFlickrCO *test* split.

Usage
-----
$ python generate_captions_xflickrco.py --model gpt4o --rows_per_lang 100

The script cycles through each of the eight language configs (de, en, es, id,
ja, ru, tr, zh) in xFlickrCO, calls the chosen vision‑language model, and writes
**id,language,caption** rows compatible with the evaluator.
"""

import os
import argparse
import pathlib

os.environ.setdefault("HF_HOME", "/nlp/data/huggingface_cache")  # reuse local cache

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd

from evaluation.llava import get_llava_response
from evaluation.gpt4o import gpt4o_caption

# Languages present in xFlickrCO (same order as on the HF card)
XFCO_LANGS = ["de", "en", "es", "id", "ja", "ru", "tr", "zh"]

###############################################################################
# Helper                                                                     #
###############################################################################

def iter_test_split(lang_code: str):
    """Return an iterator over the *test* split for one language."""
    return load_dataset("floschne/xflickrco", split="lang_code")


def process_captions(model: str = "gpt4o", rows_per_lang: int = 100) -> None:
    """Run captioning model over every language split and dump CSV.

    Parameters
    ----------
    model : str
        Which model wrapper to use: "gpt4o" (default) or "llava".
    rows_per_lang : int
        Caption at most this many examples per language. Set to <=0 for no
        limit (process the full split).
    """

    out_csv = pathlib.Path("evaluation/output") / f"xflickrco_captions_{model}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("id,language,caption\n")

        for lang in XFCO_LANGS:
            ds = iter_test_split(lang)
            for i, ex in enumerate(tqdm(ds, desc=f"{model}:{lang}")):
                if rows_per_lang > 0 and i >= rows_per_lang:
                    break  # respect user limit

                img_id = ex["id"]
                img_identifier = ex.get("img_path") or ex.get("image_link") or img_id

                try:
                    if model == "gpt4o":
                        caption = gpt4o_caption(img_identifier, lang)
                    elif model == "llava":
                        caption = get_llava_response(img_identifier, lang)
                    else:
                        raise ValueError(f"Unsupported model: {model}")

                    row = {"id": img_id, "language": lang, "caption": caption}
                    pd.DataFrame([row]).to_csv(f, header=False, index=False)
                    f.flush()
                except Exception as e:
                    print("⚠️", img_identifier, e)
                    continue

    print(f"✨ Done. Rows written incrementally to {out_csv}")

###############################################################################
# CLI entry‑point
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate xFlickrCO captions")
    parser.add_argument("--model", choices=["gpt4o", "llava"], default="gpt4o")
    parser.add_argument("--rows_per_lang", type=int, default=100,
                        help="Process at most N examples per language (<=0 for all)")
    args = parser.parse_args()

    print(f"Processing captions with {args.model} … (max {args.rows_per_lang} rows per language)")
    process_captions(model=args.model, rows_per_lang=args.rows_per_lang)
