#!/usr/bin/env python
"""
Evaluate multilingual image–caption files against the
`floschne/xflickrco` reference captions.

The CLI usage is intentionally the same as the original script so that
existing evaluation pipelines keep working:

$ python eval_captions_xflickrco.py pred1.csv pred2.csv ... --out_dir results

Differences to the original version
-----------------------------------
* Uses the *xFlickrCO* dataset instead of *multilingual_transcriptions*.
* The reference dataset is organised **per‑language split** (``en``, ``de`` …),
  so we iterate over all available splits and flatten them into a single
  ``dict`` keyed by ``(image_id, lang) → [captions]``.
* Each row in xFlickrCO contains the *image bytes*; we never touch them –
  we only need the ``id`` and the list of ``sentences``.
* Prediction CSVs now need **just three columns** – ``id``, ``language``
  (BCP‑47 or readable language name; we normalise to ISO‑639‑1), and
  ``caption`` – because the Flickr/COCO image ID uniquely identifies the
  picture.
"""
from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import evaluate                       # evaluate==0.4.0
import langcodes                      # langcodes==3.3
import numpy as np                    # numpy>=1.20
import pandas as pd                   # pandas>=1.5
from datasets import load_dataset, get_dataset_split_names  # datasets>=2.19
from tqdm.auto import tqdm            # tqdm>=4.66


# ---------------------------------------------------------------------------
# Metric helper
# ---------------------------------------------------------------------------


def generated_caption_evaluation(
    caption_preds: List[str],
    caption_golds: List[str] | List[List[str]],
    lang_id: str,
    metrics: Set[str] | None = None,
    use_gpu: bool = True,
) -> Dict[str, float | str]:
    """Exactly as in the original script – copied verbatim."""
    if metrics is None:
        metrics = {"bleu", "meteor", "rouge", "bertscore", "chrf"}

    if isinstance(caption_golds[0], list):
        max_len = max(len(x) for x in caption_golds)
        for i in range(len(caption_golds)):
            caption_golds[i] = caption_golds[i] + [""] * (
                max_len - len(caption_golds[i])
            )

    results: Dict[str, float | str] = {}

    if len(caption_golds) == 0 or len(caption_preds) == 0:
        return results

    # Skip rows where *all* refs or the prediction are empty
    has_empty_gold = [
        isinstance(x, str) and len(x.strip()) == 0
        or isinstance(x, list) and all(len(sub.strip()) == 0 for sub in x)
        for x in caption_golds
    ]
    has_empty_pred = [
        isinstance(x, str) and len(x.strip()) == 0
        or isinstance(x, list) and all(len(sub.strip()) == 0 for sub in x)
        for x in caption_preds
    ]
    if any(has_empty_gold) or any(has_empty_pred):
        return results

    if "bleu" in metrics:
        bleu = evaluate.load("bleu")
        for bleu_n in [1, 2, 3, 4]:
            res = bleu.compute(
                predictions=caption_preds,
                references=caption_golds,
                max_order=bleu_n,
            )
            results[f"bleu_{bleu_n}"] = res["bleu"]

    if "meteor" in metrics:
        meteor = evaluate.load("meteor")
        res = meteor.compute(predictions=caption_preds, references=caption_golds)
        results["meteor"] = res["meteor"]

    if "chrf" in metrics:
        chrf = evaluate.load("chrf")
        for m, wo in {"": 0, "+": 1, "++": 2}.items():
            res = chrf.compute(
                predictions=caption_preds,
                references=caption_golds,
                word_order=wo,
            )
            results[f"chrF{m}"] = res["score"]

    if "bertscore" in metrics:
        bertscore = evaluate.load("bertscore", device="cuda:0" if use_gpu else "cpu")
        # HF’s bert-score expects two-letter ISO for some languages
        if lang_id == "fil":
            lang_id = "tl"
        elif lang_id == "quz":
            lang_id = "qu"
        res = bertscore.compute(
            predictions=caption_preds,
            references=caption_golds,
            lang=lang_id,
            device="cuda:0" if use_gpu else "cpu",
        )
        for r in ["precision", "recall", "f1"]:
            results[f"avg_bertscore_{r}"] = float(np.mean(res[r]))
            results[f"std_bertscore_{r}"] = float(np.std(res[r]))

    if "cider" in metrics or "spice" in metrics:
        raise NotImplementedError("CIDEr and SPICE are not supported")
    return results


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalise_lang(tag: str) -> str:
    """Same logic as before – map anything to an ISO‑639‑1 code if possible."""
    try:
        return langcodes.find(tag).language
    except LookupError:
        return tag.lower()


# ---------------------------------------------------------------------------
# Load the xFlickrCO reference captions and flatten ✨
# ---------------------------------------------------------------------------

def load_baseline() -> Dict[Tuple[str, str], List[str]]:
    """Return a dict keyed by ``(image_id, iso_lang) → list[str]``."""
    print("📥  Downloading xFlickrCO dataset …")

    # The dataset stores **separate splits** for each language (e.g. 'en', 'de', …).
    # We fetch all available split names directly from the Hub instead of relying
    # on a (non‑existent) builder config.
    available_langs = get_dataset_split_names("floschne/xflickrco")

    # Sort and report the splits we are going to iterate over.
    available_langs = sorted(available_langs)
    print("🔤  Found language splits:", ", ".join(available_langs))

    gold_lookup: Dict[Tuple[str, str], List[str]] = defaultdict(list)

    for lang_split in available_langs:
        print(f"  • Loading split '{lang_split}' …")
        ds = load_dataset("floschne/xflickrco", split=lang_split)
        iso_lang = normalise_lang(lang_split)
        for row in tqdm(ds, desc=lang_split, unit="rows"):
            img_id = str(row["id"])
            captions = row["sentences"]
            if isinstance(captions, str):
                captions = [captions]
            gold_lookup[(img_id, iso_lang)].extend(captions)

    return gold_lookup


# ---------------------------------------------------------------------------
# Evaluation of one prediction CSV
# ---------------------------------------------------------------------------

def evaluate_csv(csv_path: pathlib.Path, gold_lookup: Dict[Tuple[str, str], List[str]]) -> Dict[str, Dict[str, float]]:
    """Compute metrics per language for *one* prediction CSV."""
    df = pd.read_csv(csv_path)

    # Ensure at least these three columns exist – ignore any extras.
    required = {"id", "language", "caption"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"{csv_path} must have at least the columns: id, language, caption; "
            f"found {list(df.columns)}"
        )

    # Normalise language tags to ISO‑639‑1
    df["lang_iso"] = df["language"].map(normalise_lang)

    results: Dict[str, Dict[str, float]] = {}

    for lang, group in tqdm(df.groupby("lang_iso"), desc=f"🔍 {csv_path.name}", unit="lang"):
        preds: List[str] = group["caption"].tolist()
        golds: List[List[str]] | List[str] = []

        # Collect gold captions for each *image id* in the same row order
        missing = 0
        for img_id in group["id"]:
            key = (str(img_id), lang)
            if key in gold_lookup:
                golds.append(gold_lookup[key])
            else:
                golds.append([""])  # graceful skip
                missing += 1

        if missing:
            print(f"⚠️  {missing} / {len(group)} rows in {lang} have no gold caption")
        if missing == len(group):
            print(f"⚠️  All rows in {lang} are missing – skipping this language")
            continue

        metrics = generated_caption_evaluation(
            preds,
            golds,
            lang_id=lang,
            use_gpu=True,
        )
        results[lang] = metrics

    # Optionally compute across *all* languages as a rough overall score
    preds_all = df["caption"].tolist()
    golds_all = [gold_lookup.get((str(row.id), row.lang_iso), [""]) for row in df.itertuples()]
    results["ALL"] = generated_caption_evaluation(preds_all, golds_all, "en", use_gpu=True)

    return results


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multilingual caption CSVs against xFlickrCO")
    parser.add_argument("csvs", nargs="+", type=pathlib.Path, help="Prediction CSV files")
    parser.add_argument(
        "--out_dir",
        type=pathlib.Path,
        default=pathlib.Path("evaluation/output"),
        help="Directory to write *.json result files (created if absent)",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gold_lookup = load_baseline()

    for csv_path in args.csvs:
        print(f"\n📊  Evaluating {csv_path} …")
        metrics = evaluate_csv(csv_path, gold_lookup)
        out_file = args.out_dir / f"{csv_path.stem}_metrics.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"✅  Wrote {out_file}")


if __name__ == "__main__":
    main()
