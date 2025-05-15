#!/usr/bin/env python
"""
Evaluate multilingual image–caption files against the
`justinsunqiu/multilingual_transcriptions_translated_english_final` baseline.

Usage
-----
python eval_captions.py pred1.csv pred2.csv ... --out_dir results
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
from datasets import load_dataset, concatenate_datasets     # datasets>=2.19
from tqdm.auto import tqdm            # tqdm>=4.66


def generated_caption_evaluation(
    caption_preds: List[str],
    caption_golds: List[str] | List[List[str]],
    lang_id: str,
    metrics: Set[str] | None = None,
    use_gpu: bool = True,
) -> Dict[str, float | str]:
    if metrics is None:
        metrics = {"bleu", "meteor", "rouge", "bertscore", "chrf"}

    # If some references have multiple gold captions, pad so every item has same len
    if isinstance(caption_golds[0], list):
        max_len = max(len(x) for x in caption_golds)
        for i in range(len(caption_golds)):
            caption_golds[i] = caption_golds[i] + [""] * (
                max_len - len(caption_golds[i])
            )

    results: Dict[str, float | str] = {}


    if len(caption_golds) == 0 or len(caption_preds) == 0:
        return results
    
    # print(caption_golds)
    # print(caption_preds)

    # make sure none of the captions in caption_golds or caption_preds has length 0
    tmp_caption_golds = [
        x for x in caption_golds if isinstance(x, str) and len(x.strip()) == 0
        or isinstance(x, list) and all(len(sub.strip()) == 0 for sub in x)  # Check all elements of list if it's a list
    ]

    tmp_caption_preds = [
        x for x in caption_preds if isinstance(x, str) and len(x.strip()) == 0
        or isinstance(x, list) and all(len(sub.strip()) == 0 for sub in x)  # Check all elements of list if it's a list
    ]

    if len(tmp_caption_golds) > 0 or len(tmp_caption_preds) > 0:
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

    # if "rouge" in metrics:
    #     rouge = evaluate.load("rouge")
    #     res = rouge.compute(predictions=caption_preds, references=caption_golds)
    #     for rouge_method, rouge_res in res.items():
    #         results[rouge_method] = rouge_res

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


def normalise_lang(tag: str) -> str:
    return 'en'
    """
    Convert various language strings (“English”, “en”, “EN-US”, ...) to two-letter ISO.
    Falls back to lower-case original when unknown.
    """
    try:
        return langcodes.find(tag).language
    except LookupError:
        return tag.lower()


def load_baseline() -> Dict[Tuple[str, str], List[str]]:
    """
    Flatten the HF dataset so we can do O(1) id+lang look-ups.

    This version loads both 'train' and 'test' splits.

    Returns
    -------
    dict keyed by (image_id, iso_lang) → list[str]  (one or more gold captions)
    """
    print("📥  Downloading baseline dataset …")
    
    # Load both train and test splits
    train_ds = load_dataset(
        "justinsunqiu/multilingual_transcriptions_final", 
        split="train"
    )
    test_ds = load_dataset(
        "justinsunqiu/multilingual_transcriptions_final", 
        split="test"
    )

    # Combine train and test splits
    combined_ds = concatenate_datasets([train_ds, test_ds])

    # Create the gold lookup dictionary
    gold_lookup: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    
    for row in tqdm(combined_ds, desc="Flatten baseline", unit="rows"):
        cap = row["summary"]
        iso_lang = normalise_lang(row["language"])
        # iso_lang = 'en'  # You could normalize lang if needed
        gold_lookup[(row["image_link"], iso_lang)].append(cap)

    return gold_lookup


def evaluate_csv(csv_path: pathlib.Path, gold_lookup: Dict[Tuple[str, str], List[str]]) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics per language for one prediction CSV.
    Returns {language_code: {metric_name: value, …}, …}
    """
    df = pd.read_csv(csv_path)
    # if not {"id", "language", "caption"}.issubset(df.columns):
    df.columns = ["id", "image_link", "language", "caption"]
        # raise ValueError(f"{csv_path} does not have required headers id,language,caption")
    print(df.columns)

    # Normalise language tags
    print(df["language"].unique())
    df["lang_iso"] = df["language"].map(normalise_lang)
    # df["lang_iso"] = 'en'

    results: Dict[str, Dict[str, float]] = {}

    for lang, group in tqdm(df.groupby("lang_iso"), desc=f"🔍 {csv_path.name}", unit="lang"):
        print(group)
        preds: List[str] = group["caption"].tolist()
        golds: List[List[str]] | List[str] = []

        # Collect gold captions, preserving row order
        missing = 0
        for img_url in group["image_link"]:
            key = (img_url, lang)
            if key in gold_lookup:
                golds.append(gold_lookup[key])
            else:
                # No reference → empty string so metric libraries skip gracefully
                golds.append([""])
                missing += 1

        if missing:
            print(f"⚠️  {missing} / {len(group)} rows in {lang} have no gold caption")

        if missing == len(group):
            print(f"⚠️  All {len(group)} rows in {lang} have no gold caption")
            continue

        metrics = generated_caption_evaluation(
            preds,
            golds,
            lang_id=lang,
            use_gpu=True,
        )
        results[lang] = metrics

    # Optionally compute “all languages together”
    preds_all = df["caption"].tolist()
    golds_all = [gold_lookup.get((row.image_link, row.lang_iso), [""]) for row in df.itertuples()]
    results["ALL"] = generated_caption_evaluation(preds_all, golds_all, "en", use_gpu=True)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multilingual caption CSVs")
    parser.add_argument("csvs", nargs="+", type=pathlib.Path, help="Prediction CSV files")
    parser.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("evaluation/output"),
                        help="Directory to write *.json result files (created if absent)")
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

