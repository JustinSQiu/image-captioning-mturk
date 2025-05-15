import os
os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from helpers import get_gpt_response
from evaluation.llava import get_llava_response
from evaluation.gpt4o import gpt4o_caption
# from evaluation.llama import get_llama_response

ds = load_dataset("justinsunqiu/multilingual_transcriptions_final", split="test")

def process_captions(model="gpt4o"):
    OUT_CSV = f"evaluation/output/multilingual_captions_{model}.csv"
    with open(OUT_CSV, mode='w', encoding='utf-8') as f:
        f.write("id,image_url,language,caption\n")
        for ex in tqdm(ds, desc=f"captioning_{model}"):
            url = ex["image_link"] if "image_link" in ex else ex["ids"][0]
            try:
                if model == "gpt4o":
                    caption = gpt4o_caption(url, ex["language"])
                elif model == "llava":
                    caption = get_llava_response(url, ex["language"])
                else:
                    raise ValueError(f"Unsupported model: {model}")

                row = {
                    "id": ex["__index_level_0__"],
                    "image_url": url,
                    "language": ex.get("language", ""),
                    "caption": caption,
                }
                pd.DataFrame([row]).to_csv(f, header=False, index=False)
                f.flush()
            except Exception as e:
                print("⚠️", url, e)
                continue

    print(f"✨ Done. Wrote rows incrementally to {OUT_CSV}")

print("Processing gpt4o captions...")
process_captions("gpt4o")
# print("Processing llava captions...")
# process_captions("llava")
