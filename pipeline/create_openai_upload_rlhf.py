#!/usr/bin/env python3
import json
from collections import defaultdict
from datasets import load_dataset

def main(split="train"):
    ds = load_dataset(
        "justinsunqiu/multilingual_transcriptions_summarized_by_native_nonnative",
        split="train"
    )
    groups = defaultdict(list)
    for ex in ds:
        groups[ex["image_link"]].append(ex)

    with open(f"processed_output/openai_rlhf_{split}.jsonl", "w", encoding="utf-8") as fout:
        # 4) for each image, if we have both native & nonnative...
        for url, examples in groups.items():
            native   = [e for e in examples if e["annotation_type"] == "native"]
            nonnative = [e for e in examples if e["annotation_type"] == "nonnative"]
            if not native or not nonnative:
                continue

            pref_text = native[0]["summary"]
            nonpref_text = nonnative[0]["summary"]
            record = {
                "input": {
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are an assistant that creates detailed captions of images, "
                                "with a strong focus on cultural elements but also an emphasis on all details."
                            )
                        },
                        {
                            "role": "user",
                            "content": "Provide a detailed caption of this image."
                        },
                        {
                            "role": "user",
                            "content": {
                                "type": "image_url",
                                "image_url": {
                                    "url": url,
                                    "detail": "auto"
                                }
                            }
                        }
                    ],
                    "tools": [],
                    "parallel_tool_calls": True
                },
                "preferred_output": [
                    {
                        "role": "assistant",
                        "content": pref_text
                    }
                ],
                "non_preferred_output": [
                    {
                        "role": "assistant",
                        "content": nonpref_text
                    }
                ]
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("Wrote preference_data.jsonl with", len(groups), "images (filtered).")

if __name__ == "__main__":
    main(split="train")
    main(split="test")