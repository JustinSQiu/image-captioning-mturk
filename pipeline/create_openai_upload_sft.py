"""
Convert the Hugging Face dataset 'justinsunqiu/multilingual_transcriptions_summarized'
into a JSONL file suitable for OpenAI fine-tuning.
"""
import json
from datasets import load_dataset

def main(split="train"):
    ds = load_dataset("justinsunqiu/multilingual_transcriptions_summarized", split=split)

    with open(f"processed_output/openai_finetune_{split}.jsonl", "w", encoding="utf-8") as out_f:
        for example in ds:
            image_url = example["image_link"]
            caption   = example["summary"]

            messages = [
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
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": image_url}
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": caption
                }
            ]

            # 4. Write one JSON object per line
            record = {"messages": messages}
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main(split="train")
    main(split="test")
