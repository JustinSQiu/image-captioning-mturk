from unsloth import FastVisionModel
import argparse, torch
from PIL import Image
from transformers import TextStreamer

# --- your helper loaders --------------------------------------------------
from finetune.data import (
    get_cvqa_dataset,
    get_multilingual_transcriptions_dataset,
    get_english_translated_transcriptions_dataset,
)
from finetune.models import get_trained_model, get_llama_11b_model, get_qwen_7b_model
# --------------------------------------------------------------------------

def load_eval_dataset(name):
    if name == "cvqa":
        return get_cvqa_dataset()[1]
    if name == "multilingual_transcriptions":
        return get_multilingual_transcriptions_dataset()[1]
    if name == "english_translated_transcriptions":
        return get_english_translated_transcriptions_dataset()[1]
    raise ValueError("Unknown dataset")

def build_prompt(ex, dataset_name):
    """
    Return a *single* instruction string that works well with the sample.
    Adapt this stub however you like.
    """
    if dataset_name == "cvqa":
        # the dataset already has a question we want answered
        return ex["question"]

    # Both transcription datasets want free‑form descriptions / captions
    if dataset_name == "multilingual_transcriptions":
        lang = ex.get("language", "unknown language")
        return (
            f"You are a cultural historian. The following caption is in {lang}:\n"
            f"---\n{ex['transcription']}\n---\n"
            "Describe the most culturally distinctive aspect visible in the image."
        )

    if dataset_name == "english_translated_transcriptions":
        return (
            "You are a detailed image captioner. "
            "Give an accurate, fluent English description of the image."
        )

    return "Describe what you see."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
        default="justinsunqiu/english_translated_llama_final",
        help="Local path or HF repo ID for the *merged* weights")
    parser.add_argument("--dataset", choices=[
        "cvqa", "multilingual_transcriptions",
        "english_translated_transcriptions"
    ], default="english_translated_transcriptions")
    parser.add_argument("--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, tokenizer = get_trained_model(args.model_dir)
    # model, tokenizer = get_llama_11b_model()
    FastVisionModel.for_inference(model) # Enable for inference!

    eval_ds = load_eval_dataset(args.dataset)
    for i in range(5):
        sample  = eval_ds[i]
        print(sample)
        image = sample["messages"][0]["content"][1]["image"]
        
        instruction = build_prompt(sample, args.dataset)

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]}
        ]
        chat_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )

        inputs = tokenizer(
            image,
            chat_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(args.device)

        text_streamer = TextStreamer(tokenizer, skip_prompt=True)
        _ = model.generate(
            **inputs,
            streamer        = text_streamer,
            max_new_tokens  = 1024,
            use_cache       = True,
            temperature     = 1.2,
            min_p           = 0.1,
        )

if __name__ == "__main__":
    main()
