import csv, argparse, torch
from pathlib import Path
from unsloth import FastVisionModel
from transformers import TextStreamer
from finetune.models import get_llama_11b_model, get_trained_model
from finetune.data import to_pillow
from datasets import load_dataset

# ---------- utility ---------- #
def load_data():
    eval_dataset = load_dataset(
        "justinsunqiu/multilingual_transcriptions_final", split="test"
    )
    eval_dataset = eval_dataset.map(to_pillow, batched=True, num_proc=4)
    return eval_dataset

def build_prompt(lang):
    lang = lang.lower()
    if lang == "english" or lang == "en":
        return "Write a detailed caption for this image."
    elif lang == "chinese" or lang == "zh" or lang == "zh-cn":
        return "为这张图片写一个详细的标题。"
    elif lang == "korean" or lang == "ko":
        return "이 이미지에 대한 자세한 캡션을 작성하세요."
    elif lang == "russian" or lang == "ru":
        return "Напишите подробную подпись к этому изображению."
    elif lang == "hindi" or lang == "hi":
        return "इस छवि के लिए एक विस्तृत कैप्शन लिखें।"
    elif lang == "tamil" or lang == "ta":
        return "இந்த படத்திற்கு விரிவான கேப்ஷன் எழுதவும்."
    elif lang == "japanese" or lang == "ja":
        return "この画像の詳細なキャプションを書いてください。"
    elif lang == "vietnamese" or lang == "vi":
        return "Viết chú thích chi tiết cho hình ảnh này."
    elif lang == "nepali" or lang == "ne":
        return "यस छविको लागि विस्तृत क्याप्सन लेख्नुहोस्।"
    elif lang == "bengali" or lang == "bn":
        return "এই ছবিটির জন্য একটি বিস্তারিত ক্যাপশন লিখুন।"
    elif lang == "spanish" or lang == "es":
        return "Escribe un título detallado para esta imagen."
    elif lang == "telugu" or lang == "te":
        return "ఈ చిత్రం కోసం వివరమైన క్యాప్షన్ రాయండి."
    elif lang == "norwegian" or lang == "no":
        return "Skriv en detaljert bildetekst for dette bildet."
    elif lang == "german" or lang == "de":
        return "Schreibe eine detaillierte Bildunterschrift für dieses Bild."
    elif lang == "amharic" or lang == "am":
        return "ለዚህ ምስል ዝርዝር መግለጫ ይጻፉ።"
    elif lang == "thai" or lang == "th":
        return "เขียนคำบรรยายโดยละเอียดสำหรับภาพนี้"
    elif lang == "urdu" or lang == "ur":
        return "اس تصویر کے لیے تفصیلی کیپشن لکھیں۔"
    elif lang == "kinyarwanda" or lang == "rw":
        return "Andika ibisobanuro birambuye kuri iyi shusho."
    elif lang == "french" or lang == "fr":
        return "Écrivez une légende détaillée pour cette image."
    else:
        return "Write a detailed caption for this image."



# ---------- main ---------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[
        "cvqa", "multilingual_transcriptions",
        "english_translated_transcriptions"
    ], default="english_translated_transcriptions")
    parser.add_argument("--model", default="llama")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    out_csv = "evaluation/output/" + args.dataset + "_" + args.model + ".csv"

    # 1.  Load model & tokenizer
    if args.model == "llama":
        model, tokenizer = get_llama_11b_model()
    else:
        model, tokenizer = get_trained_model(args.model)

    print(f"Model: {args.model}")
    FastVisionModel.for_inference(model)
    model.to(args.device).eval()

    # 2.  Prepare output file
    out_path = Path(out_csv)
    first_write = not out_path.exists()
    fp = out_path.open("a", newline='', encoding="utf-8")
    writer = csv.writer(fp)
    if first_write:
        writer.writerow(["id", "image_url", "language", "caption"])

    # 3.  Run inference example-by-example
    eval_ds = load_data()
    streamer  = TextStreamer(tokenizer, skip_prompt=True)

    for idx, sample in enumerate(eval_ds):
        img_url   = sample.get("image_link")
        if args.dataset == "multilingual_transcriptions":
            language = sample.get("language")
        else:
            language = 'english'
        prompt    = build_prompt(language)

        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        print(f"Prompt: {messages[0]['content'][1]['text']}")
        chat_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        inputs = tokenizer(
            sample.get("image"),                 # FastVisionModel will auto-fetch / load img from URL or path
            chat_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(args.device)

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                streamer       = streamer,
                max_new_tokens = 2048,
                temperature    = 1.2,
                min_p          = 0.1,
                use_cache      = True,
            )
        caption = tokenizer.decode(output_ids[0, inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True).strip()

        # Persist row
        writer.writerow([idx, img_url, language, caption])
        fp.flush()          # progress-safe

        # optional: simple progress print
        print(f"[{idx+1}/{len(eval_ds)}] ✓")

    fp.close()
    print(f"\nFinished – CSV saved to {out_path.resolve()}")

if __name__ == "__main__":
    main()
