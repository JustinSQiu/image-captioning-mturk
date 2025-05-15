from transformers import pipeline

pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf")

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


def get_llava_response(image_url, language):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_url},
                {"type": "text", "text": build_prompt(language)},
            ],
        },
    ]
    print("messages", messages)
    out = pipe(text=messages, max_new_tokens=2048)
    return out[0]["generated_text"]

if __name__ == "__main__":
    image_url = "https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png"
    prompt = "What is the text in the illustrate?"
    response = get_llava_response(image_url, prompt)
    print(response)

# import os
# os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

# from datasets import load_dataset
# from transformers import pipeline
# from PIL import Image
# import requests

# pipe = pipeline("image-text-to-text", model="llava-hf/llava-1.5-7b-hf", device=0)

# def generate_caption(batch):
#     images = []
#     for url in batch["image_link"]:
#         image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
#         images.append(image)

#     prompts = [
#         [
#             {"role": "user", "content": [
#                 {"type": "image"},
#                 {"type": "text", "text": "Write a detailed caption for this image."}
#             ]}
#         ] for _ in images
#     ]

#     outputs = pipe(images, prompts, batch_size=8, max_new_tokens=2048)
#     batch["caption"] = [output[0]["generated_text"] for output in outputs]
#     return batch

# def process_dataset():
#     ds = load_dataset("justinsunqiu/multilingual_transcriptions_translated_english_final", split="train")

#     ds = ds.map(generate_caption, batched=True, batch_size=8, desc="Generating captions")

#     ds = ds.remove_columns([col for col in ds.column_names if col not in ["__index_level_0__", "image_link", "language", "caption"]])

#     ds.to_csv("evaluation/output/english_captions_llava.csv", index=False)

# if __name__ == "__main__":
#     process_dataset()
#     print("✨ Done. Captions generated efficiently using batching.")
