import os
os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from helpers import get_gpt_response
from concurrent.futures import ThreadPoolExecutor

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

def gpt4o_caption(url, language) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt(language)},
                {"type": "image_url", "image_url": {"url": url}}
            ]
        }
    ]
    print("messages", messages, flush=True)
    summary = get_gpt_response(messages)
    return summary.replace("\n", " ").replace("\r", " ")

# def process_captions_gpt4o(batch_size=8):
#     ds = load_dataset("justinsunqiu/multilingual_transcriptions_translated_english_final", split="train")

#     OUT_CSV = "evaluation/output/multilingual_captions_gpt4o.csv"

#     with open(OUT_CSV, mode='w', encoding='utf-8') as f:
#         f.write("id,image_url,language,caption\n")

#         def process_single(ex):
#             try:
#                 url = ex["image_link"] if "image_link" in ex else ex["ids"][0]
#                 caption = gpt4o_caption(url)
#                 row = {
#                     "id": ex["__index_level_0__"],
#                     "image_url": url,
#                     "language": ex.get("language", ""),
#                     "caption": caption,
#                 }
#             except Exception as e:
#                 print("⚠️", ex["image_link"], e)
#                 row = None
#             return row

#         with ThreadPoolExecutor(max_workers=batch_size) as executor:
#             for row in tqdm(executor.map(process_single, ds), total=len(ds), desc="captioning_gpt4o"):
#                 if row:
#                     pd.DataFrame([row]).to_csv(f, header=False, index=False)
#                     f.flush()

#     print(f"✨ Done. Wrote rows incrementally to {OUT_CSV}")

if __name__ == "__main__":
    print("Processing gpt4o captions with batching and multithreading...")
    process_captions_gpt4o(batch_size=8)
