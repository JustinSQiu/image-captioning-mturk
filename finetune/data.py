import random
from datasets import load_dataset
from PIL import Image
import requests

def to_pillow(examples):
    urls = examples["image_link"]
    images = []
    for url in urls:
        try:
            image = Image.open(requests.get(url, stream=True).raw)
            images.append(image)
        except Exception as e:
            print(f"Error loading image from {url}: {e}")
            raise e

    examples["image"] = images
    return examples

def get_prompt_multilingual(sample, multilingual=False):
    if not multilingual:
        return "Write a detailed caption for this image."
    try:
        lang = sample["language"].strip().lower()
        print(f"Language detected: {lang}")
    except KeyError:
        lang = "english"

    if lang == "english":
        return "Write a detailed caption for this image."
    elif lang == "chinese":
        return "为这张图片写一个详细的标题。"
    elif lang == "korean":
        return "이 이미지에 대한 자세한 캡션을 작성하세요."
    elif lang == "russian":
        return "Напишите подробную подпись к этому изображению."
    elif lang == "hindi":
        return "इस छवि के लिए एक विस्तृत कैप्शन लिखें।"
    elif lang == "tamil":
        return "இந்த படத்திற்கு விரிவான கேப்ஷன் எழுதவும்."
    elif lang == "japanese":
        return "この画像の詳細なキャプションを書いてください。"
    elif lang == "vietnamese":
        return "Viết chú thích chi tiết cho hình ảnh này."
    elif lang == "nepali":
        return "यस छविको लागि विस्तृत क्याप्सन लेख्नुहोस्।"
    elif lang == "bengali":
        return "এই ছবিটির জন্য একটি বিস্তারিত ক্যাপশন লিখুন।"
    elif lang == "spanish":
        return "Escribe un título detallado para esta imagen."
    elif lang == "telugu":
        return "ఈ చిత్రం కోసం వివరమైన క్యాప్షన్ రాయండి."
    elif lang == "norwegian":
        return "Skriv en detaljert bildetekst for dette bildet."
    elif lang == "german":
        return "Schreibe eine detaillierte Bildunterschrift für dieses Bild."
    elif lang == "amharic":
        return "ለዚህ ምስል ዝርዝር መግለጫ ይጻፉ።"
    elif lang == "thai":
        return "เขียนคำบรรยายโดยละเอียดสำหรับภาพนี้"
    elif lang == "urdu":
        return "اس تصویر کے لیے تفصیلی کیپشن لکھیں۔"
    elif lang == "kinyarwanda":
        return "Andika ibisobanuro birambuye kuri iyi shusho."
    elif lang == "french":
        return "Écrivez une légende détaillée pour cette image."
    else:
        raise ValueError(f"Unsupported language: {lang}")

def convert_to_conversation(sample, multilingual=False):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": get_prompt_multilingual(sample, multilingual)},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["summary"]},
            ],
        },
    ]
    return {"messages": conversation}

def convert_to_conversation_cvqa(sample):
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "You are a helpful assistant that answers questions about images. "
                        "You will be given a question and a list of four multiple choice answers. "
                        "You need to select the correct one. Your output should only consist of "
                        "one singular number between 0 and 3 indicating the index of the correct answer. "
                        f"Do not include any other text.\nQuestion: {sample['Question']}\n"
                        f"Options: {sample['Options']}"
                    ),
                },
                {"type": "image", "image": sample["image"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["Label"]},
            ],
        },
    ]
    return {"messages": conversation}

def get_multilingual_transcriptions_dataset(seed=42):
    train_dataset = load_dataset(
        "justinsunqiu/multilingual_transcriptions_final", split="train"
    )
    eval_dataset = load_dataset(
        "justinsunqiu/multilingual_transcriptions_final", split="test"
    )
    train_dataset = train_dataset.shuffle(seed=seed)
    eval_dataset = eval_dataset.shuffle(seed=seed)
    train_dataset = train_dataset.map(to_pillow, batched=True, num_proc=4)
    eval_dataset = eval_dataset.map(to_pillow, batched=True, num_proc=4)
    converted_train = [convert_to_conversation(sample, multilingual=True) for sample in train_dataset]
    converted_eval = [convert_to_conversation(sample, multilingual=True) for sample in eval_dataset]
    return converted_train, converted_eval

def get_english_translated_transcriptions_dataset(seed=42):
    train_dataset = load_dataset(
        "justinsunqiu/multilingual_transcriptions_translated_english_final", split="train"
    )
    eval_dataset = load_dataset(
        "justinsunqiu/multilingual_transcriptions_translated_english_final", split="test"
    )
    train_dataset = train_dataset.shuffle(seed=seed)
    eval_dataset = eval_dataset.shuffle(seed=seed)
    train_dataset = train_dataset.map(to_pillow, batched=True, num_proc=4)
    eval_dataset = eval_dataset.map(to_pillow, batched=True, num_proc=4)
    converted_train = [convert_to_conversation(sample) for sample in train_dataset]
    converted_eval = [convert_to_conversation(sample) for sample in eval_dataset]
    return converted_train, converted_eval

def get_cvqa_dataset(eval_size=0.05, seed=42):
    dataset = load_dataset("afaji/cvqa", split="train")
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.map(to_pillow, batched=True, num_proc=4)
    converted = [convert_to_conversation_cvqa(sample) for sample in dataset]
    split_idx = int(eval_size * len(converted))
    eval_dataset = converted[:split_idx]
    train_dataset = converted[split_idx:]
    return train_dataset, eval_dataset
