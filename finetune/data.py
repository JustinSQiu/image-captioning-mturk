import random
from datasets import load_dataset, concatenate_datasets
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


def convert_to_conversation_vqa(sample, multilingual=False):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": sample["question"]},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": sample["answer"]},
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

def get_english_translated_transcriptions_synthetic_dataset(seed=42):
    # 1) Load CVQA + CulturalVQA (with unified "ID") but only to build an index map
    ds_cvqa = load_dataset("afaji/cvqa", split="test")
    ds_cult = (
        load_dataset("mair-lab/CulturalVQA", split="test")
        .rename_column("u_id", "ID")
    )
    base = concatenate_datasets([ds_cvqa, ds_cult])

    # 2) Build a lightweight mapping: ID -> row index (integer)
    #    (this only stores strings and ints, so it’s small)
    id_list = base["ID"]
    id_to_idx = {image_id: i for i, image_id in enumerate(id_list)}

    # 3) Load your synthetic dataset, rename & shuffle
    train = (
        load_dataset("justinsunqiu/cvqa_captions_synthetic_final", split="train")
        .rename_column("caption", "summary")
        .shuffle(seed=seed)
    )
    test = (
        load_dataset("justinsunqiu/cvqa_captions_synthetic_final", split="test")
        .rename_column("caption", "summary")
        .shuffle(seed=seed)
    )

    # 4) Map in the image by looking up its index and pulling just that one row
    def _attach_image(example):
        base_id = example["image_id"]
        # build list of candidates to try
        candidates = [base_id] + [f"{base_id}_{i}" for i in range(3)]
        for key in candidates:
            idx = id_to_idx.get(key)
            if idx is not None:
                # found a matching ID → load exactly that one image
                example["image"] = base[idx]["image"]
                break
        else:
            # no match at all: drop or mark missing
            example["image"] = None
        return example

    train = train.map(_attach_image, num_proc=4)
    test  = test.map(_attach_image,  num_proc=4)
    train = train.filter(lambda ex: ex["image"] is not None)
    test  = test.filter(lambda ex: ex["image"] is not None)

    # 5) (Optional) convert to pillow format if downstream needs it
    # train = train.map(to_pillow, batched=True, num_proc=4)
    # test  = test.map(to_pillow,  batched=True, num_proc=4)

    # 6) Convert to conversation format
    converted_train = [convert_to_conversation(x) for x in train]
    converted_test  = [convert_to_conversation(x) for x in test]

    return converted_train, converted_test

def get_backtranslated_transcriptions_dataset(seed=42):
    train_dataset = load_dataset(
        "justinsunqiu/multilingual_transcriptions_summarized_by_english_backtranslated_final", split="train"
    )
    eval_dataset = load_dataset(
        "justinsunqiu/multilingual_transcriptions_summarized_by_english_backtranslated_final", split="test"
    )
    # rename "summary" to "tmp" and "backtranslation" to "summary"
    train_dataset = train_dataset.rename_column("summary", "tmp")
    eval_dataset = eval_dataset.rename_column("summary", "tmp")
    train_dataset = train_dataset.rename_column("backtranslation", "summary")
    eval_dataset = eval_dataset.rename_column("backtranslation", "summary")
    train_dataset = train_dataset.shuffle(seed=seed)
    eval_dataset = eval_dataset.shuffle(seed=seed)
    train_dataset = train_dataset.map(to_pillow, batched=True, num_proc=4)
    eval_dataset = eval_dataset.map(to_pillow, batched=True, num_proc=4)
    converted_train = [convert_to_conversation(sample, multilingual=True) for sample in train_dataset]
    converted_eval = [convert_to_conversation(sample, multilingual=True) for sample in eval_dataset]
    return converted_train, converted_eval


def get_cvqa_dataset(eval_size=0.05, seed=42, eval_only=False):
    dataset = load_dataset("afaji/cvqa", split="test")
    if eval_only:
        return dataset
    dataset = dataset.shuffle(seed=seed)
    # dataset = dataset.map(to_pillow, batched=True, num_proc=4)
    converted = [convert_to_conversation_cvqa(sample) for sample in dataset]
    split_idx = int(eval_size * len(converted))
    eval_dataset = converted[:split_idx]
    train_dataset = converted[split_idx:]
    return train_dataset, eval_dataset

def get_vqa_dataset(seed=42):
    train_dataset = load_dataset(
        "justinsunqiu/multilingual_vqa_final", split="train"
    )
    eval_dataset = load_dataset(
        "justinsunqiu/multilingual_vqa_final", split="test"
    )
    train_dataset = train_dataset.shuffle(seed=seed)
    eval_dataset = eval_dataset.shuffle(seed=seed)
    train_dataset = train_dataset.map(to_pillow, batched=True, num_proc=4)
    eval_dataset = eval_dataset.map(to_pillow, batched=True, num_proc=4)
    converted_train = [convert_to_conversation_vqa(sample) for sample in train_dataset]
    converted_eval = [convert_to_conversation_vqa(sample) for sample in eval_dataset]
    return converted_train, converted_eval
from datasets import load_dataset

def get_backtranslated_vqa_dataset(seed=42):
    train_dataset = load_dataset(
        "justinsunqiu/multilingual_vqa_backtranslated_final",
        split="train",
        streaming=True
    ).shuffle(buffer_size=1_000, seed=seed)
    eval_dataset  = load_dataset(
        "justinsunqiu/multilingual_vqa_backtranslated_final",
        split="test",
        streaming=True
    ).shuffle(buffer_size=500,  seed=seed)
    train_dataset = train_dataset.rename_column("question", "tmp1")
    eval_dataset = eval_dataset.rename_column("question", "tmp1")
    train_dataset = train_dataset.rename_column("answer", "tmp2")
    eval_dataset = eval_dataset.rename_column("answer", "tmp2")
    train_dataset = train_dataset.rename_column("question_backtranslation", "question")
    eval_dataset = eval_dataset.rename_column("question_backtranslation", "question")
    train_dataset = train_dataset.rename_column("answer_backtranslation", "answer")
    eval_dataset = eval_dataset.rename_column("answer_backtranslation", "answer")
    train_dataset = train_dataset.shuffle(seed=seed)
    eval_dataset = eval_dataset.shuffle(seed=seed)
    train_dataset = train_dataset.map(to_pillow, batched=True)
    eval_dataset = eval_dataset.map(to_pillow, batched=True)
    converted_train = [convert_to_conversation_vqa(sample) for sample in train_dataset]
    converted_eval = [convert_to_conversation_vqa(sample) for sample in eval_dataset]
    return converted_train, converted_eval