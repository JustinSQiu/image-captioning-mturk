import os

os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

from unsloth import FastVisionModel
from datasets import load_dataset
from PIL import Image
import requests

def to_pillow(examples):
    urls = examples['image_link']
    images = []
    for url in urls:
        try:
            image = Image.open(requests.get(url, stream=True).raw)
            images.append(image)
        except Exception as e:
            print(f"Error loading image from {url}: {e}")
            images.append(None)
    
    examples['image'] = images
    # Remove examples with None images
    examples = [ex for ex in examples if ex['image'] is not None]
    return examples

cvqa_dataset = load_dataset('afaji/cvqa', split='test')

def convert_to_conversation_cvqa(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : "You are a helpful assistant that answers questions about images. You will be given a question and a list of four multiple choice answers. You need to select the correct one. Your output should only consist of one singular number between 0 and 3 indicating the index of the correct answer. Do not include any other text.\nQuestion: " + sample["Question"] + "\nOptions: " + str(sample["Options"])},
            {"type" : "image", "image" : sample["image"]},
          ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["Label"]} ]
        },
    ]
    return { "messages" : conversation }

# converted_dataset = [convert_to_conversation(sample) for sample in dataset]
converted_cvqa_dataset = [convert_to_conversation_cvqa(sample) for sample in cvqa_dataset]

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "cvqa_only", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = True, # Set to False for 16bit LoRA
)
FastVisionModel.for_inference(model) # Enable for inference!

image = cvqa_dataset[0]["image"]
instruction = "Describe this image."

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "You are a helpful assistant that answers questions about images. You will be given a question and a list of four multiple choice answers. You need to select the correct one. Your output should only consist of one singular number between 0 and 3 indicating the index of the correct answer. Do not include any other text.\nQuestion: " + cvqa_dataset[0]["Question"] + "\nOptions: " + str(cvqa_dataset[0]["Options"])}
    ]}
]
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt = True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)
