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
            raise e
    
    examples['image'] = images
    return examples

# dataset = load_dataset('justinsunqiu/multilingual_transcriptions_summarized', split='train')
# dataset = dataset.map(to_pillow, batched=True, num_proc=4)

dataset = load_dataset('afaji/cvqa', split='test')

def convert_to_conversation(sample):
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : f"Write a detailed caption for this image."},
            {"type" : "image", "image" : sample["image"]},
          ]
        },
        { "role" : "assistant",
          "content" : [
            {"type" : "text",  "text"  : sample["summary"]} ]
        },
    ]
    return { "messages" : conversation }

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
converted_dataset = [convert_to_conversation_cvqa(sample) for sample in dataset]
train_dataset = converted_dataset[:int(0.98*len(converted_dataset))]
eval_dataset = converted_dataset[int(0.98*len(converted_dataset)):]
eval_dataset = converted_dataset[-2:]