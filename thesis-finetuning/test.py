import os

os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

from datasets import load_dataset


cvqa_dataset = load_dataset('afaji/cvqa', split='test')

def convert_to_conversation_cvqa(sample):
    conversation = [
        {'role': 'system', 'content': "You are a helpful assistant that answers questions about images. You will be given a question and a list of four multiple choice answers. You need to select the correct one. Your output should only consist of one singular number between 0 and 3 indicating the index of the correct answer. Do not include any other text."},
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

converted_cvqa_dataset = convert_to_conversation_cvqa(cvqa_dataset[0])
print(converted_cvqa_dataset)