import os

os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

os.environ["WANDB_PROJECT"] = "thesis"
os.environ["WANDB_LOG_MODEL"] = "full"

from unsloth import FastVisionModel # FastLanguageModel for LLMs
import torch
import wandb
from sklearn.metrics import accuracy_score
import numpy as np
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import numpy as np

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth

# Load model
model, tokenizer = FastVisionModel.from_pretrained(
    # "unsloth/Qwen2-VL-7B-Instruct",
    # "unsloth/Qwen2.5-VL-7B-Instruct",
    # "unsloth/Qwen2.5-VL-3B-Instruct-unsloth-bnb-4bit",
    "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit",
    load_in_4bit = True, # Use 4bit to reduce memory use. False for 16bit LoRA.
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)


model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 32,           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 32,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 42,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
)


# Load dataset
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

dataset = load_dataset('justinsunqiu/multilingual_transcriptions_summarized', split='train')
dataset = dataset.map(to_pillow, batched=True, num_proc=4)

# dataset = load_dataset('afaji/cvqa', split='test')

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

converted_dataset = [convert_to_conversation(sample) for sample in dataset]
# converted_dataset = [convert_to_conversation_cvqa(sample) for sample in cvqa_dataset]
train_dataset = converted_dataset[:int(0.98*len(converted_dataset))]
eval_dataset = converted_dataset[int(0.98*len(converted_dataset)):]
eval_dataset = converted_dataset[-2:]

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     print(f"Predictions: {predictions}")
#     print(f"Labels: {labels}")
#     print('', flush=True)
#     accuracy = accuracy_score(labels, predictions)
#     return {"accuracy": accuracy}

# Train model
FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer), # Must use!
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    # compute_metrics = compute_metrics,
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_steps = 50,
        # max_steps = 500,
        num_train_epochs = 3, # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 42,
        output_dir = "full",
        report_to = "wandb",     # For Weights and Biases
        eval_strategy = "steps",
        eval_steps = 100,
        eval_on_start = True,

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 2048,
    ),
)

trainer_stats = trainer.train()

model.save_pretrained("full")  # Local saving
tokenizer.save_pretrained("full")