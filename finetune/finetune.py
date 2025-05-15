import os
import argparse
from unsloth import FastVisionModel
import torch
import wandb
from sklearn.metrics import accuracy_score
import numpy as np
from unsloth import is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import numpy as np

from keys import hf_token
from finetune.data import get_cvqa_dataset, get_multilingual_transcriptions_dataset, get_english_translated_transcriptions_dataset, get_vqa_dataset, get_english_translated_transcriptions_synthetic_dataset, get_backtranslated_transcriptions_dataset, get_backtranslated_vqa_dataset
from finetune.models import get_llama_11b_model, get_qwen_7b_model, get_trained_model

parser = argparse.ArgumentParser(description="Finetune a vision model on a dataset")
parser.add_argument("--output_dir", type=str, default="outputs/full",
                    help="Directory to save checkpoints and final model")
parser.add_argument("--model", type=str, default="llama",
                    help="Model to use for finetuning. Options: llama, qwen")
parser.add_argument("--dataset", type=str, default="multilingual_transcriptions",
                    help="Dataset to use for finetuning. Options: multilingual_transcriptions, cvqa, english_translated_transcriptions")
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train for")

args = parser.parse_args()

os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"
os.environ["WANDB_PROJECT"] = "thesis"
os.environ["WANDB_LOG_MODEL"] = args.output_dir

if args.model == "llama":
    model, tokenizer = get_llama_11b_model()
elif args.model == "qwen":
    model, tokenizer = get_qwen_7b_model()
else:
    model, tokenizer = get_trained_model(args.model)
    
FastVisionModel.for_training(model) # Enable for training!

if args.dataset == "cvqa":
    train_dataset, eval_dataset = get_cvqa_dataset()
elif args.dataset == "multilingual_transcriptions":
    train_dataset, eval_dataset = get_multilingual_transcriptions_dataset()
elif args.dataset == "english_translated_transcriptions":
    train_dataset, eval_dataset = get_english_translated_transcriptions_dataset()
elif args.dataset == "english_vqa":
    train_dataset, eval_dataset = get_vqa_dataset()
elif args.dataset == "synthetic_captions":
    train_dataset, eval_dataset = get_english_translated_transcriptions_synthetic_dataset()
elif args.dataset == "backtranslated_captions":
    train_dataset, eval_dataset = get_backtranslated_transcriptions_dataset()
elif args.dataset == "backtranslated_vqa":
    train_dataset, eval_dataset = get_backtranslated_vqa_dataset()
else:
    raise ValueError("Dataset not supported! Please use cvqa or multilingual_transcriptions.")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    args = SFTConfig(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.05,
        # max_steps = 500,
        num_train_epochs = args.epochs,
        learning_rate = 5e-5,
        fp16 = not is_bf16_supported(),
        bf16 = is_bf16_supported(),
        logging_steps = 1,

        # save_strategy="steps",
        # save_steps=500,
        # save_total_limit=1,

        optim = "adamw_8bit",
        weight_decay = 0.01,
        max_grad_norm = 1.0,
        lr_scheduler_type = "cosine",
        seed = 42,
        output_dir=args.output_dir,
        report_to = "wandb",
        eval_strategy = "steps",
        eval_steps = 50,
        eval_on_start = True,

        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        dataset_num_proc = 4,
        max_seq_length = 4096,
    ),
)

trainer_stats = trainer.train()

model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)


# print('Loading model from disk...', flush=True)

# local_dir = os.path.abspath(f"{args.output_dir}")
# assert os.path.isdir(local_dir), f"{local_dir} not found!"


# model, tokenizer = FastVisionModel.from_pretrained(
#     model_name = f"{local_dir}", # YOUR MODEL YOU USED FOR TRAINING
#     load_in_4bit = True, # Set to False for 16bit LoRA
# )

model.push_to_hub_merged(f"justinsunqiu/{args.output_dir}", tokenizer, token = hf_token)