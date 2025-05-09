from unsloth import FastVisionModel

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
]

def get_llama_11b_model():
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Llama-3.2-11B-Vision-Instruct-unsloth-bnb-4bit",
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,

        r = 32,
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        random_state = 42,
        use_rslora = False,
        loftq_config = None,
    )
    return model, tokenizer

def get_qwen_7b_model():
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True,
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,

        r = 32,
        lora_alpha = 32,
        lora_dropout = 0,
        bias = "none",
        random_state = 42,
        use_rslora = False,
        loftq_config = None,
    )
    return model, tokenizer

def get_trained_model(local_dir):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name = f"{local_dir}", # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit = True, # Set to False for 16bit LoRA
        max_seq_length = 2048,
    )
    return model, tokenizer
    # model, tokenizer = FastVisionModel.from_pretrained(
    #     f"justinsunqiu/{local_dir}",
    #     load_in_4bit = True,
    #     use_gradient_checkpointing = "unsloth",
    # )
    # model = FastVisionModel.get_peft_model(
    #     model,
    #     finetune_vision_layers     = True,
    #     finetune_language_layers   = True,
    #     finetune_attention_modules = True,
    #     finetune_mlp_modules       = True,

    #     r = 32,
    #     lora_alpha = 32,
    #     lora_dropout = 0,
    #     bias = "none",
    #     random_state = 42,
    #     use_rslora = False,
    #     loftq_config = None,
    # )
    # return model, tokenizer