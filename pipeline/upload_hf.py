import os
from datasets import Dataset
import pandas as pd
from huggingface_hub import login

os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"
from keys import hf_token
login(hf_token)

df = pd.read_csv('processed_output/transcription_changes.csv')
df = df[df['image_link'] != 'https://raw.githubusercontent.com/JustinSQiu/image-captioning-mturk/master/mmid_images/Chinese_images/.DS_Store']

full_ds = Dataset.from_pandas(df)
# split_ds = full_ds.train_test_split(test_size=0.1, seed=42)

# train_ds = split_ds['train']
# val_ds = split_ds['test']

repo_id = "justinsunqiu/transcription_changes"

full_ds.push_to_hub(repo_id, split="train", private=False)
# val_ds.push_to_hub(repo_id, split="test", private=False)
