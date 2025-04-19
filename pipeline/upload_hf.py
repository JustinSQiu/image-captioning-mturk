import os

os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

import pandas as pd
from datasets import Dataset
from huggingface_hub import HfApi, login
from keys import hf_token

login(hf_token)

# api = HfApi()
# api.create_repo(repo_id="justinsunqiu/multilingual_captions", repo_type="dataset")

df = pd.read_csv('processed_output/output_translated_summarized.csv')
df = df[df['image_link'] != 'https://raw.githubusercontent.com/JustinSQiu/image-captioning-mturk/master/mmid_images/Chinese_images/.DS_Store']
dataset = Dataset.from_pandas(df)

dataset.push_to_hub("justinsunqiu/multilingual_transcriptions_summarized", private=False)