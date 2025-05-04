import os
os.environ["HF_HOME"] = "/nlp/data/huggingface_cache"

import pandas as pd
from datasets import load_dataset

cvqa_test = load_dataset("afaji/cvqa", split="test")

rows = [{"ID": sample["ID"], "Prediction": 0} for sample in cvqa_test]
df = pd.DataFrame(rows)
df.to_csv("submission.csv", index=False)