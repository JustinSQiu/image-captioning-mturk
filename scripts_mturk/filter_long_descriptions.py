import pandas as pd
from datasets import load_dataset

dataset = load_dataset('Lin-Chen/ShareGPT4V', 'ShareGPT4V')
filtered_dataset = dataset.filter(lambda x: x['image'].startswith('coco/train2017/'))
final_filtered_dataset = filtered_dataset.filter(lambda x: len(x['conversations'][1]['value'].split()) >= 225)
ids = [f"{image_id}.jpg" for image_id in final_filtered_dataset['train']['id']]
df = pd.DataFrame(ids, columns=['File Name'])
df.to_csv('output/filtered_file_list.csv', index=False)

print("Filtered IDs with .jpg extensions have been saved to 'filtered_file_list.csv'.")
