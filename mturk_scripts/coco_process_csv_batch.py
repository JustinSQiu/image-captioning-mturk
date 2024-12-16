'''
Before running this script, navigate to the train2017 folder and do

ls -1 > ../output/file_list.csv
'''

import pandas as pd

url_prefix = "https://raw.githubusercontent.com/JustinSQiu/image-captioning-mturk/master/filtered_images/"
# max_image_name = '000000002000.jpg'
image_data = pd.read_csv('output/filtered_file_list.csv', header=None, names=['image_name'])
# image_data = image_data[image_data['image_name'] <= max_image_name]
image_data['image_url'] = url_prefix + image_data['image_name']
image_data[['image_url']].to_csv('mturk_upload_files/mturk_batch.csv', index=False)
