'''
Before running this script, navigate to the train2017 folder and do

ls -1 > ../{language}_file_list.csv
'''

import pandas as pd

language = 'Urdu'

url_prefix = f"https://raw.githubusercontent.com/JustinSQiu/image-captioning-mturk/master/cvqa/{language}_images/"
image_data = pd.read_csv(f'cvqa/{language}_file_list.csv', header=None, names=['image_name'])

# max_image_name = '000000002000.jpg'
# image_data = image_data[image_data['image_name'] <= max_image_name]

# check if the image name ends with _0.jpg; if not, don't add it
image_data = image_data[image_data['image_name'].str.endswith('_0.jpg')]

image_data['image_url'] = url_prefix + image_data['image_name']
image_data[['image_url']].to_csv(f'mturk_upload_files/CVQA_{language}.csv', index=False)
