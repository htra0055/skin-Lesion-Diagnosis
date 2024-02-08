import PIL.Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import pandas as pd 


# Set paths to metadata and image folders
metadata_file_path = '/Users/evelynhoangtran/Universe/MDN projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_metadata.csv'
image_file_path = '/Users/evelynhoangtran/Universe/MDN projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_images_part_1'


# Step 1: Open file
metadata_df = pd.read_csv(metadata_file_path)

# Step 2: Loop
metadata_case_list = []
case_id = 0

for index, row in metadata_df.iterrows():
    # Step 3: Extract each column and put in a dictionary
    sgl_meta_dict = {
        'lesion_id': row['lesion_id'],
        'image_id': {
            'img_filename': '',
            'img_size': 0
        },
        'dx': {
            'dx': row['dx'],
            'dx_type': row['dx_type'],
            'age': row['age'],
            'sex': row['sex'],
            'localization': row['localization'],

        }
    }

    # Step 4: Modify image path
    img_filename = row['image_id'] + ".jpg"
    sgl_image_path = os.path.join(image_file_path, img_filename)

    # Step 5: Put in one tuple
    metadata_tuple = (case_id, sgl_meta_dict, sgl_image_path)

    # Step 6: Append to metadata_case_list (For every tuple append in list metadata_case_list)
    metadata_case_list.append(metadata_tuple)

    # Step 7: Increment case_id
    case_id += 1

# Step 8: Case_total = len(metadata_case_list)
case_total = len(metadata_case_list)


# Accessing a specific case using case_id
case_id = 0
specific_case = metadata_case_list[case_id]

# Displaying the specific case
print(specific_case)
