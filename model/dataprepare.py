

import PIL.Image
import torch
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd

"""
Put all the images into one folder called HAM10000_images
"""


# # Extracting the columns from the CSV file
# lesion_id = metadata['lesion_id']
# image_id = metadata['image_id']
# dx = metadata['dx']
# dx_type = metadata['dx_type']



# Create dataset
class CustomDataset(Dataset):
    def __init__(self, images_dir, metadata_dir, transformation=None):
        self.images_dir = images_dir
        self.data_frame = pd.read_csv(metadata_dir)
        self.transformation = transformation

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx) -> PIL.Image.Image:
        # Obtain the image path using the image_id column in the metadata
        img_name = os.path.join(self.images_dir, self.data_frame.iloc[idx, 1])
        # Add filename
        img_name = img_name + '.jpg'
        print(img_name)
        image = Image.open(img_name).convert('RGB')

        if self.transformation:
            image = self.transformation(image)

        label = self.data_frame.iloc[idx, 2]

        return image, label


metadata_file_path = '/Users/evelynhoangtran/Universe/MDN projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_metadata.csv'
image_file_path = '/Users/evelynhoangtran/Universe/MDN projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_images_part_1'
dataset = CustomDataset(images_dir=image_file_path, metadata_dir=metadata_file_path)


print(dataset[1][0].show())
# # Load data
# batch_size = 64
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
#
# print('a')


# convert data to a normalized torch.FloatTensor and Resize
transform = transforms.Compose([
    transforms.Resize(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]) # Imagenet standards
    ])


# # Define transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# ])

