# preparedata.py

import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch
import pytorch_lightning as pl
import torchvision
from matplotlib import pyplot as plt
import numpy as np
from utils import label_to_str, str_to_label


# Set paths to metadata and image folders
metadata_file_path = 'data/hamDataset/HAM10000_metadata.csv'
image_file_path = 'data/hamDataset/HAM10000_images'

# metadata_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_metadata.csv'
# image_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_images_part_1'



class Metadata:
    def __init__(self, metadata_dict=None):
        """
        Class to represent metadata for a skin lesion image.

        Args:
            metadata_dict (dict, optional): Initial metadata dictionary. Defaults to None.
        """
        if metadata_dict is None:
            metadata_dict = {
                'lesion_id': '',
                'image_id': '',
                'dx': ''
            }
        self.metadata_dict = metadata_dict

    def extract_metadata(self, image_path, row):
        """
        Extract metadata from the given image path and DataFrame row.

        Args:
            image_path (str): Path to the image file.
            row (pd.Series): Row from the metadata DataFrame.
        """
        img_filename = os.path.basename(image_path)
        img_size = os.path.getsize(image_path)
        self.metadata_dict['lesion_id'] = row['lesion_id']
        self.metadata_dict['image_id'] = img_filename
        self.metadata_dict['dx'] = row['dx']

    def display_metadata(self):
        """Display the metadata information."""
        print(self.metadata_dict)


class CustomDataset(Dataset):
    def __init__(self, metadata_list, transformation=None):
        """
        Custom PyTorch dataset for skin lesion images.

        Args:
            metadata_list (list): List of Metadata instances.
            transformation (callable, optional): Image transformation. Defaults to None.
            class_mapping (dict): Mapping from string labels to integers
        """
        self.metadata_list = metadata_list
        self.transformation = transformation

       # self.class_mapping = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

    def __len__(self):
        """Get the number of samples in the dataset."""
        return len(self.metadata_list)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: Tuple containing the image and label.
        """
        metadata_instance = self.metadata_list[idx]
        img_path = os.path.join(image_file_path, metadata_instance.metadata_dict['image_id'])

        # This check before opening the image
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            return None, None  # Handle the error or continue to the next iteration

        image = Image.open(img_path).convert('RGB')

        if self.transformation:
            image = self.transformation(image)

        label_str = metadata_instance.metadata_dict['dx']
        label = self.str_to_label(label_str)
        
        return image, label
    
    def label_to_str(self, label: int) -> str:
        """
        Function to convert the label (int) to a string describing the diagnosis.

        Args:
            label (int): The numerical label to be converted.

        Returns:
            str: The string representation of the label e.g. bkl, bcc, akiec, vasc, df, mel, nv.
        """
        label_map = {
                'akiec': 0,
                'bcc': 1,
                'bkl': 2,
                'df': 3,
                'mel': 4,
                'nv': 5,
                'vasc': 6
            }
        
        return list(label_map.keys())[list(label_map.values()).index(label)]

    def str_to_label(self, diagnosis: str) -> int:
        """
        Converts a skin lesion diagnosis string to its corresponding label.

        Args:
            diagnosis (str): The skin lesion diagnosis.

        Returns:
            int: The corresponding label for the diagnosis.
        """
        label_map = {
            'akiec': 0,
            'bcc': 1,
            'bkl': 2,
            'df': 3,
            'mel': 4,
            'nv': 5,
            'vasc': 6
        }
        return label_map[diagnosis]



class SkinLesionDataModule(pl.LightningDataModule):
    def __init__(self, metadata_file, image_folder, batch_size, num_workers):
        """
        PyTorch Lightning DataModule for skin lesion image data.

        Args:
            metadata_file (str): Path to the metadata CSV file.
            image_folder (str): Path to the folder containing image files.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
            #class_mapping (dict): Mapping from string labels to integers
        """


        super().__init__()
        self.metadata_file = metadata_file
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

    
    def setup(self, stage=None):
        # Split data into train, validation, and test sets and create Metadata instances
        metadata_df = pd.read_csv(self.metadata_file)
        
        # Split into train, val, and test sets
        train_df, temp_df = train_test_split(metadata_df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

        # Create Metadata instances for training, validation, and test
        self.train_metadata_list = self.create_metadata_list(train_df)
        self.val_metadata_list = self.create_metadata_list(val_df)
        self.test_metadata_list = self.create_metadata_list(test_df)


    def create_metadata_list(self, df):
        """
        Create a list of Metadata instances from a DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing metadata.

        Returns:
            list: List of Metadata instances.
        """
        metadata_list = []
        for index, row in df.iterrows():
            metadata_instance = Metadata()
            img_filename = row['image_id'] + ".jpg"
            img_path = os.path.join(self.image_folder, img_filename)
            metadata_instance.extract_metadata(img_path, row)
            metadata_list.append(metadata_instance)
        return metadata_list

    def train_dataloader(self):
        """Get DataLoader for training data."""
        transform = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(200),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
        ])
        
        
        dataset = CustomDataset(metadata_list=self.train_metadata_list, transformation=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """Get DataLoader for validation data."""
        transform = transforms.Compose([
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
        ])
        #class_mapping = {'akiec': 0, 'bcc': 1, 'bkl': 2, 'df': 3, 'mel': 4, 'nv': 5, 'vasc': 6}

        dataset = CustomDataset(metadata_list=self.val_metadata_list, transformation=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        """Get DataLoader for testing data."""
        transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
        ])

        dataset = CustomDataset(metadata_list=self.test_metadata_list, transformation=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

