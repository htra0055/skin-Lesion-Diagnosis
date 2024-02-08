
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

#from utils import show_batch_images

# Set paths to metadata and image folders
metadata_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_metadata.csv'
image_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_images_part_1'




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
                'image_id': {
                    'img_filename': '',
                    'img_size': 0
                },
                'dx': {
                    'dx': '',
                    'dx_type': '',
                    'age': 0,
                    'sex': '',
                    'localization': '',
                }
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
        self.metadata_dict['image_id']['img_filename'] = img_filename
        self.metadata_dict['image_id']['img_size'] = img_size
        self.metadata_dict['dx']['dx'] = row['dx']
        self.metadata_dict['dx']['dx_type'] = row['dx_type']
        self.metadata_dict['dx']['age'] = row['age']
        self.metadata_dict['dx']['sex'] = row['sex']
        self.metadata_dict['dx']['localization'] = row['localization']

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
        """
        self.metadata_list = metadata_list
        self.transformation = transformation

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
        img_path = os.path.join(image_file_path, metadata_instance.metadata_dict['image_id']['img_filename'])
        
        # Add this check before opening the image
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            # Handle the error or continue to the next iteration
            return None, None

        image = Image.open(img_path).convert('RGB')

        if self.transformation:
            image = self.transformation(image)

        label = metadata_instance.metadata_dict['dx']['dx']

        return image, label


class SkinLesionDataModule(pl.LightningDataModule):
    def __init__(self, metadata_file, image_folder, batch_size, num_workers):
        """
        PyTorch Lightning DataModule for skin lesion image data.

        Args:
            metadata_file (str): Path to the metadata CSV file.
            image_folder (str): Path to the folder containing image files.
            batch_size (int): Batch size for DataLoader.
            num_workers (int): Number of workers for DataLoader.
        """
        super().__init__()
        self.metadata_file = metadata_file
        self.image_folder = image_folder
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        """Download or load data if needed."""
        pass

    def setup(self, stage=None):
        """Split data into train and validation sets and create Metadata instances."""
        metadata_df = pd.read_csv(self.metadata_file)
        train_df, val_df = train_test_split(metadata_df, test_size=0.2, random_state=42)

        # Create Metadata instances for training and validation
        self.train_metadata_list = self.create_metadata_list(train_df)
        self.val_metadata_list = self.create_metadata_list(val_df)

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
            transforms.Resize(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(200),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
        ])

        dataset = CustomDataset(metadata_list=self.train_metadata_list, transformation=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        """Get DataLoader for validation data."""
        transform = transforms.Compose([
            transforms.Resize(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Imagenet standards
        ])

        dataset = CustomDataset(metadata_list=self.val_metadata_list, transformation=transform)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


class SimpleCNN(pl.LightningModule):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 128 * 56 * 56)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


# Function to show a batch of images (replace with your own implementation)
def show_batch_images(batch):
    # Your implementation to display images goes here
    pass

# Instantiate the SkinLesionDataModule
data_module = SkinLesionDataModule(
    metadata_file=metadata_file_path,
    image_folder=image_file_path,
    batch_size=64,
    num_workers=4
)

data_module.setup()

# Load the metadata CSV file
metadata_df = pd.read_csv(metadata_file_path)

# Split the data into train and validation sets
train_df, val_df = train_test_split(metadata_df, test_size=0.2, random_state=42)

# Create Metadata instances for training and validation
train_metadata_list = data_module.create_metadata_list(train_df)
val_metadata_list = data_module.create_metadata_list(val_df)

# Instantiate the model
model = SimpleCNN()

# Assuming you have the DataLoader and LightningDataModule set up
train_dataloader = data_module.train_dataloader()
val_dataloader = data_module.val_dataloader()

# Instantiate the Lightning Trainer
# Use gpus=0 for CPU-only training
trainer = pl.Trainer(max_epochs=5, gpus=1)

# Train the model
trainer.fit(model, train_dataloader, val_dataloader)

# Accessing a specific case using case_id (example)
case_id = 0
specific_case = train_metadata_list[case_id]
specific_case.display_metadata()

# Show a batch of images (replace with your own implementation)
# Note: Make sure you have the show_batch_images function implemented
batch_images, batch_labels = next(iter(train_dataloader))
show_batch_images(batch_images)


