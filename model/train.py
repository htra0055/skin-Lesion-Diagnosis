"""
Originally called e5.py. Used to train the model.
"""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from preparedata import SkinLesionDataModule
from model import ModelCNN


# Set paths to metadata and image folders
metadata_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_metadata.csv'
image_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_images_part_1'

# metadata_file_path = 'data/hamDataset/HAM10000_metadata.csv'
# image_file_path = 'data/hamDataset/HAM10000_images'


def main():
    # Create a data module instance
    data_module = SkinLesionDataModule(
        metadata_file=metadata_file_path,
        image_folder=image_file_path,
        batch_size=16,
        num_workers=1
    )
    # Setup the data module by loading metadata and splitting into train and validation sets
    data_module.setup()

    # Create a model instance and load the data
    model = ModelCNN()
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()

    # Train the data
    trainer = pl.Trainer(max_epochs=1)
    trainer.fit(model, train_dataloader, val_dataloader)

    # Saving the trained model into file 'trained_model.pth'
    torch.save(model.state_dict(), 'trained_model.pth')

    
if __name__ == "__main__":
    main()
