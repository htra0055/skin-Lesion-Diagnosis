# test.py

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from model import ModelCNN
from preparedata import SkinLesionDataModule
import pytorch_lightning as pl

# Set paths to metadata and image folders
# metadata_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_metadata.csv'
# image_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_images_part_1'

metadata_file_path = 'data/hamDataset/HAM10000_metadata.csv'
image_file_path = 'data/hamDataset/HAM10000_images'

def main():
    # Create a data module instance
    data_module = SkinLesionDataModule(
        metadata_file=metadata_file_path,
        image_folder=image_file_path,
        batch_size=16,
        num_workers=4
    )
    # Setup the data module by loading metadata and splitting into train and validation sets
    data_module.setup()

    # Load the test data
    test_dataloader = data_module.test_dataloader()

    # Load pre-trained model from file 'trained_model.pth'
    loaded_model = ModelCNN()
    loaded_model.load_state_dict(torch.load('trained_model.pth'))

    # Set the model to evaluation mode
    loaded_model.eval()

    # Create a Trainer instance
    trainer = pl.Trainer()

    # Run the testing loop
    trainer.test(loaded_model, test_dataloader)


if __name__ == "__main__":
    main()
