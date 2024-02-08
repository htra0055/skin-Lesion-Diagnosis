# test.py

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from model import ModelCNN
from preparedata import SkinLesionDataModule
import pytorch_lightning as pl
import os
import csv

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
    model = ModelCNN()
    # loaded_model.load_state_dict(torch.load('models/trained_model_5ep1.pth'))

    # Set the model to evaluation mode
    model.eval()

    # Create a Trainer instance
    trainer = pl.Trainer()

    # Run the testing loop
    # trainer.test(loaded_model, test_dataloader)
    folder_path = "models"
    filenames = get_filenames_in_folder('models')
    output_file = 'accuracy.csv'
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Correct', 'Total', 'Accuracy'])  # Writing header
        for file in filenames:
            correct, total, accuracy = evaluate_accuracy(model, test_dataloader, file, folder_path)
            # save to csv file
            writer.writerow((correct, total, accuracy))


def evaluate_accuracy(model: pl.LightningModule, test_dataloader: torch.utils.data.DataLoader, filename, folderpath) -> float:
    model.load_state_dict(torch.load(f'{folderpath}/{filename}'))
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation during inference
        for data in test_dataloader:
            inputs, labels = data
            outputs = model(inputs)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return correct, total, accuracy

def get_filenames_in_folder(folder_path):
    filenames = os.listdir(folder_path)
    return filenames

        
        





if __name__ == "__main__":
    main()

