"""
Originally called e5.py. Used to train the model.
"""

import pytorch_lightning as pl
import torch
from preparedata import SkinLesionDataModule
from model import ModelCNN
from utils import obtain_data_path

# Set paths to metadata and image folders (put 'A' for Aaron, or 'E' for Evelyn)
metadata_file_path, image_file_path = obtain_data_path('A')

def main():
    # Create a data module instance
    data_module = SkinLesionDataModule(
        metadata_file=metadata_file_path,
        image_folder=image_file_path,
        batch_size=16,
        num_workers=3
    )
    # Setup the data module by loading metadata and splitting into train and validation sets
    data_module.setup()

    # Create a model instance and load the data
    model = ModelCNN()
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    # Train the data
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloader, val_dataloader)

    # Saving the trained model into file 'trained_model.pth'
    torch.save(model.state_dict(), 'trained_model_2_ep.pth')

    
if __name__ == "__main__":
    main()
