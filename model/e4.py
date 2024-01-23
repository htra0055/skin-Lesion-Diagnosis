# test.py

import pandas as pd
from sklearn.model_selection import train_test_split
from preparedata import SkinLesionDataModule
from model import ModelCNN

# Set paths to metadata and image folders
metadata_file_path = 'data/hamDataset/HAM10000_metadata.csv'
image_file_path = 'data/hamDataset/HAM10000_images'


def main():
    data_module = SkinLesionDataModule(
        metadata_file=metadata_file_path,
        image_folder=image_file_path,
        batch_size=4,
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

    #  # Accessing a specific case using case_id (example)
    # case_id = 0
    # specific_case = train_metadata_list[case_id]
    # specific_case.display_metadata()

    # Show a batch of images
    batch_images, batch_labels = next(iter(data_module.train_dataloader()))
    ModelCNN.show_batch_images(batch_images, batch_labels)

   

if __name__ == "__main__":
    main()
