"""
For testing and running things
"""
from preparedata import SkinLesionDataModule
from model.utils import show_batch_images



if __name__ == '__main__':
    # Set paths to metadata and image folders
    metadata_file_path = 'data/hamDataset/HAM10000_metadata.csv'
    image_file_path = 'data/hamDataset/HAM10000_images'

    # metadata_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_metadata.csv'
    # image_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_images_part_1'

    # Creating the data module
    data_module = SkinLesionDataModule(
    metadata_file=metadata_file_path,
    image_folder=image_file_path,
    batch_size=4,
    num_workers=4
    )

    # Setup the data module
    data_module.setup()
    loader = data_module.train_dataloader()

    # Show the first batch of images
    show_batch_images(loader)
