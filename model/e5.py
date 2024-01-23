# train.py

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from preparedata import SkinLesionDataModule, CustomDataset
from model import ModelCNN


# Set paths to metadata and image folders
metadata_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_metadata.csv'
image_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_images_part_1'


def main():
    data_module = SkinLesionDataModule(
        metadata_file=metadata_file_path,
        image_folder=image_file_path,
        batch_size=64,
        num_workers=4
    )

    data_module.setup(stage='fit')

    model = ModelCNN()
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()

    trainer = pl.Trainer(max_epochs=5, gpus=1)
    trainer.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()
