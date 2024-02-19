from resnetmodel import MyResNet
from model import ModelCNN
from preparedata import SkinLesionDataModule
from utils import evaluate_in_folder, obtain_data_path, evaluate_accuracy
import torch


# Set paths to metadata and image folders (put 'A' for Aaron, or 'E' for Evelyn)
metadata_file_path, image_file_path = obtain_data_path('A')

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

    # Evaluate ResNet50 models (confusion matrix + accuracy)
    
    # model = MyResNet()
    # model.eval()
    # folder_path = "resnetv2models"
    # output_file = 'results/accuracy_resv2.csv'

    # evaluate_in_folder(model, test_dataloader, folder_path, output_file)

    # # model = ModelCNN()
    # # model.eval()
    # # folder_path = "models"
    # # output_file = 'results/accuracy.csv'

    # # evaluate_in_folder(model, test_dataloader, folder_path, output_file)

    # Evaluate checkpoints
    # model = MyResNet()
    # model.eval()
    # folder_path = "logs"
    # output_file = 'results/accuracy_reslog.csv'

    # evaluate_in_folder(model, test_dataloader, folder_path, output_file, True)
    path = 'logs\checkpoint-epoch=13-val_loss=0.63.ckpt'
    model = MyResNet.load_from_checkpoint(path)
    
if __name__ == "__main__":
    main()
