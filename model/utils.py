import torchvision
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import csv
import torch
import os

from typing import Tuple
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from resnetmodel import MyResNet



def obtain_data_path(user: str) -> Tuple[str, str]:
    """
    Obtain the file paths for the metadata and image files based on the user.
    If the user is 'A', the file paths are for Aaron's local machine.
    If the user is 'E', the file paths are for Evelyn's local machine.

    Args:
        user (str): The user identifier.

    Returns:
        Tuple[str, str]: A tuple containing the metadata file path and image file path.
    """
    if user == 'A':
        metadata_file_path = 'data/hamDataset/HAM10000_metadata.csv'
        image_file_path = 'data/hamDataset/HAM10000_images'
        return metadata_file_path, image_file_path
    else:
        metadata_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_metadata.csv'
        image_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_images_part_1'
        return metadata_file_path, image_file_path


def label_to_str(label: int) -> str:
    """
    Convert the label (int) to a string describing the diagnosis.

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


def str_to_label(diagnosis: str) -> int:
    """
    Convert a skin lesion diagnosis string to its corresponding label.

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


def show_batch_images(dataloader: torch.utils.data.DataLoader):
    """
    Display a batch of images from a dataloader.

    Parameters:
    dataloader (torch.utils.data.DataLoader): The dataloader containing the images.

    Retuns:
    None
    """
    # get random training images with iter function
    dataiter = iter(dataloader)
    images, labels = next(dataiter)

    for index, label in enumerate(labels):
        # Assuming labels are stored as tensors
        print(f'Index: {index} Label: {label}')

    # # call function on our images
    images = torchvision.utils.make_grid(images)

    images = images / 2 + 0.5  # unnormalize
    npimg = images.numpy()  # convert to numpy objects
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def evaluate(model: pl.LightningModule, test_dataloader: torch.utils.data.DataLoader, folderpath: str, filename: str, checkpoint: bool = False) -> Tuple[int, int, float]:
    """
    Evaluate the performance of a model on a test dataset. This includes 
    calculating the accuracy and generating confusion matrix.

    Args:
        model (pl.LightningModule): The model to evaluate.
        test_dataloader (torch.utils.data.DataLoader): The dataloader for the test dataset.
        filepath (str): The filepath to load the model's state dictionary from.

    Returns:
        Tuple[int, int, float]: A tuple containing the number of correct predictions, total number of predictions,
        and the accuracy of the model on the test dataset.
    """
    print(f'Loading model from {folderpath}/{filename}')
    # if .pth file
    if not checkpoint:
        model.load_state_dict(torch.load(f'{folderpath}/{filename}', map_location=torch.device('cpu')))
    else:
        model = MyResNet.load_from_checkpoint(f'{folderpath}/{filename}')
    
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient calculation during inference
        for data in test_dataloader:
            inputs, labels = data
            outputs = model(inputs)  # Forward pass

            _, predicted = torch.max(outputs.data, 1)

            all_preds.append(predicted)
            all_labels.append(labels)

    all_preds_flat = torch.cat(all_preds, dim=0)
    all_labels_flat = torch.cat(all_labels, dim=0)

    # Accuracy
    correct, total, accuracy = evaluate_accuracy(
        all_preds_flat, all_labels_flat)
    print(f'Correct: {correct}, Total: {total}, Accuracy: {accuracy}')

    # Confusion matrix
    show_confusion_matrix(all_preds_flat, all_labels_flat, filename)

    return (correct, total, accuracy)


def show_confusion_matrix(all_preds: torch.tensor, all_labels:torch.tensor, filename:str):
    """
    Display the confusion matrix based on the predicted and true labels.

    Args:
        all_preds (torch.Tensor): The predicted labels.
        all_labels (torch.Tensor): The true labels.
        filename (str): The filename to save the confusion matrix image.
    """
    # Calculate the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Visualize the confusion matrix
    class_names = ['blk', 'nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc']

    # Visualize the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.savefig(f'confusion_matrix/{filename}.png')


def evaluate_accuracy(predicted: torch.Tensor, labels: torch.tensor) -> float:
    """
    Calculate the accuracy based on the predicted and true labels.

    Args:
        predicted (torch.Tensor): The predicted labels.
        labels (torch.Tensor): The true labels.

    Returns:
        float: The accuracy of the predictions.
    """
    correct = 0
    total = 0

    total = labels.size(0)
    correct = (predicted == labels).sum().item()

    accuracy = correct / total
    return (correct, total, accuracy)


def get_filenames_in_folder(folder_path: str):
    """
    Get the filenames of all files in a folder.

    Args:
        folder_path (str): The path to the folder.

    Returns:
        List[str]: A list of filenames in the folder.
    """
    filenames = os.listdir(folder_path)
    return filenames


def evaluate_in_folder(model: pl.LightningModule, test_dataloader: torch.utils.data.DataLoader, folderpath: str, output_file: str, checkpoint: bool = False) -> float:
    """
    Evaluate the performance of a model on all files in a folder.

    Args:
        model (pl.LightningModule): The model to evaluate.
        test_dataloader (torch.utils.data.DataLoader): The dataloader for the test dataset.
        folderpath (str): The path to the folder containing the files.
        output_file (str): The path to the output file to save the evaluation results.

    Returns:
        float: The average accuracy of the model on the files in the folder.
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Correct', 'Total',
                        'Accuracy'])  # Writing header

    filenames = get_filenames_in_folder(folderpath)

    for file in filenames:
        correct, total, accuracy = evaluate(model, test_dataloader, folderpath, file, checkpoint)
        # save to csv file
        save_results_to_csv(file, output_file, (correct, total, accuracy))


def save_results_to_csv(file: str, output_file: str, data: Tuple[int, int, float]):
    """
    Save the evaluation results to a CSV file.

    Args:
        file (str): The filename.
        output_file (str): The path to the output file.
        data (Tuple[int, int, float]): The evaluation results.

    Returns:
        None
    """
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        correct, total, accuracy = data
        writer.writerow((file, correct, total, accuracy))
