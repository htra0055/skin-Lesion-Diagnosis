import torchvision
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import os
import csv
import torch

def obtain_data_path(user):
    if user == 'A':
        metadata_file_path = 'data/hamDataset/HAM10000_metadata.csv'
        image_file_path = 'data/hamDataset/HAM10000_images'
        return metadata_file_path, image_file_path
    else:
        metadata_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_metadata.csv'
        image_file_path = '/Users/evelynhoangtran/Universe/MDN_projects/skin-Lesion-Diagnosis/data/hamDataset/HAM10000_images_part_1'

def label_to_str(label: int) -> str:
    """
    Function to convert the label (int) to a string describing the diagnosis.

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
    Converts a skin lesion diagnosis string to its corresponding label.

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


"""
TO SHOW THE FIRST BATCH. DEMONSTRATES THAT EACH BATCH IS DIFFERENT AND RANDOMISED. 
"""


def show_batch_images(dataloader):
    """
    Display a batch of images from a dataloader.

    Parameters:
    dataloader (torch.utils.data.DataLoader): The dataloader containing the images.

    Retuns:
    shows the images and prints the corresponding labels. 
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

###########################################################
# Testing Evaluation
###########################################################
def evaluate(model: pl.LightningModule, test_dataloader: torch.utils.data.DataLoader, filepath) -> float:
    model.load_state_dict(torch.load(filepath))
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
    correct, total, accuracy = evaluate_accuracy(all_preds_flat, all_labels_flat)
    print(f'Correct: {correct}, Total: {total}, Accuracy: {accuracy}')

    # Confusion matrix
    show_confusion_matrix(all_preds_flat, all_labels_flat, filepath)

    return (correct, total, accuracy)


def show_confusion_matrix(all_preds, all_labels, filename):
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


def evaluate_accuracy(predicted, labels) -> float:
    correct = 0
    total = 0

    total = labels.size(0)
    correct = (predicted == labels).sum().item()

    accuracy = correct / total
    return (correct, total, accuracy)


def get_filenames_in_folder(folder_path):
    filenames = os.listdir(folder_path)
    return filenames


def evaluate_in_folder(model, test_dataloader: torch.utils.data.DataLoader, folderpath, output_file) -> float:
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Correct', 'Total',
                        'Accuracy'])  # Writing header

    filenames = get_filenames_in_folder(folderpath)

    for file in filenames:
        filepath = f'{folderpath}/{file}'
        correct, total, accuracy = evaluate(model, test_dataloader, filepath)
        # save to csv file
        save_results_to_csv(file, output_file, (correct, total, accuracy))


def save_results_to_csv(file, output_file, data):
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        correct, total, accuracy = data
        writer.writerow((file, correct, total, accuracy))
