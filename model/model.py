# model.py

import json
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision
import matplotlib.pyplot as plt
from utils import label_to_str

class ModelCNN(pl.LightningModule):
    """
    A PyTorch Lightning module representing a Convolutional Neural Network (CNN) model.

    Args:
        num_classes (int): The number of classes for classification. 
            Default is 7 as there are 7 skin lesion classifications.

    Attributes:
        conv1 (nn.Conv2d): The first convolutional layer.
        conv2 (nn.Conv2d): The second convolutional layer.
        pool (nn.MaxPool2d): The max pooling layer.
        fc1 (nn.Linear): The first fully connected layer.
        fc2 (nn.Linear): The second fully connected layer.
        relu (nn.ReLU): The ReLU activation function.
        test_saved_outputs (list): A list to store the outputs and losses during testing.

    Methods:
        forward: Performs forward pass through the model.
        training_step: Defines the training step.
        validation_step: Defines the validation step.
        test_step: Defines the testing step.
        configure_optimizers: Configures the optimizer for training.
        show_batch_images: Displays a batch of images with their labels.
        on_test_epoch_end: Called at the end of each testing epoch to calculate and log the average test loss.

    """

    def __init__(self, num_classes=7):
        super(ModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 56 * 56, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

        self.test_saved_outputs = []

    def forward(self, x):
        """
        Performs forward pass through the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            x (torch.Tensor): The output tensor.

        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Args:
            batch (tuple): The input batch containing images and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            loss (torch.Tensor): The training loss.

        """
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step.

        Args:
            batch (tuple): The input batch containing images and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            loss (torch.Tensor): The validation loss.

        """
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Defines the testing step.

        Args:
            batch (tuple): The input batch containing images and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            loss (torch.Tensor): The testing loss.

        """
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        self.test_saved_outputs.append({'outputs': outputs, 'loss': loss.item()})
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer for training.
        Learning rate is set to 0.001.
        Returns:
            torch.optim.Optimizer: The optimizer.

        """
        return optim.Adam(self.parameters(), lr=0.001)

    @staticmethod
    def show_batch_images(batch_images, batch_labels):
        """
        Display a batch of images with their corresponding labels.

        Args:
            batch_images (torch.Tensor or tuple): The batch of images to display.
                If a tuple is provided, it should contain two elements: images and labels.
                If a single tensor is provided, it is assumed to be the images.
            batch_labels (torch.Tensor or None): The labels corresponding to the batch of images.
                If `batch_images` is a tuple, this argument is ignored.

        Returns:
            None

        """
        if isinstance(batch_images, tuple):
            images, labels = batch_images
        else:
            images, labels = batch_images, batch_labels

        labels = tuple(labels.tolist())
        labels = [label_to_str(label) for label in labels]
        grid = torchvision.utils.make_grid(images, nrow=8)
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(f'Labels: {labels}')
        plt.axis('off')
        plt.show()

    def on_test_epoch_end(self):
        """
        Function called at the end of each testing epoch.
        Calculates the average loss, logs it, and saves the results to a file.

        Returns:
            dict: A dictionary containing the average test loss.

        """
        all_outputs = self.test_saved_outputs
        losses = torch.tensor([output['loss'] for output in all_outputs])
        avg_loss = losses.mean()
        self.log('test_loss', avg_loss, prog_bar=True)
        results_dict = {'test_loss': avg_loss.item()}
        with open('test_results.json', 'w') as file:
            json.dump(results_dict, file)
        return {'avg_test_loss': avg_loss.item()}