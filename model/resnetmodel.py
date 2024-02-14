import matplotlib.pyplot as plt
import torchmetrics
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
import pytorch_lightning as pl
import seaborn as sns
import matplotlib.pyplot as plt

from preparedata import SkinLesionDataModule
from sklearn.metrics import confusion_matrix
from torch import nn


class MyResNet(pl.LightningModule):
    def __init__(self, num_classes=7, pretrained=False):
        """
        Initializes a MyResNet model.

        Args:
            num_classes (int): The number of classes for classification.
            pretrained (bool, optional): Whether to use pre-trained weights for the ResNet model. Defaults to True.
        """
        super(MyResNet, self).__init__()

        # Load pre-trained ResNet50 model
        self.resnet = models.resnet50(pretrained=pretrained)

        # Modify the last layer for transfer learning
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=7)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        """
        Defines the training step.

        Args:
            batch (tuple): The input batch containing images and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The training loss.
        """
        images, labels = batch
        outputs = self(images)

        self.accuracy(outputs, labels)
        self.log('train_acc_step', self.accuracy,
                 on_step=True, on_epoch=False, prog_bar=True)

        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines the validation step.

        Args:
            batch (tuple): The input batch containing images and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The validation loss.
        """
        images, labels = batch
        outputs = self(images)

        self.accuracy(outputs, labels)
        self.log('val_acc_step', self.accuracy,
                 on_step=True, on_epoch=False, prog_bar=True)

        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss

    def test_step(self, batch, batch_idx):
        """
        Defines the testing step.

        Args:
            batch (tuple): The input batch containing images and labels.
            batch_idx (int): The index of the current batch.

        Returns:
            tuple: The predicted labels and true labels.
        """
        images, labels = batch
        outputs = self(images)
        _, preds = torch.max(outputs, 1)

        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.log('val_acc_step', self.accuracy,
                 on_step=True, on_epoch=False, prog_bar=True)

        self.test_step_preds.append(preds)
        self.test_step_labels.append(labels)

        return preds, labels

    def configure_optimizers(self):
        """
        Configures the optimizer for training.

        Returns:
            torch.optim.Optimizer: The optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_test_epoch_end(self) -> None:
        """
        Performs operations at the end of the testing epoch.
        """
        # Flatten the labels and predictions to create confusion matrix
        all_preds = self.test_step_preds
        all_labels = self.test_step_labels

        # Calculate the confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Visualize the confusion matrix
        class_names = ['blk', 'nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc']

        # Compute confusion matrix
        cm = confusion_matrix(all_labels.cpu(), all_preds.cpu())

        # Visualize the confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        plt.show()

        correct, total, accuracy = self.evaluate_accuracy(
            self.test_dataloader())
        print(correct, total, accuracy)

