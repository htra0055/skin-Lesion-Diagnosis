import matplotlib.pyplot as plt
import torchmetrics
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
import pytorch_lightning as pl
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from torch import nn


class MyVGG(pl.LightningModule):
    def __init__(self, num_classes=7, pretrained=True):
        """
        Initializes a MyResNet model.

        Args:
            num_classes (int): The number of classes for classification.
            pretrained (bool, optional): Whether to use pre-trained weights for the VGG model. Defaults to True.
        """
        super(MyVGG, self).__init__()

        # Load pre-trained ResNet50 model
        self.vgg = models.vgg16(pretrained=pretrained)

        # Modify the last layer for transfer learning
        num_ftrs = self.vgg.classifier[-1].in_features
        self.vgg.classifier[-1] = nn.Linear(num_ftrs, num_classes)


        if pretrained:
            for param in self.vgg.parameters():
                param.requires_grad = False
            for param in self.vgg.classifier.parameters():
                param.requires_grad = True

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=7)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.vgg(x)

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
        loss = nn.CrossEntropyLoss()(outputs, labels)

        self.log('val_acc_step', self.accuracy,
                 on_step=True, on_epoch=False, prog_bar=True)
        self.log('val_loss', loss)
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



