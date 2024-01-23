# model.py

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision
import matplotlib.pyplot as plt

class ModelCNN(pl.LightningModule):
    def __init__(self, num_classes=7):
        super(ModelCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 56 * 74, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

       

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)

        # print(x.size())  # Print the size of the tensor before the linear layer
        # torch.Size([64, 128, 56, 74]) 

        x = x.view(x.size(0), -1) # Dynamically calculate the size for the linear layer
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
    
    @staticmethod
    def show_batch_images(batch_images, batch_labels):
        if isinstance(batch_images, tuple):
            images, labels = batch_images
        else:
            images, labels = batch_images, batch_labels

        grid = torchvision.utils.make_grid(images, nrow=8)
        plt.imshow(grid.permute(1, 2, 0))
        plt.title(f'Labels: {labels}')
        plt.axis('off')
        plt.show()

       
       
