import matplotlib.pyplot as plt
import torchmetrics
import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
from preparedata import SkinLesionDataModule
from torch import nn
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


class MyResNet(pl.LightningModule):
    def __init__(self, num_classes, pretrained=True):
        super(MyResNet, self).__init__()

        # Load pre-trained ResNet50 model
        self.resnet = models.resnet50(pretrained=pretrained)

        # Modify the last layer for transfer learning
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=7)

        self.test_step_preds = []
        self.test_step_labels = []

    def forward(self, x):
        return self.resnet(x)

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
            loss (torch.Tensor): The validation loss.

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
            loss (torch.Tensor): The testing loss.

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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def on_test_epoch_end(self) -> None:
        # Flatten the labels and predictions to create confusion matrix
        all_preds = self.test_step_preds
        all_labels = self.test_step_labels

        # all_labels = labels.view(-1)
        # all_preds = preds.view(-1)

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

    def evaluate_accuracy(self, test_dataloader):
        self.eval()  # Set model to evaluation mode
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient calculation during inference

            data = test_dataloader
            inputs, labels = data
            outputs = self(inputs)  # Forward pass
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return correct, total, accuracy


def main():
    metadata_file_path = 'data/hamDataset/HAM10000_metadata.csv'
    image_file_path = 'data/hamDataset/HAM10000_images'

    # Create a data module instance
    data_module = SkinLesionDataModule(
        metadata_file=metadata_file_path,
        image_folder=image_file_path,
        batch_size=16,
        num_workers=3
    )
    # Setup the data module by loading metadata and splitting into train and validation sets
    data_module.setup()

    # Example usage
    num_classes = 7  # Example number of classes
    model = MyResNet(num_classes)

    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()

    # Train the data
    trainer = pl.Trainer(max_epochs=10)
    trainer.fit(model, train_dataloader, val_dataloader)

    # Saving the trained model into file 'trained_model.pth'
    torch.save(model.state_dict(), 'trained_model_2_ep.pth')


if __name__ == "__main__":
    main()
