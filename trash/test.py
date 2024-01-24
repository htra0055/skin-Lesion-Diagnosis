import torch
import torch.nn as nn
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision
import matplotlib.pyplot as plt
from model import ModelCNN

# Instantiate the model
model = ModelCNN()

print(model.train_metadata_list[0].image)

# Example input size (batch_size, channels, height, width)
example_input = torch.randn(1, 3, 224, 224)

# Forward pass
output = model(example_input)

# Print the size of the output tensor
print("Output size:", output.size())




