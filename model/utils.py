import torchvision
from matplotlib import pyplot as plt
import numpy as np 


"""
CONVERTS LABELS TO INT - MAY BE UNNECESSARY
"""
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
        print(f'Index: {index} Label: {label}')  # Assuming labels are stored as tensors

    # # call function on our images
    images = torchvision.utils.make_grid(images)

    images = images / 2 + 0.5 # unnormalize
    npimg = images.numpy() # convert to numpy objects
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

