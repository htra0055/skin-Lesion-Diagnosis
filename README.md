# *Preparation*
## File Link 
https://drive.google.com/drive/folders/1YdXOgVXatZEjofkiRB2JtUp47-iQW8bD?usp=drive_link


## Prepare Directory:
1. Unzip hamDataset.zip into 'data' folder
2. Combine both folders containing the images into one folder called 'HAM10000_images'
3. Create a folder called 'models' that contains the 'trained_model_11ep{i}.pth' files
4. Create a folder called 'resnetmodels' that contains the 'trained_resnet_model_11ep{i}.pth' files

## To Run Files:
1. On lines 8 of test.py and train.py, input 'A' or 'E' into the function depending on who is using it, so it will initialise the correct file path.

# *Results:*
## Average Accuracy:
Basic Homemade Model: 69.16%
Basic ResNet50 Architecture: 75.58%

## Best Accuracy: 
Basic Homemade Model: 76.55%
Basic Homemade Model: 71.96%