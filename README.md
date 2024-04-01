# *Preparation*
## File Link 
https://drive.google.com/drive/folders/1YdXOgVXatZEjofkiRB2JtUp47-iQW8bD?usp=drive_link


## Prepare Directory:
1. Download dataset from https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000/
2. Unzip hamDataset.zip into 'data' folder which is placed in the base of the repo
3. Combine both folders containing the images into one folder called 'HAM10000_images'
4. Create a folder called 'models' that contains the 'trained_model_11ep{i}.pth' files
5. Create a folder called 'resnetmodels' that contains the 'trained_resnet_model_11ep{i}.pth' files
6. In the first function in model/utils.py called obtain_data_path(), add another if statement to set the path of data and image folders.
7. In the Skin Lesion Google Drive, specifically 'Models' folder, there are pretrained models already
    

## To Run Files:
1. On lines 8 of test.py and train.py, input 'A' or 'E' into the function depending on who is using it, so it will initialise the correct file path.

## To Use Google Collab:
1. Zip data folder prepared in steps 1-3 of "Prepare Directory" and name it hamDataset.zip
2. Upload that zip folder into your Google Drive at the base
3. Go into https://colab.research.google.com/drive/1U-F4G0bXJ5enApsUYAsGeoEcb8bz_0CT?usp=sharing
4. Run the Collab code, but in the drive.mount() line, you need to click accept and continue to get link Collab with Google Drive
5. Run the rest of the code
6. Can edit and run the training/testing code!

# *Results:*
## Average Accuracy:
Basic Homemade Model: 69.16%
Basic ResNet50 Architecture: 75.58%

## Best Accuracy: 
Basic Homemade Model: 76.55%
Basic Homemade Model: 71.96%