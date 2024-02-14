import os

# Specify the folder path
folder_path = "confusion_matrix"

# Loop through the range of numbers from 0 to 16 (inclusive)
for i in range(1,17):
    # Generate the old and new file names
    old_file_name_0 = f"trained_model_5ep{i}.pth.png"
    new_file_name_0 = f"trained_model_11ep{i}.png"



    # Construct the full paths for the files
    old_file_path_0 = f'{folder_path}/{old_file_name_0}'
    new_file_path_0 = f'{folder_path}/{new_file_name_0}'



    # Rename the files
    os.rename(old_file_path_0, new_file_path_0)

    print(f"Renamed {old_file_name_0} to {new_file_name_0}")