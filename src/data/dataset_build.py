# TODO Select random sample of images
# TODO Ensure balanced classes
# TODO Move from source folder (train) to target folder (test, val)

import os
import numpy as np
import shutil


"""
Train/test split
"""

# Variables
# Source folder
source_folder = "src/data/type_data/train"
# validation folder path
train_path = "src/data/type_data/new_train"
test_path = "src/data/type_data/test"

# Constants
TEST_SIZE = 20

# Loop through each folder in img
for folder_name in os.listdir(source_folder):
    os.mkdir(os.path.join(train_path, folder_name))
    os.mkdir(os.path.join(test_path, folder_name))
    file_list = os.listdir(os.path.join(source_folder, folder_name))
    np.random.shuffle(file_list)
    # Create train/test splits
    test = file_list[:TEST_SIZE]
    train = file_list[TEST_SIZE:]
    # Copy train images to new folder
    for i, orig_file in enumerate(train):
        src_path = os.path.join(source_folder, folder_name, orig_file)
        dst_path = os.path.join(train_path, folder_name, orig_file)
        shutil.copy(src_path, dst_path)
    # Move test images to new folder
    for i, orig_file in enumerate(test):
        src_path = os.path.join(source_folder, folder_name, orig_file)
        dst_path = os.path.join(test_path, folder_name, orig_file)
        shutil.copy(src_path, dst_path)
