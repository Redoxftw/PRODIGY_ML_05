# --- split_data.py ---
# This is a one-time helper script to build our 'test' folder.
#
# It will:
# 1. Look at your 'data/train' folder.
# 2. Find all the dish subfolders (e.g., 'samosa', 'biryani').
# 3. Create a new 'data/test' folder with the same dish subfolders.
# 4. *Move* 20% of the images from each 'data/train/dish' folder
#    to the 'data/test/dish' folder.
#
# !! WARNING: This script MOVES files. Run it only ONCE. !!

import os
import glob
import shutil
from sklearn.model_selection import train_test_split

print("Starting dataset split...")

# --- 1. Configuration ---
BASE_PATH = "data"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")
TEST_SPLIT_SIZE = 0.2 # We'll use 20% of the images for testing

# --- 2. Find Classes and Create Test Folders ---
# First, find all the dish folders you have
try:
    class_names = [f for f in os.listdir(TRAIN_PATH) if os.path.isdir(os.path.join(TRAIN_PATH, f))]
    if not class_names:
        print(f"Error: No subfolders found in {TRAIN_PATH}")
        print("Please make sure your 'data/train' folder is full of your dish folders.")
        exit()
except FileNotFoundError:
    print(f"Error: The folder '{TRAIN_PATH}' was not found.")
    print("Please make sure your data structure is: PRODIGY_ML_05/data/train/...")
    exit()

print(f"Found {len(class_names)} classes: {class_names}")

# Create the 'data/test' folder
os.makedirs(TEST_PATH, exist_ok=True)

# Create all the subfolders inside 'data/test'
for class_name in class_names:
    os.makedirs(os.path.join(TEST_PATH, class_name), exist_ok=True)

print(f"Created 'data/test' folder and all class subfolders.")

# --- 3. Split and Move Files ---
for class_name in class_names:
    print(f"--- Processing class: {class_name} ---")
    
    # Get a list of all images for this class
    source_dir = os.path.join(TRAIN_PATH, class_name)
    all_image_paths = [os.path.join(source_dir, f) for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not all_image_paths:
        print(f"  WARNING: No images found for {class_name}. Skipping.")
        continue
        
    print(f"  Found {len(all_image_paths)} total images.")

    # 3. Split the list of images
    train_files, test_files = train_test_split(
        all_image_paths,
        test_size=TEST_SPLIT_SIZE,
        random_state=42 # So the split is repeatable
    )
    
    print(f"  Splitting into {len(train_files)} train / {len(test_files)} test files.")

    # 4. *MOVE* the test files
    destination_dir = os.path.join(TEST_PATH, class_name)
    for file_to_move in test_files:
        filename = os.path.basename(file_to_move)
        dest_path = os.path.join(destination_dir, filename)
        shutil.move(file_to_move, dest_path)

    print(f"  Successfully moved {len(test_files)} files to data/test/{class_name}")

print("\n--- SCRIPT COMPLETE ---")
print(f"Your 'data' folder is now split into 'train' and 'test'.")