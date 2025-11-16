# --- Task 5: Food Recognition & Calorie Estimator (Trainer) ---
# This is the trainer script for my final task.
# I'm building a custom Indian Food classifier!
#
# The Plan:
# 1. Load all images from my 'data/train' folder (my 80% split).
# 2. Load all images from my 'data/test' folder (my 20% split).
# 3. Use Transfer Learning (MobileNetV2), which I know works.
# 4. Use the *correct* pre-processing ([-1, 1] normalization)
#    which I learned about the hard way in Task 4.
# 5. Train a model to recognize my 12 food classes.
# 6. Save the final model as 'indian_food_model.h5'.

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import image_dataset_from_directory
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Ignoring warnings to keep the output clean
warnings.filterwarnings('ignore')

print("--- Task 5: Indian Food Classifier (Trainer) ---")
print(f"Using TensorFlow version: {tf.__version__}")

# --- 1. Setup ---
# Define constants for my project
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
TRAIN_DIR = 'data/train' # My 80%
TEST_DIR = 'data/test'   # My 20%
MODEL_FILE = 'indian_food_model.h5'

# --- 2. Load Data (Simple Version) ---
# My 'split_data.py' script already built my 'train' and 'test' folders.
# So now I can just point Keras to them.

print(f"Loading training data from {TRAIN_DIR}...")
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    color_mode='rgb' # These are COLOR images!
)

print(f"Loading test data from {TEST_DIR}...")
test_ds = image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int',
    color_mode='rgb' # Also color
)

# Get the class names (e.g., 'biryani', 'dosa', etc.)
# This is SUPER important for my 'calorie_database.py'
class_names = train_ds.class_names
print(f"Found {len(class_names)} classes:")
print(class_names)
print("!! MAKE SURE these 12 class names EXACTLY match your 'calorie_database.py' file !!")


# --- 3. Pre-process the Data ---
# These are real photos, so they are 3-channel (RGB).
# We just need to apply the correct MobileNetV2 normalization (pixels -> [-1, 1])
# No more grayscale nonsense!

def preprocess_fn(image, label):
    # This is the *only* step we need.
    image = preprocess_input(image) 
    return image, label

train_ds = train_ds.map(preprocess_fn)
test_ds = test_ds.map(preprocess_fn)

# Cache for speed
train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
print("Data pre-processing complete (Normalized for MobileNetV2).")

# --- 4. Build the Model (Transfer Learning) ---
print("Building model with MobileNetV2 base...")

# 1. Load the Base Model (MobileNetV2)
base_model = MobileNetV2(
    input_shape=(224, 224, 3), # 3 = RGB color
    include_top=False,
    weights='imagenet'
)
# 2. Freeze the Base Model
base_model.trainable = False

# 3. Create my new "Head"
inputs = Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x) # Flattens the output
x = Dense(128, activation='relu')(x) # A hidden layer to learn
x = Dropout(0.5)(x) # Helps prevent overfitting
# The output layer MUST have 12 units (one for each food)
outputs = Dense(len(class_names), activation='softmax')(x)
model = Model(inputs, outputs)

# --- 5. Compile the Model ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Good for integer labels (0, 1, 2...)
    metrics=['accuracy']
)

print("Model built and compiled successfully.")
model.summary()

# --- 6. Train the Model ---
print("\n--- Starting Model Training ---")
print("This will take 10-20 minutes...")

EPOCHS = 10 # 10 epochs should be a good start
history = model.fit(
    train_ds,
    validation_data=test_ds, # Use our real test set
    epochs=EPOCHS
)

print("--- Training Complete ---")

# --- 7. Evaluate the Model ---
print("Evaluating model on the *test* dataset...")
loss, accuracy = model.evaluate(test_ds)

print(f"\n--- Model Evaluation ---")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")

# --- 8. Save the Model ---
model.save(MODEL_FILE)
print(f"\nModel saved successfully to '{MODEL_FILE}'")
print("\n--- Task 5 Trainer Script Finished ---")
print("You can now run 'streamlit run app.py' to use this model!")