import kagglehub
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform

# Enable GPU
print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# Download latest version
dataset_path = kagglehub.dataset_download("techsash/waste-classification-data")

print("Path to dataset files:", dataset_path)

# Data Preprocessing
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
categories = os.listdir(os.path.join(dataset_path, 'DATASET/TRAIN'))

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    os.path.join(dataset_path, 'DATASET/TRAIN'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training')
val_generator = datagen.flow_from_directory(
    os.path.join(dataset_path, 'DATASET/TRAIN'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation')
test_generator = ImageDataGenerator(rescale=1./255).flow_from_directory(
    os.path.join(dataset_path, 'DATASET/TEST'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), kernel_initializer=GlorotUniform()),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(categories), activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.callbacks import ModelCheckpoint

# Define checkpoint callback
checkpoint_callback = ModelCheckpoint(
    "best_model.h5",  # Save the best model
    monitor="val_loss",  # Track validation loss
    save_best_only=True,  # Save only if validation loss improves
    mode="min",  # Lower validation loss is better
    verbose=1
)

# Define the path where you want to save the model
model_save_path = "waste_classification_model.h5"

# Train Model
epochs = 10

# Train Model on GPU
with tf.device('/GPU:0'):
    history = model.fit(train_generator, validation_data=val_generator, epochs=10, callbacks=[checkpoint_callback])


# Save the trained model
model.save(model_save_path)

print(f"Model saved at {model_save_path}")