# %%
import os
import numpy as np
from tensorflow import keras
import pickle as pk
from tensorflow.keras import layers
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, ImageDataGenerator
import torch
use_gpu = torch.cuda.is_available() and not os.environ['USE_CPU']
use_gpu

# %%
# Define the paths to the DIV2K dataset
div2k_train_dir = r'C:\Users\cl502_03\Downloads\image super-resolution\DIV2K_train_HR'
div2k_val_dir = r'C:\Users\cl502_03\Downloads\image super-resolution\DIV2K_valid_HR'

# Image dimensions
input_shape = (1440, 2560, 3)

# Load DIV2K images
def load_div2k_images(directory):
    images = []
    for filename in os.listdir(directory):
        img = load_img(os.path.join(directory, filename), target_size=input_shape)
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
        images.append(img)
    return np.array(images)

x_train = load_div2k_images(div2k_train_dir)
x_val = load_div2k_images(div2k_val_dir)

with open("x_train.pkl", "wb") as file:
    pk.dump(x_train, file)

with open("x_val.pkl", "wb") as file:
    pk.dump(x_val, file)

# %%
input_shape = (1440, 2560, 3)
with open("x_train.pkl", "rb") as file:
    x_train = pk.load(file)

# Load x_val from the binary file
with open("x_val.pkl", "rb") as file:
    x_val = pk.load(file)

# %%
# Create a simple SRCNN model (you can replace this with a more advanced model)

model = keras.Sequential([
    layers.Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=input_shape),
    layers.Conv2D(32, (1, 1), activation='relu', padding='same'),
    layers.Conv2D(3, (5, 5), padding='same'),
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, x_train, epochs=2, batch_size=16, validation_data=(x_val, x_val))

# %%
model_path = r'C:\Users\cl502_03\Downloads\image super-resolution\model1.h5'

# Save the model
model.save(model_path)

print(f"Model saved to {model_path}")

# %%
model_path = r'model1.h5'
input_shape = (1080, 1920, 3)
model = load_model(model_path)

# %%
# Use the model for super-resolution
input_image = load_img(r"0801.png", target_size=input_shape)
input_image_array = img_to_array(input_image) / 255.0  # Normalize to [0, 1]
input_image_array = np.expand_dims(input_image_array, axis=0)  # Add batch dimension

super_res_image = model.predict(input_image_array)

# Save the super-resolved image
super_res_image = array_to_img(super_res_image[0])
super_res_image.save(r'car_1080_1920.png')

print("Super-resolution complete.")

# %%



