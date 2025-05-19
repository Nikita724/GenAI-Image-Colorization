import numpy as np

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import layers

import cv2

import matplotlib.pyplot as plt

# Similar to basic colorization but with user-defined conditions


def buildconditional_model():
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])
    return model

def colorizewith_condition(model, gray_image, condition):
    # Apply condition to the model output (this is a placeholder)
    colorized_image = model.predict(np.expand_dims(gray_image, axis=0))[0]
    
    if condition == 'sky_blue':
        colorized_image[..., 0] = 0.5  # Adjust blue channel
    return colorized_image

if __name__ == "__main__":
    model = buildconditional_model()
    gray_image = np.random.rand(32, 32, 1)
    
    condition = 'sky_blue' 
    
    colorized_image = colorize_with_condition(model, gray_image, condition)

    plt.imshow(colorized_image)
    plt.title("Conditionally Colorized Image")
    plt.show()
