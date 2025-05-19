import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt

def load_data():
    (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    return x_train, x_test

def rgb_to_gray(images):
    return np.dot(images[..., :3], [0.2989, 0.5870, 0.1140])

def build_model():
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 1)),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')
    ])
    return model

def train_model():
    x_train, x_test = load_data()
    x_train_gray = rgb_to_gray(x_train)
    x_train_gray = np.expand_dims(x_train_gray, axis=-1) / 255.0
    x_train_color = x_train / 255.0

    model = build_model()
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train_gray, x_train_color, epochs=10, batch_size=32, validation_split=0.2)

    return model

def colorize_image(model, gray_image):
    gray_image = np.expand_dims(gray_image, axis=0)
    colorized_image = model.predict(gray_image)
    return colorized_image[0]

if __name__ == "__main__":
    model = train_model()
    x_test = load_data()[1]
    gray_image = rgb_to_gray(x_test[0])
    colorized_image = colorize_image(model, gray_image)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Grayscale Image")
    plt.imshow(gray_image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title("Colorized Image")
    plt.imshow(colorized_image)
    plt.show()
