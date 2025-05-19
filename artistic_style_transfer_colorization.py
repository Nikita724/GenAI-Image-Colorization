import numpy as np

import tensorflow as tf

from tensorflow import keras

import matplotlib.pyplot as plt

import tensorflow_hub as hub

def load_image(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(256, 256))
    
    img = keras.preprocessing.image.img_to_array(img) / 255.0
    
    return np.expand_dims(img, axis=0)

def style_transfer(content_image, style_image):
    model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    
    stylized_image = model(tf.constant(content_image), tf.constant(style_image))[0]
    
    return stylized_image.numpy()

if __name__ == "__main__":
    content_image = load_image('path_to_grayscale_image.jpg')  # Replace with your image path
    
    style_image = load_image('path_to_style_image.jpg')  # Replace with your style image path
    
    stylized_image = style_transfer(content_image, style_image)

    plt.imshow(stylized_image[0])
    
    plt.title("Stylized Colorized Image")
    
    plt.show()
