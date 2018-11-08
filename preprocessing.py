import numpy as np
import keras as K
import keras.preprocessing.image as kpi
from keras.applications import vgg19


def output_image_dimensions(width, height, desired_height=400):
    return desired_height, int(width * desired_height / height)


def images_to_tensors(information_image, style_image):
    return K.variable(information_image), K.variable(style_image)


def preprocess_image(image):
    image = kpi.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return vgg19.preprocess_input(image)


def preprocess_images(information_image_path, style_image_path, desired_height=400):
    base_width, base_height = kpi.load_img(information_image_path).size
    height, width = output_image_dimensions(base_width, base_height, desired_height)
    style_image = kpi.load_img(style_image_path, target_size=(height, width))
    information_image = kpi.load_img(information_image_path, target_size=(height, width))

    information_image = preprocess_image(information_image)
    style_image = preprocess_image(style_image)

    information_image, style_image = images_to_tensors(information_image, style_image)

    return information_image, style_image, height, width


def deprocess_image(output_matrix, height, width):
    output_matrix = output_matrix.reshape((height, width, 3))
    output_matrix[:, :, 0] += 123.68
    output_matrix[:, :, 1] += 116.779
    output_matrix[:, :, 2] += 103.939
    # output_matrix = output_matrix[:, :, ::-1]
    output_matrix = np.clip(output_matrix, 0, 255).astype('uint8')
    return output_matrix
