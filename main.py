import argparse
from keras.applications import vgg19
from keras import backend as K
import keras.preprocessing.image as kpi
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import os


parser = argparse.ArgumentParser(description='ArtificalArt.')
parser.add_argument('content_image_path', metavar='content', type=str,
                    help='Path to the image containing information to paint.')
parser.add_argument('style_image_path', metavar='style', type=str,
                    help='Path to the image with style used for painting.')
parser.add_argument('email', metavar='email', type=str, 
                    help='Email address where the result will be sent') 
parser.add_argument('--iter', type=int, default=61, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--cw', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--sw', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv', type=float, default=1.0, required=False,
                    help='Total Variation weight.')

args = parser.parse_args()
CONTENT_IMAGE_PATH = args.content_image_path
STYLE_IMAGE_PATH = args.style_image_path
EMAIL_ADDRESS = args.email
iterations = args.iter


total_variation_weight = args.tv
style_weight = args.sw
content_weight = args.cw

os.environ["CUDA_VISIBLE_DEVICES"]="0"


def output_image_dimensions(width, height, desired_height):
    if desired_height > 0:
        return desired_height, int(width * desired_height / height)
    else:
        return height, width


def image_to_k_variable(image):
    return K.variable(image)


base_width, base_height = kpi.load_img(CONTENT_IMAGE_PATH).size
img_height, img_width = output_image_dimensions(base_width, base_height, desired_height=400)


def gram_matrix(features):
    feature_map = K.batch_flatten(K.permute_dimensions(features, (2, 0, 1)))
    return K.dot(feature_map, K.transpose(feature_map))


def content_loss(content, alternation):
    return K.sum(K.square(alternation - content))


def style_loss(style, alternation, height, width):
    G = gram_matrix(style)
    A = gram_matrix(alternation)
    return K.sum(K.square(G - A)) / (4.0 * (3.0**2) * (width * height)**2)


def total_loss(features, height, width):
    a = K.square(features[:, :height - 1, :width - 1, :] - features[:, 1:, :width - 1, :])
    b = K.square(features[:, :height - 1, :width - 1, :] - features[:, :height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def preprocess_image(image_path):
    image = kpi.load_img(image_path, target_size=(img_height, img_width))
    image = kpi.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return vgg19.preprocess_input(image)


def deprocess_image(output_matrix, height, width):
    output_matrix = output_matrix.reshape((height, width, 3))
    output_matrix[:, :, 0] += 103.939
    output_matrix[:, :, 1] += 116.779
    output_matrix[:, :, 2] += 123.68
    output_matrix = output_matrix[:, :, ::-1]
    output_matrix = np.clip(output_matrix, 0, 255).astype('uint8')
    return output_matrix


# ____________________PREPARE IMAGES____________________
content_image = preprocess_image(CONTENT_IMAGE_PATH)
style_image = preprocess_image(STYLE_IMAGE_PATH)

content_image_tensor = image_to_k_variable(content_image)
style_image_tensor = image_to_k_variable(style_image)
alternated_image_tensor = K.placeholder((1, img_height, img_width, 3))

input_tensor = K.concatenate([content_image_tensor, style_image_tensor, alternated_image_tensor], axis=0)


# ____________________LOAD MODEL____________________
model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
print('VGG19 model has been successfully loaded.')


# ____________________GET LAYER-WISE OUTPUT____________________
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

loss = K.variable(0.0)
layer_features = outputs_dict['block5_conv2']
content_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]


# ____________________CALCULATE LOSS____________________
loss += (content_weight * content_loss(content_features, combination_features))

feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_features = layer_features[1, :, :, :]
    alternation_features = layer_features[2, :, :, :]
    _style_loss = style_loss(style_features, alternation_features, img_height, img_width)
    loss += ((style_weight / len(feature_layers)) * _style_loss)
loss += (total_variation_weight * total_loss(alternated_image_tensor, img_height, img_width))

grads = K.gradients(loss, alternated_image_tensor)
outputs = [loss]

if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

loss_function_outputs = K.function([alternated_image_tensor], outputs)


def eval_loss_and_grads(x):
    x = x.reshape((1, img_height, img_width, 3))
    outs = loss_function_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


# ____________________PARALLEL CALCULATION OF LOSS AND GRAD FOR OPTIMIZER____________________
# https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()


X = preprocess_image(CONTENT_IMAGE_PATH)

# ____________________APPLY ALTERNATION____________________

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, X.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    if i % 10 == 0:
        img = deprocess_image(x.copy(), img_height, img_width)  
        fname = str(content_weight) + '_' + str(style_weight) + '_at_iteration_%d.png' % i
        kpi.save_img(fname, img)
        print('Image saved as', fname)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
    if K.image_data_format() == 'channels_first':
        print("ERROR")
