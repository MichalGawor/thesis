from __future__ import print_function
from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse

from keras.applications import vgg19
from keras import backend as K

from preprocessing import *
from utils import *

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('content_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('style_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.025, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=1.0, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=1.0, required=False,
                    help='Total Variation weight.')

args = parser.parse_args()
CONTENT_IMAGE_PATH = args.content_image_path
STYLE_IMAGE_PATH = args.style_image_path
result_prefix = args.result_prefix
iterations = args.iter

total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

# ____________________OUTPUT SIZE____________________
width, height = load_img(CONTENT_IMAGE_PATH).size
img_height = 400
img_width = int(width * height / height)


# ____________________PREPARE IMAGES____________________
content_image = K.variable(preprocess_image(STYLE_IMAGE_PATH))
style_image = K.variable(preprocess_image(STYLE_IMAGE_PATH))
alternated_image = K.placeholder((1, img_height, img_width, 3))
input_tensor = K.concatenate([content_image,
                              style_image,
                              alternated_image], axis=0)


# ____________________LOAD MODEL____________________
model = vgg19.VGG19(input_tensor=input_tensor,
                    weights='imagenet', include_top=False)
print('VGG19 model has been successfully loaded.')


# ____________________GET LAYER-WISE OUTPUT____________________
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

loss = K.variable(0.)
layer_features = outputs_dict['block5_conv2']
content_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]


# ____________________CALCULATE LOSS____________________
loss += content_weight * content_loss(content_features,
                                      combination_features)

feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_features = layer_features[1, :, :, :]
    alternation_features = layer_features[2, :, :, :]
    _style_loss = style_loss(style_features, alternation_features, img_height, img_width)
    loss += (style_weight / len(feature_layers)) * _style_loss
loss += total_variation_weight * total_loss(alternated_image, img_height, img_width)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, alternated_image)

outputs = [loss]
if isinstance(grads, (list, tuple)):
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([alternated_image], outputs)


def eval_loss_and_grads(f_outputs, x, height, width):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


# ____________________PARALLEL CALCULATION OF LOSS AND GRAD FOR OPTIMIZER____________________
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


# ____________________APPLY ALTERNATION____________________
x = preprocess_image(CONTENT_IMAGE_PATH)

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.copy(), img_height, img_width)
    fname = result_prefix + '_at_iteration_%d.png' % i
    save_img(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
