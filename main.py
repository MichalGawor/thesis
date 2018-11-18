import argparse
from keras.applications import vgg19
from keras import backend as K
from keras.preprocessing.image import save_img
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time

from preprocessing import images_to_tensors, preprocess_images, deprocess_image
from utils import content_loss, style_loss, total_loss


parser = argparse.ArgumentParser(description='ArtificalArt.')
parser.add_argument('content_image_path', metavar='content', type=str,
                    help='Path to the image containing information to paint.')
parser.add_argument('style_image_path', metavar='style', type=str,
                    help='Path to the image with style used for painting.')
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
iterations = args.iter

total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight


# ____________________PREPARE IMAGES____________________
content_image, style_image, img_height, img_width = preprocess_images(CONTENT_IMAGE_PATH, STYLE_IMAGE_PATH)
content_image_tensor, style_image_tensor = images_to_tensors(content_image, style_image)

alternated_image = K.placeholder((1, img_height, img_width, 3))

input_tensor = K.concatenate([content_image, style_image, alternated_image], axis=0)


# ____________________LOAD MODEL____________________
model = vgg19.VGG19(input_tensor=input_tensor, weights='imagenet', include_top=False)
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

feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_features = layer_features[1, :, :, :]
    alternation_features = layer_features[2, :, :, :]
    _style_loss = style_loss(style_features, alternation_features, img_height, img_width)
    loss += (style_weight / len(feature_layers)) * _style_loss
loss += total_variation_weight * total_loss(alternated_image, img_height, img_width)

grads = K.gradients(loss, alternated_image)

outputs = [loss]
outputs += grads

loss_function_outputs = K.function([alternated_image], outputs)


def eval_loss_and_grads(loss_function_outputs, x, height, width):
    x = x.reshape((1, height, width, 3))
    outputs = loss_function_outputs([x])
    loss_value = outputs[0]
    # if len(outputs[1:]) == 1:
    grad_values = outputs[1].flatten().astype('float64')
    # else:
    #    grad_values = np.array(outputs[1:]).flatten().astype('float64')
    return loss_value, grad_values


# ____________________PARALLEL CALCULATION OF LOSS AND GRAD FOR OPTIMIZER____________________
# https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(loss_function_outputs, x, img_height, img_width)
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

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, content_image.flatten(),
                                     fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.copy(), img_height, img_width)
    fname = '_at_iteration_%d.png' % i
    save_img(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i, end_time - start_time))
