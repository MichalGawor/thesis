from keras import backend as K


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
