import keras.backend as K


def gram_matrix(x):
    x = K.permute_dimensions(x, (0, 3, 1, 2))
    s = K.shape(x)
    feat = K.reshape(x, (s[0], s[1], s[2] * s[3]))
    feat_T = K.permute_dimensions(feat, (0, 2, 1))
    norm_factor = K.prod(K.cast(s[1:], K.floatx()))
    return K.batch_flatten(K.batch_dot(feat, feat_T) / norm_factor)
