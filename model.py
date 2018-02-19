import keras.backend as K
import numpy as np
from keras.applications import vgg16
from keras.layers import Dense, Flatten, Input, Lambda, Reshape
from keras.models import Model
from keras.optimizers import Adam
from keras.layers.merge import concatenate

from util import gram_matrix
from images import image_from_matrix

NUM_WEIGHTS = 8


def make_content_targets(model, content_input):
    result = model.predict(np.expand_dims(content_input, 0))
    return [K.variable(r.flatten()) for r in result]


def get_content_loss(content_targets, content_actuals, weights):
    losses = []
    weights = [weights[0][i] for i in range(NUM_WEIGHTS)]
    for i, (target, actual, weight) in enumerate(zip(content_targets, content_actuals, weights)):
        flat_actual = Flatten()(actual)
        losses.append(Lambda(
            lambda m: weight * K.mean(K.square(target - m), axis=-1),
            output_shape=(1,),
            name=f'content_loss_{i}')(flat_actual))

    # return Lambda(sum, name=f'content_loss_sum', output_shape=(1,))(losses)
    return concatenate(losses, name='content_loss')


def make_style_targets(model, style_input):
    result = model.predict(np.expand_dims(style_input, 0))
    grams = []

    for i, r in enumerate(result):
        var = K.variable(K.eval(gram_matrix(r)))
        grams.append(Input(tensor=var, name=f'style_target_{i}'))

    return grams


def get_style_loss(style_targets, content_actuals, weights):
    losses = []
    weights = [weights[0][i] for i in range(NUM_WEIGHTS)]
    for i, (target, actual, weight) in enumerate(zip(style_targets, content_actuals, weights)):
        s = actual.shape
        style_actual = Lambda(
            gram_matrix, name=f'gram_matrix_{i}', output_shape=(int(s[3]) ** 2,))(actual)
        losses.append(Lambda(
            lambda m: weight * K.mean(K.square(target - m), axis=-1),
            output_shape=(1,),
            name=f'style_loss_{i}'
        )(style_actual))

    # return Lambda(sum, name=f'style_loss_sum', output_shape=(1,))(losses)
    return concatenate(losses, name='style_loss')


def build_feature_model(layers):
    vgg = vgg16.VGG16(include_top=False)
    layer_outputs = [vgg.get_layer(l).output for l in layers]
    model = Model(vgg.input, layer_outputs)
    model.trainable = False
    return model


def build_model(layers, content_input, style_input):
    (height, width, channels) = content_input.shape

    model = build_feature_model(layers)

    content_target = make_content_targets(model, content_input)
    style_targets = make_style_targets(model, style_input)

    content_weights_input = Input((NUM_WEIGHTS,))
    style_weights_input = Input((NUM_WEIGHTS,))

    x = inp = Input((1,))
    x = Dense(
        (height * width * channels),
        use_bias=False,
        name='image',
    )(x)
    x = Reshape((height, width, channels))(x)
    content_actual = model(x)

    content_loss = get_content_loss(
        content_target, content_actual, content_weights_input)
    style_loss = get_style_loss(
        style_targets, content_actual, style_weights_input)

    #loss = Lambda(lambda x: 1e-2 * x[0] + x[1])([content_loss, style_loss])

    loss = concatenate([content_loss, style_loss], name='loss')

    model = Model([inp, content_weights_input, style_weights_input], loss)
    #model.get_layer('image').set_weights([content_input.reshape((1, -1))])
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=5.0))

    return model


def weights_to_input(weights, steps):
    return np.tile(np.array(weights), [steps, 1])


def run_model(model, steps, shape, content_weights, style_weights):
    inputs = [
        np.ones(steps),
        weights_to_input(content_weights, steps),
        weights_to_input(style_weights, steps),
    ]
    model.fit(inputs, np.zeros((steps, NUM_WEIGHTS * 2)),
              epochs=1, batch_size=1)

    losses = model.predict_on_batch([
        np.ones(1),
        weights_to_input(content_weights, 1),
        weights_to_input(style_weights, 1),
    ])

    p = model.get_layer('image')

    img = K.eval(p.weights[0]).reshape(shape)

    return image_from_matrix(img), losses
