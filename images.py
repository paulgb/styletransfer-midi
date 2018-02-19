from PIL import Image
from keras.applications import vgg16
import numpy as np


def matrix_from_image_file(filename, scale=1.0):
    img = Image.open(filename)
    if scale != 1.0:
        w, h = img.size
        img = img.resize((int(w * scale), int(h * scale)))

    #pylint: disable=E1101
    return vgg16.preprocess_input(np.array(img, dtype=np.float32))


def image_from_matrix(img):
    #pylint: disable=E1101
    rn_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32)
    a = np.clip(img[:, :, [2, 1, 0]] + rn_mean, 0, 255).astype(np.uint8)
    return Image.fromarray(a)
