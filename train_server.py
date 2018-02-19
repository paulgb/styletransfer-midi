#!/usr/bin/env python

import argparse
import mmap
from json import dump
from shutil import rmtree
from os import makedirs

from model import build_model, run_model
from images import image_from_matrix, matrix_from_image_file
from server import WEIGHTS_FILE, RESULTS_FILE

CONTENT_WEIGHT_MULTIPLIER = 1e-3
RUNS_PER_EPOCH = 20

layers = ['block{}_conv{}'.format(*x) for x in [
    (1, 2),
    (2, 1),
    (2, 2),
    (3, 1),
    (3, 3),
    (4, 1),
    (4, 3),
    (5, 1),
]]


class WeightProvider:
    def __init__(self, weight_file=WEIGHTS_FILE):
        fh = open(WEIGHTS_FILE, 'r+b')
        self.mm = mmap.mmap(fh.fileno(), 0)

    def get_weights(self):
        cw = [float(self.mm[i]) for i in range(8)]
        sw = [float(self.mm[i]) for i in range(8, 16)]
        return cw, sw


def package_layers(layers):
    return [{
        'name': name,
        'parts': [
            {
                'loss': float(content_loss),
                'weight': float(content_weight),
            },
            {
                'loss': float(style_loss),
                'weight': float(style_weight),
            },
        ]
    } for name, content_weight, content_loss, style_weight, style_loss in layers]


def main():
    rmtree('out')
    makedirs('out')

    parser = argparse.ArgumentParser()
    parser.add_argument('content_image')
    parser.add_argument('style_image')
    args = parser.parse_args()

    content_input = matrix_from_image_file(args.content_image, 0.1)
    style_input = matrix_from_image_file(args.style_image)

    model = build_model(layers, content_input, style_input)
    wp = WeightProvider()

    for i in range(10000):
        content_weights, style_weights = wp.get_weights()

        print('Using weights:\n{}\n{}'.format(content_weights, style_weights))
        outfile = f'out/run_{i:0>3}.png'

        rel_content_weights = [
            w * CONTENT_WEIGHT_MULTIPLIER for w in content_weights]

        img, losses = run_model(model, RUNS_PER_EPOCH, content_input.shape,
                                rel_content_weights, style_weights)
        img.save(outfile)

        r = {
            'img': outfile,
            'layers': package_layers(
                zip(layers, content_weights,
                    losses[:8], style_weights, losses[8:])
            ),
            'parts': ['content', 'style'],
        }

        with open(RESULTS_FILE, 'w') as fh:
            dump(r, fh)


if __name__ == '__main__':
    main()
