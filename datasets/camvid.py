import numpy as np
import os
import random

import torch
from torch.utils import data
import torchvision.transforms.functional as tf

from PIL import Image


class CamvidDataset(data.Dataset):
    def __init__(self, data, is_train=True):
        self.images, self.labels = [tpl[0] for tpl in data], \
                                   [tpl[1] for tpl in data]
        self.is_train = is_train

    @staticmethod
    def get_params(width, height, distortion_scale):
        """Get parameters for ``perspective`` for a random perspective transform.

        Args:
            width : width of the image.
            height : height of the image.

        Returns:
            List containing [top-left, top-right, bottom-right, bottom-left] of the original image,
            List containing [top-left, top-right, bottom-right, bottom-left] of the transformed image.
        """
        half_height = int(height / 2)
        half_width = int(width / 2)
        topleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(0, int(distortion_scale * half_height)))
        topright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(0, int(distortion_scale * half_height)))
        botright = (random.randint(width - int(distortion_scale * half_width) - 1, width - 1),
                    random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        botleft = (random.randint(0, int(distortion_scale * half_width)),
                   random.randint(height - int(distortion_scale * half_height) - 1, height - 1))
        startpoints = [(0, 0), (width - 1, 0),
                       (width - 1, height - 1), (0, height - 1)]
        endpoints = [topleft, topright, botright, botleft]
        return startpoints, endpoints

    def transform(self, index):
        input, target = map(lambda im: Image.open(im),
                            (self.images[index], self.labels[index]))
        tfm_input, tfm_target = (
            tf.resize(input, (360, 480)),
            tf.resize(target, (360, 480), interpolation=Image.NEAREST)
        )
        if self.is_train and False:  # Affine transformations
            max_dx = 0.1 * tfm_input.size[0]
            max_dy = 0.1 * tfm_input.size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
            rotation = random.uniform(0, 15)
            tfm_input, tfm_target = tf.affine(tfm_input, rotation, translations, 1, 0), \
                tf.affine(tfm_target, rotation, translations, 1, 0)
            if random.random() < 0.5:
                width, height = input.size
                startpoints, endpoints = self.get_params(width, height, 0.5)
                tfm_input, tfm_target = tf.perspective(tfm_input, startpoints, endpoints), \
                    tf.perspective(tfm_target, startpoints, endpoints)
        
        tfm_input, tfm_target = map(tf.to_tensor, (tfm_input, tfm_target))
        torch.clamp((255 * tfm_target), 0, 32, out=tfm_target)
        return tf.normalize(tfm_input, (0.5,), (0.5,)), tfm_target.long()

    def __getitem__(self, index):
        return self.transform(index)

    def __len__(self):
        return len(self.images)


def load_dataset(data_directory):
    with open(os.path.join(data_directory, "valid.txt"), "r") as f:
        val_names = [line.strip() for line in f]
    with open(os.path.join(data_directory, "codes.txt"), "r") as f:
        label_mapping = {l.strip(): i for i, l in enumerate(f)}
    data = []
    image_index_mapping = {}
    for im_f in os.listdir(os.path.join(data_directory, "images")):
        if im_f.split('.')[-1] != 'png':
            continue
        image_index_mapping[im_f] = len(data)
        fp = os.path.join(data_directory, "images", im_f)
        data.append(fp)
    for label_f in os.listdir(os.path.join(data_directory, "labels")):
        im_f = label_f.split('.')
        im_f[0] = '_'.join(im_f[0].split('_')[:-1])
        im_f = '.'.join(im_f)
        index = image_index_mapping[im_f]
        fp = os.path.join(data_directory, "labels", label_f)
        data[index] = (data[index], fp)
    val_indices = [image_index_mapping[name] for name in val_names]
    return data, val_indices, label_mapping
