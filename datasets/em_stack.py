import numpy as np
import os
import random
import tifffile as tiff

from torch.utils import data
import torchvision.transforms.functional as tf

from PIL import Image


class EMStackDataset(data.Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, x if y is None else y

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
        input, target = map(
            Image.fromarray, (self.x[..., index], self.y[..., index]))
        max_dx = 0.1 * input.size[0]
        max_dy = 0.1 * input.size[1]
        translations = (np.round(random.uniform(-max_dx, max_dx)),
                        np.round(random.uniform(-max_dy, max_dy)))
        rotation = random.uniform(0, 15)
        tfm_input, tfm_target = tf.affine(input, rotation, translations, 1, 0), \
            tf.affine(target, rotation, translations, 1, 0)
        if random.random() < 0.5:
            width, height = input.size
            startpoints, endpoints = self.get_params(width, height, 0.5)
            tfm_input, tfm_target = tf.perspective(input, startpoints, endpoints), \
                tf.perspective(target, startpoints, endpoints)
        tfm_input, tfm_target = map(tf.to_tensor, (input, target))
        return tf.normalize(tfm_input, (0.5,), (0.5,)), tfm_target.long()

    def __getitem__(self, index):
        return self.transform(index)

    def __len__(self):
        return self.x.shape[-1]


def load_dataset(data_directory):
    train_volume = tiff.imread(os.path.join(
        data_directory, 'train-volume.tif')).transpose((1, 2, 0))
    train_labels = tiff.imread(os.path.join(
        data_directory, 'train-labels.tif')).transpose((1, 2, 0))
    test_volume = tiff.imread(os.path.join(
        data_directory, 'test-volume.tif')).transpose((1, 2, 0))
    return train_volume, train_labels, test_volume
