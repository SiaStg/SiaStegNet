from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import random


class RandomRot(object):

    def __call__(self, sample):
        rot = random.randint(0, 3)
        return {
            'image': np.rot90(sample['image'], rot, axes=[-3, -2]).copy(),
            'label': sample['label'],
        }


class RandomFlip(object):

    def __init__(self, p=0.5):
        self._p = p

    def __call__(self, sample):
        if random.random() < self._p:
            return {
                'image': np.flip(sample['image'], axis=-2).copy(),
                'label': sample['label'],
            }
        else:
            return sample


class ToTensor(object):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if image.ndim == 3:  # HxWxC
            image = image.transpose(2, 0, 1)
        else:  # NxHxWxC
            image = image.transpose(0, 3, 1, 2)
        return {
            'image': torch.from_numpy(image).type(torch.FloatTensor),
            'label': torch.tensor(label).long()
        }
