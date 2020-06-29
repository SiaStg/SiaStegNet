from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2

from ..matlab import S_UNIWARD


class CoverStegoDataset(Dataset):

    def __init__(self, cover_dir, stego_dir, transform=None):
        self._transform = transform

        self.images, self.labels = self.get_items(cover_dir, stego_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]))
        image = np.expand_dims(image, 2)  # (H, W, C)
        assert image.ndim == 3

        sample = {
            'image': image,
            'label': self.labels[idx]
        }

        if self._transform:
            sample = self._transform(sample)
        return sample

    @staticmethod
    def get_items(cover_dir, stego_dir):
        images, labels = [], []

        cover_names = sorted(os.listdir(cover_dir))
        if stego_dir is not None:
            stego_names = sorted(os.listdir(stego_dir))
            assert cover_names == stego_names

        file_names = cover_names
        if stego_dir is None:
            dir_to_label = [(cover_dir, 0), ]
        else:
            dir_to_label = [(cover_dir, 0), (stego_dir, 1)]
        for image_dir, label in dir_to_label:
            for file_name in file_names:
                image_path = osp.join(image_dir, file_name)
                if not osp.isfile(image_path):
                    raise FileNotFoundError('{} not exists'.format(image_path))
                images.append(image_path)
                labels.append(label)

        return images, labels


class OnTheFly(Dataset):

    def __init__(self, cover_dir, num=16, payload=0.4, transform=None):
        self._transform = transform
        self._num = num
        self._payload = payload

        self.cover_label = 0
        self.stego_label = 1

        self.images = self.get_items(cover_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = self.images[item]
        image = cv2.imread(image, flags=cv2.IMREAD_GRAYSCALE).astype(np.float64)  # HxW

        h, w = image.shape
        crop_h = np.random.randint(h * 3 // 4, h + 1)
        crop_w = np.random.randint(w * 3 // 4, w + 1)

        h0s = np.random.randint(0, h - crop_h + 1, (self._num,))
        w0s = np.random.randint(0, w - crop_w + 1, (self._num,))

        new_images, new_labels = [], []
        for h0, w0 in zip(h0s, w0s):
            cover_img = image[h0: h0 + crop_h, w0: w0 + crop_w]
            new_images.append(cover_img)
            new_labels.append(self.cover_label)

            stego_img = S_UNIWARD(cover_img, self._payload)
            new_images.append(stego_img)
            new_labels.append(self.stego_label)

        idxs = np.random.permutation(len(new_images))  # N

        new_images = np.stack(new_images, axis=0)  # NxHxW
        new_images = new_images[idxs]
        new_images = new_images[:, :, :, None]  # NxHxWxC

        new_labels = np.asarray(new_labels)
        new_labels = new_labels[idxs]

        sample = {
            'image': new_images,
            'label': new_labels,
        }

        if self._transform:
            sample = self._transform(sample)

        return sample

    @staticmethod
    def get_items(cover_dir):
        file_names = sorted(os.listdir(cover_dir))

        images = []
        for file_name in file_names:
            image_file = osp.join(cover_dir, file_name)
            if not osp.isfile(image_file):
                raise FileNotFoundError('{} not exists'.format(image_file))
            images.append(image_file)

        return images
