from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import logging
import math

import torchvision
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data import SequentialSampler

from .dataset import CoverStegoDataset, OnTheFly
from .transform import *
from .. import utils

logger = logging.getLogger(__name__)


class TrainingSampler(Sampler):

    def __init__(self, size, seed=None, shuffle=True):
        self._size = size
        self._shuffle = shuffle

        if seed is None:
            seed = utils.get_random_seed()
        self._seed = seed

    def __iter__(self):
        yield from itertools.islice(self._infinite_indices(), 0, None, 1)

    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)


class BalancedBatchSampler(BatchSampler):

    def __init__(self, sampler, group_ids, batch_size):
        """
        Args:
            sampler (Sampler): Base sampler.
            group_ids (list[int]): If the sampler produces indices in range [0, N),
                `group_ids` must be a list of `N` ints which contains the group id of each
                sample. The group ids must be a set of integers in [0, num_groups).
            batch_size (int): Size of mini-batch.
        """
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of torch.utils.data.Sampler, "
                             "but got sampler={}".format(sampler))

        self._sampler = sampler
        self._group_ids = np.asarray(group_ids)
        assert self._group_ids.ndim == 1
        self._batch_size = batch_size
        groups = np.unique(self._group_ids).tolist()
        assert batch_size % len(groups) == 0

        # buffer the indices of each group until batch size is reached
        self._buffer_per_group = {k: [] for k in groups}
        self._group_size = batch_size // len(groups)

    def __iter__(self):
        for idx in self._sampler:
            group_id = self._group_ids[idx]
            self._buffer_per_group[group_id].append(idx)
            if all(len(v) >= self._group_size for k, v in self._buffer_per_group.items()):
                idxs = []
                # Collect across all groups
                for k, v in self._buffer_per_group.items():
                    idxs.extend(v[:self._group_size])
                    del v[:self._group_size]

                idxs = np.random.permutation(idxs)
                yield idxs

    def __len__(self):
        raise NotImplementedError("len() of GroupedBatchSampler is not well-defined.")


def build_train_loader(cover_dir, stego_dir, batch_size=32, num_workers=0):
    transform = torchvision.transforms.Compose([
        RandomRot(),
        RandomFlip(),
        ToTensor(),
    ])
    dataset = CoverStegoDataset(cover_dir, stego_dir, transform)

    size = len(dataset)
    sampler = TrainingSampler(size)
    if stego_dir is not None:
        batch_sampler = BalancedBatchSampler(sampler, dataset.labels, batch_size)
    else:
        batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)
    epoch_length = math.ceil(size / batch_size)

    logger.info('Training set length is {}'.format(size))
    logger.info('Training epoch length is {}'.format(epoch_length))

    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        worker_init_fn=worker_init_reset_seed,
    )
    return train_loader, epoch_length


def build_otf_train_loader(cover_dir, num_workers=0):
    batch_size = 1
    transform = torchvision.transforms.Compose([
        RandomRot(),
        RandomFlip(),
        ToTensor(),
    ])
    dataset = OnTheFly(cover_dir, transform=transform)
    size = len(dataset)
    sampler = TrainingSampler(size)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)

    epoch_length = math.ceil(size // batch_size)

    logger.info('Training set length is {}'.format(size))
    logger.info('Training epoch length is {}'.format(epoch_length))

    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        worker_init_fn=worker_init_reset_seed,
    )
    return train_loader, epoch_length


def build_val_loader(cover_dir, stego_dir, batch_size=32, num_workers=0):
    transform = torchvision.transforms.Compose([
        ToTensor(),
    ])
    dataset = CoverStegoDataset(cover_dir, stego_dir, transform)

    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)

    logger.info('Testing set length is {}'.format(len(dataset)))

    test_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
    )
    return test_loader


def worker_init_reset_seed(worker_id):
    utils.set_random_seed(np.random.randint(2 ** 31) + worker_id)
