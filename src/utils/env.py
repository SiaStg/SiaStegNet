from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import logging
import os
import random

import numpy as np
import torch


def set_random_seed(seed=None):
    """Sets random seed for reproducibility.

    Args:
        seed (int, optional): Random seed.
    """
    if seed is None:
        seed = (
                os.getpid()
                + int(datetime.now().strftime("%S%f"))
                + int.from_bytes(os.urandom(2), "big")
        )
        logger = logging.getLogger(__name__)
        logger.info('Using a generated random seed {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())


def get_random_seed():
    return np.random.randint(2 ** 31)
