import random
from typing import Iterator, Optional

import numpy as np
import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from opensora.datasets import collate
from opensora.datasets.collate import default_collate
from .sampler import VariableNBAClipsBatchSampler


class NBAClipsDataloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        return iter(self.batch_sampler)


def prepare_variable_dataloader(
    dataset,
    batch_size: int,
    bucket_config,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    process_group=None,
    num_bucket_build_workers=1,
    **kwargs,
):
    """
    TODO: we arn't usign the `batch_size` arg at the moment.
    """

    _kwargs = kwargs.copy()
    process_group = process_group or _get_default_group()
    batch_sampler = VariableNBAClipsBatchSampler(
        dataset,
        bucket_config,
        num_replicas=process_group.size(),
        rank=process_group.rank(),
        shuffle=shuffle,
        seed=seed,
        drop_last=drop_last,
        verbose=False,
        num_bucket_build_workers=num_bucket_build_workers,
    )

    # deterministic dataloader
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    # TODO: batch size is mututally exclusive with batch sampler
    return torch.utils.data.DataLoader(
        dataset,
        collate_fn=default_collate,
        batch_sampler=batch_sampler,
        worker_init_fn=seed_worker,
        pin_memory=pin_memory,
        num_workers=num_workers,
        **_kwargs,
    )
    
def safe_collate_fn(dataloader: torch.utils.data.DataLoader):
    pass
