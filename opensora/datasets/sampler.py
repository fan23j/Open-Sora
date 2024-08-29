import warnings
import torch
import torch.distributed as dist

from pprint import pprint
from collections import OrderedDict, defaultdict
from typing import Iterator, List, Optional, Tuple

from pandarallel import pandarallel
from torch.utils.data import DistributedSampler
from opensora.utils.sampler_entities import MicroBatch
from .bucket import Bucket
from .datasets import NBAClipsDataset

HARD_CODED_NUM_BATCHES = 1


def apply(data, method=None, frame_interval=None, seed=None, num_bucket=None):
    return method(
        data["num_frames"],
        data["height"],
        data["width"],
        frame_interval,
        seed + data["id"] * num_bucket,
    )


class VariableNBAClipsBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset: NBAClipsDataset,
        bucket_config: dict,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
        num_bucket_build_workers: int = 1,
    ) -> None:
        super().__init__(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )
        self.dataset: NBAClipsDataset = dataset
        self.bucket = Bucket(bucket_config)
        self.verbose: bool = verbose
        self.last_micro_batch_access_index: int = 0
        self.approximate_num_batch: Optional[int] = None
        self._get_num_batch_cached_bucket_sample_dict: Optional[dict] = None
        self.num_bucket_build_workers: int = num_bucket_build_workers

    def group_by_bucket(self) -> dict[Tuple[str, int, str], List[str]]:
        """
        Place samples into buckets containing similar resolution / # frames.
        TODO: we place all samples into the same bucket for now.
        """
        
        assert (
            type(self.dataset) is NBAClipsDataset
        ), f"Error: dataset.ann is {type(self.dataset)}"
        
        # HACK: hard-coding buckets for now
        bucket_sample_dict = {
            (
                "360p",
                4,
                "1.00",
            ): self.dataset.filtered_dataset.filtered_clip_annotations_file_paths[:100]
        }
        return bucket_sample_dict

    def get_num_batch(self) -> int:
        """
        TODO: hard-code the number of batches to 1.
        """
        return HARD_CODED_NUM_BATCHES

    def __iter__(self) -> Iterator[List[MicroBatch]]:

        if self._get_num_batch_cached_bucket_sample_dict is not None:
            bucket_sample_dict = self._get_num_batch_cached_bucket_sample_dict
            self._get_num_batch_cached_bucket_sample_dict = None
        else:
            bucket_sample_dict = self.group_by_bucket()
            if self.verbose:
                self._print_bucket_info(bucket_sample_dict)

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_micro_batch_count = OrderedDict()
        bucket_last_consumed = OrderedDict()

        # process the samples
        for bucket_id, data_list in bucket_sample_dict.items():
            # handle droplast
            bs_per_gpu = self.bucket.get_batch_size(bucket_id)
            remainder = len(data_list) % bs_per_gpu
            if remainder > 0:
                if not self.drop_last:
                    # if there is remainder, we pad to make it divisible
                    data_list += data_list[: bs_per_gpu - remainder]
                else:
                    # we just drop the remainder to make it divisible
                    data_list = data_list[:-remainder]
            bucket_sample_dict[bucket_id] = data_list
            # handle shuffle
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in data_indices]
                bucket_sample_dict[bucket_id] = data_list

            # compute how many micro-batches each bucket has
            num_micro_batches = len(data_list) // bs_per_gpu
            bucket_micro_batch_count[bucket_id] = num_micro_batches

        # compute the bucket access order
        # each bucket may have more than one batch of data
        # thus bucket_id may appear more than 1 time
        bucket_id_access_order = []
        for bucket_id, num_micro_batch in bucket_micro_batch_count.items():
            bucket_id_access_order.extend([bucket_id] * num_micro_batch)

        # randomize the access order
        if self.shuffle:
            bucket_id_access_order_indices = torch.randperm(
                len(bucket_id_access_order), generator=g
            ).tolist()
            bucket_id_access_order = [
                bucket_id_access_order[i] for i in bucket_id_access_order_indices
            ]

        # make the number of bucket accesses divisible by dp size
        remainder = len(bucket_id_access_order) % self.num_replicas
        if remainder > 0:
            if self.drop_last:
                bucket_id_access_order = bucket_id_access_order[
                    : len(bucket_id_access_order) - remainder
                ]
            else:
                bucket_id_access_order += bucket_id_access_order[
                    : self.num_replicas - remainder
                ]

        # prepare each batch from its bucket
        # according to the predefined bucket access order
        num_iters = len(bucket_id_access_order) // self.num_replicas
        start_iter_idx = self.last_micro_batch_access_index // self.num_replicas

        # re-compute the micro-batch consumption
        # this is useful when resuming from a state dict with a different number of GPUs
        self.last_micro_batch_access_index = start_iter_idx * self.num_replicas
        for i in range(self.last_micro_batch_access_index):
            bucket_id: int = bucket_id_access_order[i]
            bucket_bs = self.bucket.get_batch_size(bucket_id)
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs
            else:
                bucket_last_consumed[bucket_id] = bucket_bs

        for i in range(start_iter_idx, num_iters):
            bucket_access_list = bucket_id_access_order[
                i * self.num_replicas : (i + 1) * self.num_replicas
            ]
            self.last_micro_batch_access_index += self.num_replicas

            # compute the data samples consumed by each access
            bucket_access_boundaries = []
            for bucket_id in bucket_access_list:
                bucket_bs = self.bucket.get_batch_size(bucket_id)
                last_consumed_index = bucket_last_consumed.get(bucket_id, 0)
                bucket_access_boundaries.append(
                    [last_consumed_index, last_consumed_index + bucket_bs]
                )

                # update consumption
                if bucket_id in bucket_last_consumed:
                    bucket_last_consumed[bucket_id] += bucket_bs
                else:
                    bucket_last_consumed[bucket_id] = bucket_bs

            # compute the range of data accessed by each GPU
            bucket_id = bucket_access_list[self.rank]
            boundary = bucket_access_boundaries[self.rank]
            cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0] : boundary[1]]

            # encode t, h, w into the sample index
            real_t, real_h, real_w = self.bucket.get_thw(bucket_id)

            cur_micro_batch: List[MicroBatch] = [
                MicroBatch(i, real_t, real_h, real_w)
                for i, idx in enumerate(cur_micro_batch)
            ]
            yield cur_micro_batch

        self._reset()

    def _reset(self):
        self.last_micro_batch_access_index = 0

    def state_dict(self, num_steps: int) -> dict:
        # the last_micro_batch_access_index in the __iter__ is often
        # not accurate during multi-workers and data prefetching
        # thus, we need the user to pass the actual steps which have been executed
        # to calculate the correct last_micro_batch_access_index
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "last_micro_batch_access_index": num_steps * self.num_replicas,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)

    def set_epoch(self, epoch: int) -> None:
        super().set_epoch(epoch)

    def __len__(self) -> int:
        warnings.warn(
            "The length of VariableVideoBatchSampler is dynamic and may not be accurate. Return the max value."
        )
        min_batch_size = None
        for v in self.bucket.bucket_bs.values():
            for bs in v.values():
                if bs is not None and (min_batch_size is None or bs < min_batch_size):
                    min_batch_size = bs
        if self.drop_last:
            return len(self.dataset) // min_batch_size
        else:
            return (len(self.dataset) + min_batch_size - 1) // min_batch_size
