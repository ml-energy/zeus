# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Parts of the code here are adapted from https://github.com/NVIDIA/Megatron-LM/blob/v2.6/megatron/data/data_samplers.py

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import collections
import warnings
import datasets
from collections.abc import Mapping
from .. import mpu
from .. import print_rank_0


_SPLIT_DATA_SUPPORTED_MODEL_NAMES = [
    "ViTForImageClassification",
    "GPT2LMHeadModel",
    "T5ForConditionalGeneration",
    "BertForMaskedLM",
]


class MegatronPretrainingRandomSampler:

    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_size = data_parallel_size
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size
        self.epoch = 0

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        # self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples
        assert current_epoch_samples % self.micro_batch_times_data_parallel_size == 0

        # data sharding and random sampling
        bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                       * self.micro_batch_size
        bucket_offset = current_epoch_samples // self.data_parallel_size
        start_idx = self.data_parallel_rank * bucket_size
        
        g = torch.Generator()
        g.manual_seed(self.epoch)
        random_idx = torch.randperm(bucket_size, generator=g).tolist()
        idx_range = [start_idx + x for x in random_idx[bucket_offset:]]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []
    def set_epoch(self, epoch):
        self.epoch = epoch

class MegatronPretrainingSampler:


    def __init__(self, total_samples, consumed_samples, micro_batch_size,
                 data_parallel_rank, data_parallel_size):
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = consumed_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = self.micro_batch_size * \
                                                    data_parallel_size

        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)


    def __len__(self):
        return self.total_samples


    def __iter__(self):
        # a = []
        # batch = np.array(a, dtype=np.int64)
        # Last batch if not complete will be dropped.
        batch = []
        for idx in range(self.consumed_samples, self.total_samples):
            # s_time = time.time()
            batch.append(idx)
            if len(batch) == self.micro_batch_times_data_parallel_size:
                start_idx = self.data_parallel_rank * self.micro_batch_size
                end_idx = start_idx + self.micro_batch_size
                # print("batch sampler time(ms):", (time.time() - s_time) * 1000, "rank is ", self.data_parallel_rank)
                yield batch[start_idx:end_idx]
                batch = []

        # Check the last partial batch and see drop_last is set
        if len(batch) > 0 and not self.drop_last:
            start_idx, end_idx = self.get_start_end_idx()
            yield batch[start_idx:end_idx]


class DistributedDataset:

    def __init__(self, pipe_model, dataset, input_to_stage_dic, data_collator, args):

        # self.train_model = train_model
        self.pipe_model = pipe_model
        self.dataset = dataset
        self.data_collator = data_collator
        self.args = args
        for key in list(input_to_stage_dic):
            if not input_to_stage_dic.get(key):
                del input_to_stage_dic[key]
        self.input_to_stage_dic = input_to_stage_dic
        self.len_dataset = len(dataset)


    def get_sampler(self):
        self.dataset = self.split_dataset()

        if self.dataset is not None:
            generator = None

            # Build the sampler.
            if mpu.get_pipe_parallel_world_size() > 1:

                return MegatronPretrainingRandomSampler(
                    total_samples=len(self.dataset),
                    consumed_samples=0,
                    micro_batch_size=self.args.per_device_train_batch_size,
                    data_parallel_rank=mpu.get_data_parallel_rank(), #self.args.process_index,
                    data_parallel_size=mpu.get_data_parallel_world_size()
                )

            else:
                return DistributedSampler(
                    self.dataset,
                    num_replicas=mpu.get_data_parallel_world_size(),
                    rank=mpu.get_data_parallel_rank(),
                    seed=self.args.seed,
                )
        else:
            return None
    
    def get_dataloader(self):
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
        to distributed training if necessary) otherwise.
        Subclass and override this method if you want to inject some custom behavior.
        """

        train_sampler = self.get_sampler()
        if self.dataset is not None:
            return DataLoader(
                self.dataset,
                batch_sampler=train_sampler,
                collate_fn=self.data_collator if not isinstance(self.dataset, torch.utils.data.Dataset) else None, 
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            return None


    def split_dataset(self):
        print_rank_0("Using split input method ............")
        if not isinstance(self.dataset, (datasets.arrow_dataset.Dataset, torch.utils.data.Dataset)):
            warnings.warn("DistributedDataset ceate failed, current only support datasets.arrow_dataset.Dataset and torch.utils.data.Dataset, we recommend to use datasets to make dataset")
            return self.dataset
        if self.pipe_model.num_stages <= 1:
            warnings.warn("DistributedDataset failed, user should be used in pipeline model, but you only run with data distributed")
            return self.dataset
        if not isinstance(self.dataset, collections.abc.Sized):
            return None
        if isinstance(self.dataset, torch.utils.data.Dataset):
            warnings.warn("Because of using DistributedDataset, your default data collator will change to split data collator")
        # print(input_to_stage_dic, "\n", train_dataset.column_names, pipe_model.stage_id)

        labels_name = ['label', 'labels', 'label_ids']


        if not self.dataset:
            self.dataset = None
            return None

        if isinstance(self.dataset, datasets.arrow_dataset.Dataset):

            column_names = self.dataset.column_names

            if len(column_names) > 2:
                if self.pipe_model.is_last_stage():
                    for i in labels_name:
                        if i in column_names:
                            column_names.remove(i)
                    if 'special_tokens_mask' in self.dataset.column_names:
                        column_names.remove('special_tokens_mask')
                        column_names.remove('input_ids')
                    self.dataset = self.dataset.remove_columns(column_names)       # remove_columns之后需要重新赋值
                elif self.pipe_model.stage_id in self.input_to_stage_dic.keys():
                    for key, values in self.input_to_stage_dic.items():
                        for input_name in values:
                            if self.pipe_model.stage_id == key and input_name in column_names:
                                column_names.remove(input_name)
                    self.dataset = self.dataset.remove_columns(column_names)
                else:
                    self.dataset = None
            elif len(column_names) == 2:
                if self.pipe_model.is_last_stage():
                    for i in labels_name:
                        if i in column_names:
                            column_names.remove(i)
                    self.dataset = self.dataset.remove_columns(column_names)
                elif self.pipe_model.stage_id in self.input_to_stage_dic.keys():
                    for i in labels_name:
                        if i in column_names:
                            self.dataset = self.dataset.remove_columns(i)
                else:
                    self.dataset = None
            else:
                pass
            return self.dataset
        
        if isinstance(self.dataset, torch.utils.data.Dataset):
            # 仅第一和最后一个stage加载数据
            if self.pipe_model.stage_id not in self.input_to_stage_dic.keys() and not self.pipe_model.is_last_stage():
                self.dataset = None
            return self.dataset




