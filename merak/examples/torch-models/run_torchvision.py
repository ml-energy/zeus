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

import Merak
from Merak import MerakTrainer, MerakArguments

from transformers import (
    HfArgumentParser
)

from config import get_config, get_wide_resnet
from data import build_loader


def parse_option(parser):
    group = parser.add_argument_group('Torchvision model training and evaluation script')
    group.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    group.add_argument('--data_path', type=str, default=None, help='path to data folder', )
    group.add_argument('--pp', type=int, default=4, help='Pipeline parallel degree')
    group.add_argument('--tp', type=int, default=1, help='Tensor parallel degree')
    group.add_argument('--dp', type=int, default=1, help='Data parallel degree')

    return parser


def main():
    # merge args
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    pp = args.pp
    tp = args.tp
    dp = args.dp
    Merak.init(pp, tp, dp)

    # using data config from swin transformer
    config = get_config(args)
    dataset_train, dataset_val, _, _, _ = build_loader(config)

    # Get Wide-ResNet variant
    num_layers, width_factor = map(int, training_args.model_name[11:].split("_"))
    model = get_wide_resnet(num_layers=num_layers, width_factor=width_factor)

    training_args.shard_count = num_layers * 10

    trainer = MerakTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_val,
    )
    # Train or profile
    if training_args.profile:
        trainer.profile()
    else:
        train_result = trainer.train()
        if train_result is not None:
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)


if __name__ == '__main__':
    main()
