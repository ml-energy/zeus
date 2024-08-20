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

# using our distributed trainer
import Merak
from Merak import MerakArguments, MerakTrainer
from utils import create_tokenizer, load_wikitext, preprocessing_datasets
from config import load_config_and_model

from transformers import (
    default_data_collator,
    set_seed,
    HfArgumentParser,
)
from bloom.tokenization_bloom_fast import BloomTokenizerFast


def parse_option(parser):
    # easy config modification
    # parser.add_argument('--cache-dir', type=str, help='where to save cache')
    # parser.add_argument('--dataset-name', type=str, help='name of dataset from the datasets package')
    parser.add_argument('--validation-split-percentage', type=int, default=5, help='split data for validation')
    parser.add_argument('--pp', type=int, default=4, help='Pipeline parallel degree')
    parser.add_argument('--tp', type=int, default=1, help='Tensor parallel degree')
    parser.add_argument('--dp', type=int, default=1, help='Data parallel degree')

    return parser

def main():
    # merge args
    hfparser = HfArgumentParser(MerakArguments)
    parser = parse_option(hfparser)
    training_args, args = parser.parse_args_into_dataclasses()

    # init dist
    pp = args.pp
    tp = args.tp
    dp = args.dp
    Merak.init(pp, tp, dp)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # load data
    raw_datasets = load_wikitext(training_args.cache_dir, args.validation_split_percentage)

    # Load model config and instantiate model.
    config, model = load_config_and_model(training_args.model_name)

    # Set shard_count based on number of layers
    # bloom-560m  => 26 layers (word embedding, 24 transformer decoders, lm_head)
    # bloom-1b1   => 26 layers (word embedding, 24 transformer decoders, lm_head)
    # bloom-1b7   => 26 layers (word embedding, 24 transformer decoders, lm_head)
    # bloom-3b    => 32 layers (word embedding, 30 transformer decoders, lm_head)
    # bloom-7b1   => 32 layers (word embedding, 30 transformer decoders, lm_head)
    # bloom       => 72 layers (word embedding, 70 transformer decoders, lm_head)
    training_args.shard_count = 1 + config.n_layer + 1
    training_args.num_transformers = config.n_layer
    training_args.num_initial_embeddings = 1
    training_args.seq_length = 1024

    # create tokenizer
    tokenizer = create_tokenizer(training_args.cache_dir, "bigscience/bloom", config)

    # Preprocessing the datasets.
    train_dataset, eval_dataset = preprocessing_datasets(raw_datasets, tokenizer, training_args.model_name)


    # Initialize our Trainer.
    trainer = MerakTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
    )

    # Train or profile
    if training_args.profile:
        trainer.profile()
    else:
        train_result = trainer.train()
        if train_result is not None:
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)


if __name__ == "__main__":
    main()
